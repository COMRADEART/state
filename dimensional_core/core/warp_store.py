# dimensional_core/core/warp_store.py
from __future__ import annotations

import json
import os
import time
import uuid

_WRITE_RETRY_COUNT = 3      # max retries for atomic warp-point write
_WRITE_RETRY_SLEEP_S = 0.05 # base sleep for write retries (doubles each attempt)


class WarpStore:
    """
    Saves/loads warp instance points:
      state/warp_point_W0.json
      state/warp_point_W1.json

    Windows-safe write:
    - write to unique temp file
    - flush + fsync
    - try os.replace
    - if blocked, fall back to delete+rename
    """

    def __init__(self, root_dir: str = "dimensional_core/state"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, warp_id: str) -> str:
        return os.path.join(self.root_dir, f"warp_point_{warp_id}.json")

    def save(self, warp_id: str, data: dict) -> str:
        path = self._path(warp_id)
        tmp = f"{path}.tmp.{uuid.uuid4().hex}"

        payload = dict(data)
        payload.setdefault("_ts", time.time())
        payload.setdefault("warp", warp_id)

        # write temp
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # atomic replace with exponential backoff (Windows may hold file locks briefly)
        for attempt in range(_WRITE_RETRY_COUNT):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                if attempt < _WRITE_RETRY_COUNT - 1:
                    time.sleep(_WRITE_RETRY_SLEEP_S * (2 ** attempt))

        # All os.replace attempts failed; try delete+rename as last resort
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        return path

    def load(self, warp_id: str) -> dict | None:
        path = self._path(warp_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
