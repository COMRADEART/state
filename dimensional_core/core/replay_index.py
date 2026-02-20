# dimensional_core/core/replay_index.py
from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional


class ReplayIndex:
    """
    Stores a fast cursor for replay:
      - last_eid
      - byte_offset into events.jsonl

    File: dimensional_core/state/replay_index.json
    """

    def __init__(self, path: str = "dimensional_core/state/replay_index.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> Dict[str, Any] | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, last_eid: int, byte_offset: int) -> None:
        data = {"last_eid": int(last_eid), "byte_offset": int(byte_offset)}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
