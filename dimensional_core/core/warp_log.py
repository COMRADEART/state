# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, Iterable, Optional


class WarpLog:
    """
    Per-warp micro log:
      dimensional_core/state/warp_log_W0.jsonl
      dimensional_core/state/warp_log_W1.jsonl
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, warp: str) -> str:
        return os.path.join(self.root_dir, f"warp_log_{warp}.jsonl")

    def append(self, warp: str, rec: Dict[str, Any]) -> None:
        path = self._path(warp)
        rec = dict(rec)
        rec["_ts"] = time.time()
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)

        # append-only is very stable on Windows
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def iter_tail(self, warp: str, max_lines: int = 200) -> Iterable[Dict[str, Any]]:
        """
        Read last N lines (simple, safe). For large logs, this is still OK for MVP.
        """
        path = self._path(warp)
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max_lines:]

        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    def clear(self, warp: str) -> None:
        path = self._path(warp)
        if os.path.exists(path):
            os.remove(path)