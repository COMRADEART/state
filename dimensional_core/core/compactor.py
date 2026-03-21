# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, Optional


class Compactor:
    def __init__(self, compact_point_path: str = "dimensional_core/state/compact_point.json"):
        self.path = compact_point_path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, compact_eid: int, meta: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        data = {"compact_eid": int(compact_eid), "meta": dict(meta), "timestamp": time.time()}
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)