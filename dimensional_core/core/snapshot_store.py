# EXPERIMENTAL — not used by the production engine
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class SnapshotStore:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _dim_dir(self, dimension: str) -> Path:
        d = self.base_dir / dimension
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_json(self, dimension: str, state: Dict[str, Any]) -> tuple[str, str]:
        # canonical JSON: stable hashes
        b = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = sha256_bytes(b)
        path = self._dim_dir(dimension) / f"{h}.json"
        if not path.exists():
            path.write_bytes(b)
        return str(path), h

    def load_json(self, snapshot_ref: str) -> Dict[str, Any]:
        p = Path(snapshot_ref)
        b = p.read_bytes()
        return json.loads(b.decode("utf-8"))

    def verify_hash(self, snapshot_ref: str, expected_hash: str) -> None:
        p = Path(snapshot_ref)
        b = p.read_bytes()
        actual = sha256_bytes(b)
        if actual != expected_hash:
            raise ValueError(f"Snapshot hash mismatch: expected={expected_hash} actual={actual}")