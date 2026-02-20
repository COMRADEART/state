from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import time


@dataclass
class L2Stats:
    hits: int = 0
    misses: int = 0
    bytes: int = 0
    lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"l2_hits": self.hits, "l2_misses": self.misses, "l2_bytes": self.bytes, "l2_lines": self.lines}


@dataclass
class SharedL2Cache:
    """
    Shared across all warps.
    Similar to L1 but bigger TTL and line_buckets.
    """
    line_buckets: int = 256
    ttl_seconds: float = 4.0
    line_bytes: int = 128
    _line_last: Dict[int, float] = field(default_factory=dict)

    def _line_id(self, key: str) -> int:
        return (hash(key) & 0x7FFFFFFF) % self.line_buckets

    def access_lines(self, keys: List[str]) -> L2Stats:
        now = time.time()
        line_ids = [self._line_id(k) for k in keys]
        unique_lines = sorted(set(line_ids))

        hits = 0
        misses = 0
        for lid in unique_lines:
            last = self._line_last.get(lid)
            if last is not None and (now - last) <= self.ttl_seconds:
                hits += 1
            else:
                misses += 1
                self._line_last[lid] = now

        return L2Stats(
            hits=hits,
            misses=misses,
            bytes=misses * self.line_bytes,
            lines=len(unique_lines),
        )
