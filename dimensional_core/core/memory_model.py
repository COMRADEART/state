from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import time

from .l2_cache import SharedL2Cache, L2Stats


@dataclass
class MemStats:
    hits: int = 0
    misses: int = 0
    bytes: int = 0
    lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "lines": self.lines}


@dataclass
class MemoryModel:
    """
    L1 per-warp cache + optional Shared L2.
    L1 hit => no L2 access.
    L1 miss => consult L2; L2 miss => counts as global memory bytes.
    """
    line_buckets: int = 64
    ttl_seconds: float = 1.5
    line_bytes: int = 128
    _line_last: Dict[int, float] = field(default_factory=dict)

    def _line_id(self, key: str) -> int:
        return (hash(key) & 0x7FFFFFFF) % self.line_buckets

    def access_keys(self, keys: List[str], l2: SharedL2Cache | None = None) -> Tuple[MemStats, L2Stats | None]:
        now = time.time()
        line_ids = [self._line_id(k) for k in keys]
        unique_lines = sorted(set(line_ids))

        l1_hits = 0
        l1_misses = 0
        # determine which keys miss in L1 (so they go to L2)
        miss_keys: List[str] = []

        for k in keys:
            lid = self._line_id(k)
            last = self._line_last.get(lid)
            if last is not None and (now - last) <= self.ttl_seconds:
                l1_hits += 1
            else:
                l1_misses += 1
                self._line_last[lid] = now
                miss_keys.append(k)

        # L1 bytes count = misses * line_bytes (simple model)
        l1 = MemStats(hits=l1_hits, misses=l1_misses, bytes=l1_misses * self.line_bytes, lines=len(unique_lines))

        l2stats = None
        if l2 is not None and miss_keys:
            l2stats = l2.access_lines(miss_keys)

        return l1, l2stats
