# dimensional_core/core/priority_scheduler.py
from __future__ import annotations

import heapq
from typing import Dict, List, Tuple


class HeapWarpScheduler:
    """
    C21: Heap-based warp scheduler.

    priority = score + age_bonus
    - lower priority value is chosen first
    - age increases over time to prevent starvation
    """

    def __init__(self, age_bonus: float = 0.001) -> None:
        self.age_bonus = float(age_bonus)
        self.scores: Dict[str, float] = {}
        self.ages: Dict[str, int] = {}
        self._heap: List[Tuple[float, int, str]] = []
        self._tick = 0

    def register(self, wid: str) -> None:
        if wid not in self.scores:
            self.scores[wid] = 0.0
        if wid not in self.ages:
            self.ages[wid] = 0
        self._push(wid)

    def priority(self, wid: str) -> float:
        score = float(self.scores.get(wid, 0.0))
        age = int(self.ages.get(wid, 0))
        # age reduces effective priority to prevent starvation
        return score - self.age_bonus * age

    def update_score(self, wid: str, score: float) -> None:
        self.scores[wid] = float(score)
        self.ages[wid] = 0
        self._push(wid)

    def _push(self, wid: str) -> None:
        self._tick += 1
        heapq.heappush(self._heap, (self.priority(wid), self._tick, wid))

    def choose(self) -> str | None:
        """
        Pop until we find a current (non-stale) entry.
        """
        while self._heap:
            pr, _, wid = heapq.heappop(self._heap)
            # If this entry matches current computed priority, accept it
            cur = self.priority(wid)
            if abs(cur - pr) < 1e-12:
                # aging: everyone else gets older when one is chosen
                for k in list(self.ages.keys()):
                    if k != wid:
                        self.ages[k] = self.ages.get(k, 0) + 1
                        self._push(k)
                return wid
        return None
