from __future__ import annotations

import heapq
import threading
from typing import Dict, List, Tuple

_DEFAULT_AGE_BONUS = 0.001      # score reduction per epoch since last served
_HEAP_COMPACTION_FACTOR = 4     # compact when heap > active_warps * this
_PRIORITY_TOLERANCE = 1e-12     # treat priority differences below this as equal


class HeapWarpScheduler:
    """
    Heap scheduler without O(n) re-push on every dispatch.

    Effective priority:
        score - age_bonus * (epoch - last_served_epoch)

    Lower value wins.

    Notes:
    - Aging is computed lazily from a global epoch.
    - Heap entries are versioned; stale entries are discarded on pop.
    - Heap is compacted periodically so it does not grow without bound.
    """

    def __init__(self, age_bonus: float = _DEFAULT_AGE_BONUS) -> None:
        self.age_bonus = float(age_bonus)

        self.scores: Dict[str, float] = {}
        self.last_served_epoch: Dict[str, int] = {}
        self._versions: Dict[str, int] = {}
        self._heap: List[Tuple[float, int, str]] = []
        self._epoch = 0

        self._lock = threading.Lock()

    def register(self, wid: str) -> None:
        with self._lock:
            if wid not in self.scores:
                self.scores[wid] = 0.0
            if wid not in self.last_served_epoch:
                self.last_served_epoch[wid] = self._epoch
            self._versions[wid] = self._versions.get(wid, 0) + 1
            self._push_locked(wid)

    def unregister(self, wid: str) -> None:
        with self._lock:
            self.scores.pop(wid, None)
            self.last_served_epoch.pop(wid, None)
            self._versions.pop(wid, None)
            # stale heap entries are ignored later

    def priority(self, wid: str) -> float:
        with self._lock:
            return self._priority_locked(wid)

    def update_score(self, wid: str, score: float) -> None:
        with self._lock:
            if wid not in self.scores:
                self.scores[wid] = 0.0
                self.last_served_epoch[wid] = self._epoch
            self.scores[wid] = float(score)
            self._versions[wid] = self._versions.get(wid, 0) + 1
            self._push_locked(wid)
            self._maybe_compact_locked()

    def choose(self) -> str | None:
        with self._lock:
            self._epoch += 1

            while self._heap:
                pr, ver, wid = heapq.heappop(self._heap)

                if wid not in self.scores:
                    continue  # unregistered

                if ver != self._versions.get(wid, -1):
                    continue  # stale entry

                cur = self._priority_locked(wid)
                if abs(cur - pr) > _PRIORITY_TOLERANCE:
                    # entry is outdated due to lazy aging; refresh it
                    self._versions[wid] = self._versions.get(wid, 0) + 1
                    self._push_locked(wid)
                    continue

                # chosen warp: reset its age anchor to "now"
                self.last_served_epoch[wid] = self._epoch
                self._versions[wid] = self._versions.get(wid, 0) + 1
                self._push_locked(wid)
                self._maybe_compact_locked()
                return wid

            return None

    def _priority_locked(self, wid: str) -> float:
        score = float(self.scores.get(wid, 0.0))
        last_served = int(self.last_served_epoch.get(wid, self._epoch))
        age = max(0, self._epoch - last_served)
        return score - self.age_bonus * age

    def _push_locked(self, wid: str) -> None:
        ver = self._versions.get(wid, 0)
        heapq.heappush(self._heap, (self._priority_locked(wid), ver, wid))

    def _maybe_compact_locked(self) -> None:
        active = max(1, len(self.scores))
        if len(self._heap) <= active * _HEAP_COMPACTION_FACTOR:
            return

        fresh: List[Tuple[float, int, str]] = []
        for wid in self.scores.keys():
            ver = self._versions.get(wid, 0)
            fresh.append((self._priority_locked(wid), ver, wid))
        heapq.heapify(fresh)
        self._heap = fresh