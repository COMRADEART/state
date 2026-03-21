# EXPERIMENTAL — not used by the production engine
# dimensional_core/core/ready_set.py
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Iterable, List


class ReadySet:
    """
    O(1) ready queue with de-dup.

    - add(id): enqueue if not already queued
    - pop(): FIFO pop
    - pop_many(k): pop up to k
    """

    def __init__(self) -> None:
        self._q: Deque[str] = deque()
        self._in: Dict[str, bool] = {}

    def __len__(self) -> int:
        return len(self._q)

    def add(self, node_id: str) -> None:
        if self._in.get(node_id):
            return
        self._in[node_id] = True
        self._q.append(node_id)

    def add_many(self, ids: Iterable[str]) -> None:
        for i in ids:
            self.add(i)

    def pop(self) -> str | None:
        if not self._q:
            return None
        nid = self._q.popleft()
        # allow re-add later
        self._in.pop(nid, None)
        return nid

    def pop_many(self, k: int) -> List[str]:
        out: List[str] = []
        k = max(0, int(k))
        for _ in range(k):
            nid = self.pop()
            if nid is None:
                break
            out.append(nid)
        return out