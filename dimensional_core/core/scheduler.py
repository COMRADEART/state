# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from typing import List
from .graph import TaskGraph, Node


class BatchScheduler:
    def __init__(self, batch_size: int = 3) -> None:
        self.batch_size = batch_size

    def choose_batch(self, graph: TaskGraph) -> List[Node]:
        ready = graph.ready_nodes()
        if not ready:
            return []
        # deterministic ordering (replay stability)
        ready = sorted(ready, key=lambda n: n.id)
        return ready[: self.batch_size]