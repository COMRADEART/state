# dimensional_core/core/batch_scheduler.py
from __future__ import annotations
from typing import List

from .graph import TaskGraph, Node


class BatchScheduler:
    """
    Picks a small batch of READY nodes from a graph.

    READY means:
    - status == "PENDING"
    - all deps are DONE
    """

    def __init__(self, max_batch: int = 2):
        self.max_batch = int(max_batch)

    def choose_batch(self, g: TaskGraph) -> List[Node]:
        ready: List[Node] = []

        # stable order so runs look consistent
        for node_id in sorted(g.nodes.keys()):
            n = g.nodes[node_id]
            if getattr(n, "status", None) != "PENDING":
                continue

            deps = getattr(n, "deps", []) or []
            ok = True
            for d in deps:
                dn = g.nodes.get(d)
                if dn is None or getattr(dn, "status", None) != "DONE":
                    ok = False
                    break

            if ok:
                ready.append(n)

            if len(ready) >= self.max_batch:
                break

        return ready
