# EXPERIMENTAL — not used by the production engine
# dimensional_core/core/graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .ready_set import ReadySet


@dataclass
class Node:
    id: str
    op: str
    params: Dict[str, Any]
    status: str = "PENDING"  # PENDING | QUEUED | RUNNING | DONE
    result: Optional[Dict[str, Any]] = None


class TaskGraph:
    """
    C21 TaskGraph:
      - nodes: id -> Node
      - deps:  id -> set(of prerequisite node ids)
      - children: id -> list(of downstream node ids)
      - ready: ReadySet of runnable node ids (deps satisfied)
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.deps: Dict[str, Set[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.ready = ReadySet()
        self.entry: str | None = None

        # Per-warp persistent local state is attached by Engine / warp factory
        # self._local = {}

    # ---------------------------
    # build
    # ---------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.deps.setdefault(node.id, set())
        self.children.setdefault(node.id, [])
        # if no deps, it is runnable
        if not self.deps[node.id] and node.status == "PENDING":
            self.ready.add(node.id)

    def add_edge(self, a: str, b: str) -> None:
        """
        dependency a -> b (b depends on a)
        """
        self.deps.setdefault(a, set())
        self.deps.setdefault(b, set()).add(a)
        self.children.setdefault(a, []).append(b)
        self.children.setdefault(b, [])
        # b is not ready until deps clear
        # (no need to remove from ready set; take_ready validates status + deps)

    # ---------------------------
    # scheduling
    # ---------------------------
    def take_ready(self, k: int = 1) -> List[Node]:
        """
        Pops up to k runnable nodes.
        Marks them QUEUED to prevent re-scheduling duplicates.
        """
        out: List[Node] = []
        for nid in self.ready.pop_many(k):
            n = self.nodes.get(nid)
            if n is None:
                continue

            # validate still runnable
            if n.status != "PENDING":
                continue
            if self.deps.get(nid):
                # deps reintroduced or not cleared -> skip
                continue

            n.status = "QUEUED"
            out.append(n)
        return out

    def ready_count(self) -> int:
        return len(self.ready)

    # ---------------------------
    # completion
    # ---------------------------
    def mark_done(self, node_id: str, result: Dict[str, Any]) -> None:
        n = self.nodes.get(node_id)
        if n is None:
            return
        n.status = "DONE"
        n.result = dict(result)

        # release children
        for child in self.children.get(node_id, []):
            ds = self.deps.get(child)
            if not ds:
                continue
            ds.discard(node_id)
            if not ds:
                cn = self.nodes.get(child)
                if cn is not None and cn.status == "PENDING":
                    self.ready.add(child)

    # ---------------------------
    # loop helpers
    # ---------------------------
    def rearm(self, node_id: str) -> None:
        """
        Reset a node back to PENDING and make it runnable if deps cleared.
        (Used for warp loops: step/score)
        """
        n = self.nodes.get(node_id)
        if n is None:
            return
        n.status = "PENDING"
        # deps should already be empty in your loop graph
        if not self.deps.get(node_id):
            self.ready.add(node_id)