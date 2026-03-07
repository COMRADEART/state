from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class WarpNode:
    id: str
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"   # PENDING | READY | RUNNING | DONE
    result: Any = None


class WarpGraph:
    """
    Minimal warp execution graph.

    Expected flow:
        init -> step -> score

    The engine re-arms step/score after score completes, so the graph becomes:
        init -> step -> score -> step -> score -> ...
    """

    def __init__(self, wid: str) -> None:
        self.wid = wid
        self.nodes: Dict[str, WarpNode] = {}
        self.deps: Dict[str, Set[str]] = {}
        self.revdeps: Dict[str, Set[str]] = {}
        self.ready: Set[str] = set()
        self._local: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: WarpNode) -> None:
        self.nodes[node.id] = node
        self.deps.setdefault(node.id, set())
        self.revdeps.setdefault(node.id, set())

    def add_edge(self, src: str, dst: str) -> None:
        self.deps.setdefault(dst, set()).add(src)
        self.revdeps.setdefault(src, set()).add(dst)

    def finalize(self) -> None:
        self.ready.clear()
        for nid, node in self.nodes.items():
            if node.status == "PENDING" and not self.deps.get(nid):
                self.ready.add(nid)

    # ------------------------------------------------------------------
    # runtime operations
    # ------------------------------------------------------------------

    def take_ready(self, limit: int = 1) -> List[WarpNode]:
        """
        Returns up to `limit` ready nodes and marks them RUNNING.
        """
        if limit <= 0:
            return []

        chosen_ids = sorted(self.ready)[:limit]
        out: List[WarpNode] = []

        for nid in chosen_ids:
            self.ready.discard(nid)
            node = self.nodes[nid]
            node.status = "RUNNING"
            out.append(node)

        return out

    def mark_done(self, nid: str, result: Dict[str, Any]) -> None:
        """
        Marks node done and releases dependent nodes whose deps are all DONE.
        """
        if nid not in self.nodes:
            return

        node = self.nodes[nid]
        node.status = "DONE"
        node.result = result

        for child_id in self.revdeps.get(nid, set()):
            child = self.nodes.get(child_id)
            if child is None or child.status != "PENDING":
                continue

            parents = self.deps.get(child_id, set())
            all_done = True
            for pid in parents:
                p = self.nodes.get(pid)
                if p is None or p.status != "DONE":
                    all_done = False
                    break

            if all_done:
                self.ready.add(child_id)

    def rearm(self, nid: str) -> None:
        """
        Resets a node back to PENDING so it can run again.
        Used by engine for cyclic step/score execution.
        """
        if nid not in self.nodes:
            return

        node = self.nodes[nid]
        node.status = "PENDING"
        node.result = None

        parents = self.deps.get(nid, set())
        all_done = True
        for pid in parents:
            p = self.nodes.get(pid)
            if p is None or p.status != "DONE":
                all_done = False
                break

        if all_done:
            self.ready.add(nid)

    # ------------------------------------------------------------------
    # debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "wid": self.wid,
            "nodes": {
                nid: {
                    "status": node.status,
                    "op": node.op,
                    "params": dict(node.params),
                }
                for nid, node in self.nodes.items()
            },
            "ready": sorted(self.ready),
        }


def build_warp_graph(
    wid: str,
    lane_gids: Iterable[str] | None,
    shared_instance: Dict[str, Any] | None,
    lanes: int = 1,
) -> WarpGraph:
    """
    Build a simple persistent compute graph for one warp.

    Node ids:
        <wid>:init
        <wid>:step
        <wid>:score

    Engine _compile_task() expects op='VISA' and params['stage'] in {'init','step','score'}.
    """
    lane_gids = list(lane_gids or [])
    shared_instance = shared_instance or {}

    g = WarpGraph(wid)

    init_node = WarpNode(
        id=f"{wid}:init",
        op="VISA",
        params={
            "stage": "init",
            "lane_gids": lane_gids,
        },
    )

    step_node = WarpNode(
        id=f"{wid}:step",
        op="VISA",
        params={
            "stage": "step",
            "lane_gids": lane_gids,
            "lr": 0.10,
        },
    )

    score_node = WarpNode(
        id=f"{wid}:score",
        op="VISA",
        params={
            "stage": "score",
            "lane_gids": lane_gids,
        },
    )

    g.add_node(init_node)
    g.add_node(step_node)
    g.add_node(score_node)

    # init must happen first
    g.add_edge(f"{wid}:init", f"{wid}:step")

    # step produces score
    g.add_edge(f"{wid}:step", f"{wid}:score")

    g.finalize()
    return g