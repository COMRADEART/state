# dimensional_core/core/task_graph.py
"""
3D Task Graph — the node/edge model for one warp's execution.

Graph structure (cyclic, 3 persistent nodes per warp)
------------------------------------------------------

    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
  W0:X  ──►  W0:Y  ──►  W0:Z                             │
  (init)    (optimise)  (verify)                          │
                           │                              │
                           │  verified=True ──────────────┘  (rearm X→Y→Z)
                           │
                           │  verified=False ── rollback ──► restore X-snapshot
                                                             reset X to PENDING

Node state machine
------------------
  PENDING → RUNNING → DONE
                    ↘ FAILED → (rollback) → PENDING
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ── Enumerations ────────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    PENDING     = "PENDING"
    RUNNING     = "RUNNING"
    DONE        = "DONE"
    FAILED      = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class Dimension(str, Enum):
    X = "X"   # Initialisation
    Y = "Y"   # Optimisation / Transformation
    Z = "Z"   # Verification


# ── Memory point (graph node) ───────────────────────────────────────────────────

@dataclass
class MemoryPoint:
    """
    A node in the 3D task graph representing one memory state.

    Fields
    ------
    id                  : unique node identifier (e.g. "W0:Y")
    dimension           : which execution layer this node belongs to
    op                  : VISA opcode family ("VISA")
    params              : per-node arguments forwarded to the dimension operator
    status              : current execution status
    result              : output dict after execution (None until DONE)
    instance_snapshot   : copy of instance state at node creation (for rollback)
    transition_meta     : engine bookkeeping (timestamps, attempt counts, etc.)
    """

    id: str
    dimension: Dimension
    op: str = "VISA"
    params: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    instance_snapshot: Dict[str, Any] = field(default_factory=dict)
    transition_meta: Dict[str, Any] = field(default_factory=dict)


# ── 3D Task Graph ───────────────────────────────────────────────────────────────

class TaskGraph3D:
    """
    Cyclic 3D execution graph for a single warp.

    Three persistent nodes: W<n>:X, W<n>:Y, W<n>:Z.
    Forward edges:  X → Y → Z
    Backward edges: automatically tracked for rollback traversal.

    The graph does NOT create new nodes per cycle — it re-arms the same three
    nodes, keeping memory usage O(1) per warp regardless of run length.

    Local state
    -----------
    ``_local``          : live computation state (dict of dicts — V, S, …)
    ``_rollback_local`` : deep-copy of _local saved after X completes,
                          restored when Z fails.

    Lock requirements
    -----------------
    The caller (engine) holds its own RLock when mutating this graph.
    The graph itself is NOT thread-safe and must not be shared raw.
    """

    def __init__(self, wid: str) -> None:
        self.wid = wid
        self.nodes: Dict[str, MemoryPoint] = {}
        self.forward_edges: Dict[str, Set[str]] = {}
        self.backward_edges: Dict[str, Set[str]] = {}
        self.ready: Set[str] = set()
        self._local: Dict[str, Any] = {}
        self._rollback_local: Optional[Dict[str, Any]] = None

        # Metrics
        self.verify_pass_count: int = 0
        self.verify_fail_count: int = 0

    # ── Graph construction ─────────────────────────────────────────────────────

    def add_node(self, node: MemoryPoint) -> None:
        self.nodes[node.id] = node
        self.forward_edges.setdefault(node.id, set())
        self.backward_edges.setdefault(node.id, set())

    def add_edge(self, src: str, dst: str) -> None:
        """Add a directed edge src → dst."""
        self.forward_edges.setdefault(src, set()).add(dst)
        self.backward_edges.setdefault(dst, set()).add(src)

    def finalize(self) -> None:
        """Seed the ready set with all PENDING nodes that have no dependencies."""
        self.ready.clear()
        for nid, node in self.nodes.items():
            if node.status == NodeStatus.PENDING and not self.backward_edges.get(nid):
                self.ready.add(nid)

    # ── Runtime operations ─────────────────────────────────────────────────────

    def take_ready(self, limit: int = 1) -> List[MemoryPoint]:
        """
        Return up to *limit* ready nodes, marking each RUNNING.
        Stable sort ensures deterministic dispatch order.
        """
        if limit <= 0:
            return []
        chosen = sorted(self.ready)[:limit]
        out: List[MemoryPoint] = []
        for nid in chosen:
            self.ready.discard(nid)
            node = self.nodes[nid]
            node.status = NodeStatus.RUNNING
            out.append(node)
        return out

    def mark_done(self, nid: str, result: Dict[str, Any]) -> None:
        """
        Mark *nid* DONE, store its result, and release any newly-ready
        forward neighbours.
        """
        node = self.nodes.get(nid)
        if node is None:
            return
        node.status = NodeStatus.DONE
        node.result = dict(result)

        for child_id in self.forward_edges.get(nid, set()):
            child = self.nodes.get(child_id)
            if child is None or child.status != NodeStatus.PENDING:
                continue
            # Release child only when all its parents are DONE
            parents = self.backward_edges.get(child_id, set())
            if all(
                self.nodes.get(p) is not None
                and self.nodes[p].status == NodeStatus.DONE
                for p in parents
            ):
                self.ready.add(child_id)

    def mark_failed(self, nid: str, reason: str) -> None:
        """Mark *nid* FAILED (used internally before rollback)."""
        node = self.nodes.get(nid)
        if node is not None:
            node.status = NodeStatus.FAILED
            node.result = {"error": reason}

    # ── Rollback ───────────────────────────────────────────────────────────────

    def save_rollback_snapshot(self) -> None:
        """
        Save a deep-copy of _local after Dimension X completes.
        This is the state we restore if Dimension Z fails.
        """
        self._rollback_local = copy.deepcopy(self._local)

    def rollback(self) -> str:
        """
        Roll back the current cycle on Z verification failure.

        Steps:
        1. Mark Y and Z as ROLLED_BACK.
        2. Restore _local from the X-snapshot.
        3. Reset X to PENDING and add to ready set so the cycle re-runs.

        Returns the X node ID (the re-run starting point).
        """
        self.verify_fail_count += 1

        x_nid = f"{self.wid}:X"
        y_nid = f"{self.wid}:Y"
        z_nid = f"{self.wid}:Z"

        for nid in (y_nid, z_nid):
            node = self.nodes.get(nid)
            if node is not None:
                node.status = NodeStatus.ROLLED_BACK
                node.result = None

        # Restore pre-Y computation state
        if self._rollback_local is not None:
            self._local = copy.deepcopy(self._rollback_local)

        # Re-arm X so the cycle re-runs from initialisation
        x_node = self.nodes.get(x_nid)
        if x_node is not None:
            x_node.status = NodeStatus.PENDING
            x_node.result = None
            self.ready.add(x_nid)

        return x_nid

    # ── Cyclic re-arm ──────────────────────────────────────────────────────────

    def rearm_cycle(self) -> None:
        """
        After a successful Z verification, reset all three nodes to PENDING
        so the next X→Y→Z cycle can begin.
        """
        self.verify_pass_count += 1

        x_nid = f"{self.wid}:X"
        y_nid = f"{self.wid}:Y"
        z_nid = f"{self.wid}:Z"

        for nid in (y_nid, z_nid):
            node = self.nodes.get(nid)
            if node is not None:
                node.status = NodeStatus.PENDING
                node.result = None

        x_node = self.nodes.get(x_nid)
        if x_node is not None:
            x_node.status = NodeStatus.PENDING
            x_node.result = None
            self.ready.add(x_nid)

    # ── Inspection ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "wid": self.wid,
            "nodes": {
                nid: {
                    "status": node.status,
                    "dimension": node.dimension,
                    "result_keys": list((node.result or {}).keys()),
                }
                for nid, node in self.nodes.items()
            },
            "ready": sorted(self.ready),
            "verify_pass": self.verify_pass_count,
            "verify_fail": self.verify_fail_count,
        }


# ── Factory ─────────────────────────────────────────────────────────────────────

_DEFAULT_LR = 0.10


def build_3d_graph(
    wid: str,
    lane_gids: List[str],
    shared_instance: Dict[str, Any],
    lr: float = _DEFAULT_LR,
    y_opcode: str = "VSTEP_Y",
) -> TaskGraph3D:
    """
    Construct a 3-node cyclic X→Y→Z task graph for warp *wid*.

    Parameters
    ----------
    wid            : warp identifier (e.g. "W0")
    lane_gids      : logical lane IDs assigned to this warp
    shared_instance: reference to the shared instance dict
    lr             : gradient-descent learning rate for Dimension Y
    y_opcode       : VISA opcode for Y ("VSTEP_Y" or "VMUTATE_Y")

    Node IDs: ``{wid}:X``, ``{wid}:Y``, ``{wid}:Z``
    """
    lane_gids = list(lane_gids or [])
    g = TaskGraph3D(wid)

    g.add_node(MemoryPoint(
        id=f"{wid}:X",
        dimension=Dimension.X,
        op="VISA",
        params={"stage": "X", "lane_gids": lane_gids},
    ))

    g.add_node(MemoryPoint(
        id=f"{wid}:Y",
        dimension=Dimension.Y,
        op="VISA",
        params={"stage": "Y", "lane_gids": lane_gids, "lr": lr, "y_opcode": y_opcode},
    ))

    g.add_node(MemoryPoint(
        id=f"{wid}:Z",
        dimension=Dimension.Z,
        op="VISA",
        params={"stage": "Z", "lane_gids": lane_gids},
    ))

    g.add_edge(f"{wid}:X", f"{wid}:Y")
    g.add_edge(f"{wid}:Y", f"{wid}:Z")

    g.finalize()
    return g
