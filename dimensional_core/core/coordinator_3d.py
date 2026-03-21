# EXPERIMENTAL — not used by the production engine
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import threading

from .task_graph_3d import TaskGraph3D, MemoryPoint, DimEdge
from .snapshot_store import SnapshotStore


@dataclass
class CommitResult:
    mp_id: str
    snapshot_ref: str
    state_hash: str
    generation: int
    seq_end: int


class Coordinator3D:
    def __init__(self, store: SnapshotStore) -> None:
        self.store = store
        self.graph = TaskGraph3D()
        self._lock = threading.Lock()

    def bootstrap_root(self, state: Dict[str, Any], seq_end: int = 0) -> CommitResult:
        with self._lock:
            snapshot_ref, h = self.store.save_json("ROOT", state)
            mp_id = self.graph.new_id("MP")
            mp = MemoryPoint(
                mp_id=mp_id,
                parent_mp_id=None,
                generation=0,
                seq_end=seq_end,
                snapshot_ref=snapshot_ref,
                state_hash=h,
                verified=True,   # root treated as verified
                invalid=False,
            )
            self.graph.add_memory_point(mp)
            return CommitResult(mp_id, snapshot_ref, h, 0, seq_end)

    def commit_next(
        self,
        axis: str,
        parent_mp_id: str,
        state: Dict[str, Any],
        seq_end: int,
        op_name: str,
        op_params_hash: str = "",
        verified: bool = False,
    ) -> CommitResult:
        with self._lock:
            parent = self.graph.memory_points[parent_mp_id]
            snapshot_ref, h = self.store.save_json(axis, state)
            mp_id = self.graph.new_id("MP")
            mp = MemoryPoint(
                mp_id=mp_id,
                parent_mp_id=parent_mp_id,
                generation=parent.generation + 1,
                seq_end=seq_end,
                snapshot_ref=snapshot_ref,
                state_hash=h,
                verified=verified,
                invalid=False,
            )
            self.graph.add_memory_point(mp)

            eid = self.graph.new_id("E")
            edge = DimEdge(
                edge_id=eid,
                axis=axis,  # type: ignore
                src_mp_id=parent_mp_id,
                dst_mp_id=mp_id,
                op_name=op_name,
                op_params_hash=op_params_hash,
            )
            self.graph.add_edge(edge)

            return CommitResult(mp_id, snapshot_ref, h, mp.generation, seq_end)

    def mark_verified(self, mp_id: str) -> None:
        with self._lock:
            mp = self.graph.memory_points[mp_id]
            self.graph.memory_points[mp_id] = MemoryPoint(
                mp_id=mp.mp_id,
                parent_mp_id=mp.parent_mp_id,
                generation=mp.generation,
                seq_end=mp.seq_end,
                snapshot_ref=mp.snapshot_ref,
                state_hash=mp.state_hash,
                verified=True,
                invalid=mp.invalid,
            )

    def invalidate_descendants(self, mp_id: str) -> None:
        with self._lock:
            for d in self.graph.descendants(mp_id):
                mp = self.graph.memory_points[d]
                if mp.invalid:
                    continue
                self.graph.memory_points[d] = MemoryPoint(
                    mp_id=mp.mp_id,
                    parent_mp_id=mp.parent_mp_id,
                    generation=mp.generation,
                    seq_end=mp.seq_end,
                    snapshot_ref=mp.snapshot_ref,
                    state_hash=mp.state_hash,
                    verified=False,
                    invalid=True,
                )

    def latest_verified_ancestor(self, mp_id: str) -> str:
        with self._lock:
            cur = mp_id
            while True:
                mp = self.graph.memory_points[cur]
                if mp.verified and not mp.invalid:
                    return cur
                if mp.parent_mp_id is None:
                    return cur
                cur = mp.parent_mp_id