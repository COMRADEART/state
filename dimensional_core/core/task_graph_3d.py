# EXPERIMENTAL — not used by the production engine
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import uuid

Axis = Literal["X", "Y", "Z"]


@dataclass(frozen=True)
class MemoryPoint:
    mp_id: str
    parent_mp_id: Optional[str]
    generation: int
    seq_end: int
    snapshot_ref: str
    state_hash: str
    verified: bool = False
    invalid: bool = False


@dataclass(frozen=True)
class DimEdge:
    edge_id: str
    axis: Axis
    src_mp_id: str
    dst_mp_id: str
    op_name: str
    op_params_hash: str = ""


@dataclass
class TaskGraph3D:
    memory_points: Dict[str, MemoryPoint] = field(default_factory=dict)
    edges: Dict[str, DimEdge] = field(default_factory=dict)
    adj_out: Dict[str, List[str]] = field(default_factory=dict)
    adj_in: Dict[str, List[str]] = field(default_factory=dict)
    root_mp_id: Optional[str] = None

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def add_memory_point(self, mp: MemoryPoint) -> None:
        self.memory_points[mp.mp_id] = mp
        self.adj_out.setdefault(mp.mp_id, [])
        self.adj_in.setdefault(mp.mp_id, [])
        if self.root_mp_id is None:
            self.root_mp_id = mp.mp_id

    def add_edge(self, edge: DimEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.adj_out.setdefault(edge.src_mp_id, []).append(edge.edge_id)
        self.adj_in.setdefault(edge.dst_mp_id, []).append(edge.edge_id)

    def get_children(self, mp_id: str) -> List[str]:
        out_edges = self.adj_out.get(mp_id, [])
        children = []
        for eid in out_edges:
            children.append(self.edges[eid].dst_mp_id)
        return children

    def descendants(self, mp_id: str) -> List[str]:
        # BFS
        seen = set()
        q = [mp_id]
        out = []
        while q:
            cur = q.pop(0)
            for child in self.get_children(cur):
                if child in seen:
                    continue
                seen.add(child)
                out.append(child)
                q.append(child)
        return out