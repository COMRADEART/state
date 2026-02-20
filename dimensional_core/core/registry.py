from __future__ import annotations
from typing import Dict, Optional, Any
from .graph import TaskGraph


class GraphRegistry:
    def __init__(self) -> None:
        self.graphs: Dict[str, TaskGraph] = {}

    def add(self, gid: str, g: TaskGraph) -> None:
        self.graphs[gid] = g

    def get(self, gid: str) -> Optional[TaskGraph]:
        return self.graphs.get(gid)

    def remove(self, gid: str) -> None:
        if gid in self.graphs:
            del self.graphs[gid]

    def active_ids(self):
        return list(self.graphs.keys())

    def is_empty(self) -> bool:
        return len(self.graphs) == 0

    # --------- C+6 serialize ---------

    def to_dict(self) -> Dict[str, Any]:
        return {gid: g.to_dict() for gid, g in self.graphs.items()}

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        self.graphs = {gid: TaskGraph.from_dict(gd) for gid, gd in d.items()}
