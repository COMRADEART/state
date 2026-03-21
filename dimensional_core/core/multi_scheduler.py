# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from typing import Optional, Dict
from .registry import GraphRegistry
from .z_state import GraphZ


class PriorityScheduler:
    """
    C+7 priority scheduler.
    Chooses graph with smallest (best_score - age_bonus).
    Age bonus prevents starvation.
    """
    def __init__(self) -> None:
        self.z: Dict[str, GraphZ] = {}

    def _get_z(self, gid: str) -> GraphZ:
        if gid not in self.z:
            self.z[gid] = GraphZ()
        return self.z[gid]

    def on_graph_removed(self, gid: str) -> None:
        self.z.pop(gid, None)

    def update_score(self, gid: str, score: float) -> Dict:
        z = self._get_z(gid)
        return z.update(score)

    def choose_graph(self, registry: GraphRegistry) -> Optional[str]:
        ids = registry.active_ids()
        if not ids:
            return None

        # age all graphs first
        for gid in ids:
            self._get_z(gid).age += 1

        best_gid = None
        best_priority = None

        for gid in ids:
            z = self._get_z(gid)

            # lower best_score is better; age provides a bonus
            # tweak 0.001 if you want stronger fairness
            priority = z.best_score - (0.001 * z.age)

            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_gid = gid

        # chosen graph age resets
        if best_gid is not None:
            self._get_z(best_gid).age = 0

        return best_gid