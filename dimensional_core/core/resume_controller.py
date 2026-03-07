from __future__ import annotations

from typing import Any, Dict


class ResumeController:
    """
    Normalizes old and new checkpoint formats.

    Canonical key going forward: warps_state
    Backward-compatible legacy key: graphs_state
    """

    def resolve(self, instance_point: Dict[str, Any] | None) -> Dict[str, Any]:
        if not instance_point:
            return {
                "global_step": 0,
                "cycle": 0,
                "last_eid": 0,
                "instance": {},
                "warps_state": {},
                "graphs_state": {},  # legacy alias for older code/tools
            }

        inst = instance_point.get("instance", {})
        if not isinstance(inst, dict):
            inst = {}

        warps_state = instance_point.get("warps_state")
        graphs_state = instance_point.get("graphs_state")

        if not isinstance(warps_state, dict):
            warps_state = {}
        if not isinstance(graphs_state, dict):
            graphs_state = {}

        # Prefer canonical warps_state; fall back to legacy graphs_state.
        merged_warps_state = warps_state or graphs_state

        return {
            "global_step": int(instance_point.get("global_step", 0) or 0),
            "cycle": int(instance_point.get("cycle", 0) or 0),
            "last_eid": int(instance_point.get("last_eid", 0) or 0),
            "instance": inst,
            "warps_state": merged_warps_state,
            "graphs_state": merged_warps_state,  # alias for compatibility
        }