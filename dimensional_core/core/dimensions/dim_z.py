# dimensional_core/core/dimensions/dim_z.py
"""Dimension Z — Verification operator."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import DimensionOperator
from ..visa.vm import VectorVM, VInstruction


class DimZOperator(DimensionOperator):
    """
    Dimension Z: verifies correctness, finiteness, and determinism.

    Dispatches VSCORE_Z which:
    - Checks all scores are finite
    - Checks best score is within threshold
    - Computes a deterministic checksum for replay validation

    If any check fails, ``result["verified"] = False`` and
    ``result["rollback"] = True``, signalling the engine to roll back
    to the most recent X-dimension snapshot.
    """

    dimension = "Z"

    def __init__(self) -> None:
        self._vm = VectorVM()

    def execute(
        self,
        node_id: str,
        warp_id: str,
        instance: Dict[str, Any],
        local: Dict[str, Any],
        lane_gids: List[str],
        args: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        return self._vm.run(
            VInstruction(opcode="VSCORE_Z", args=args),
            warp_id=warp_id,
            node_id=node_id,
            dimension="Z",
            instance=instance,
            local=local,
            lane_gids=lane_gids,
            step=step,
        )
