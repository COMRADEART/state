# dimensional_core/core/dimensions/dim_x.py
"""Dimension X — Initialisation operator."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import DimensionOperator
from ..visa.vm import VectorVM, VInstruction


class DimXOperator(DimensionOperator):
    """
    Dimension X: initialises per-lane computation state.

    Dispatches VINIT_X which loads x-values from the shared instance
    (or falls back to defaults) and seeds the deterministic RNG.
    """

    dimension = "X"

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
            VInstruction(opcode="VINIT_X", args=args),
            warp_id=warp_id,
            node_id=node_id,
            dimension="X",
            instance=instance,
            local=local,
            lane_gids=lane_gids,
            step=step,
        )
