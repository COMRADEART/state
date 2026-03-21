# dimensional_core/core/dimensions/dim_y.py
"""Dimension Y — Optimisation operator."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import DimensionOperator
from ..visa.vm import VectorVM, VInstruction

# Default Y opcode; can be overridden per-node via args["y_opcode"]
_DEFAULT_Y_OPCODE = "VSTEP_Y"


class DimYOperator(DimensionOperator):
    """
    Dimension Y: applies an optimisation / transformation step.

    The specific VISA opcode is configurable via node args so callers can
    switch between gradient descent (VSTEP_Y) and random mutation (VMUTATE_Y)
    without changing the operator class.
    """

    dimension = "Y"

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
        # Allow per-node opcode override without changing the operator class
        args = dict(args)
        opcode = args.pop("y_opcode", _DEFAULT_Y_OPCODE)

        return self._vm.run(
            VInstruction(opcode=opcode, args=args),
            warp_id=warp_id,
            node_id=node_id,
            dimension="Y",
            instance=instance,
            local=local,
            lane_gids=lane_gids,
            step=step,
        )
