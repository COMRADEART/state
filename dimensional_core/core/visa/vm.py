# dimensional_core/core/visa/vm.py
"""
VectorVM — thin dispatcher that runs a single VInstruction through the
self-registering VISA registry.

Keeps all instruction logic in visa/instructions.py; the VM is only
responsible for assembling the ExecutionContext and calling dispatch().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .registry import ExecutionContext, dispatch


@dataclass
class VInstruction:
    """
    A single VISA instruction ready to execute.

    Attributes
    ----------
    opcode : str  — registered opcode string (e.g. "VINIT_X")
    args   : dict — keyword arguments forwarded to the instruction
    """

    opcode: str
    args: Dict[str, Any] = field(default_factory=dict)


class VectorVM:
    """
    Executes VInstruction objects through the self-registering dispatch table.

    Thread-safe: stateless — all mutable state lives in ``local`` / ``instance``
    dicts owned by the caller.
    """

    def run(
        self,
        instr: VInstruction,
        warp_id: str,
        node_id: str,
        dimension: str,
        instance: Dict[str, Any],
        local: Dict[str, Any],
        lane_gids: List[str],
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Build an ExecutionContext and dispatch the instruction.

        Parameters
        ----------
        instr      : VISA instruction to run
        warp_id    : owning warp ID (e.g. "W0")
        node_id    : graph node ID (e.g. "W0:Y")
        dimension  : "X", "Y", or "Z"
        instance   : shared global instance state (mutable)
        local      : warp-local state dict (mutable)
        lane_gids  : logical lane identifiers
        step       : monotonic global step at dispatch time

        Returns
        -------
        Result dict from the instruction — always a ``dict``.
        """
        ctx = ExecutionContext(
            warp_id=warp_id,
            node_id=node_id,
            dimension=dimension,
            instance=instance,
            local=local,
            lane_gids=lane_gids,
            args=dict(instr.args),   # copy so instructions can pop safely
            step=step,
        )
        return dispatch(instr.opcode, ctx)
