# dimensional_core/core/visa/registry.py
"""
Self-registering VISA instruction system.

Usage
-----
    from dimensional_core.core.visa.registry import visa_instruction, VISAInstruction

    @visa_instruction("MY_OP")
    class MyOp(VISAInstruction):
        def execute(self, ctx: ExecutionContext) -> dict:
            return {"result": "ok"}

Instructions register themselves at import time — no manual dispatch table edits.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

# ── Global registry ────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, Type["VISAInstruction"]] = {}


def visa_instruction(opcode: str):
    """
    Class decorator that registers a VISAInstruction subclass.

    The opcode string uniquely identifies the instruction.  Registering the
    same opcode twice raises ``RuntimeError`` to prevent silent shadowing.
    """
    def decorator(cls: Type["VISAInstruction"]) -> Type["VISAInstruction"]:
        if opcode in _REGISTRY:
            raise RuntimeError(
                f"VISA opcode {opcode!r} is already registered by {_REGISTRY[opcode].__name__}. "
                f"Cannot re-register with {cls.__name__}."
            )
        cls.opcode = opcode
        _REGISTRY[opcode] = cls
        return cls

    return decorator


def get_instruction(opcode: str) -> Optional[Type["VISAInstruction"]]:
    """Return the instruction class for *opcode*, or ``None`` if not found."""
    return _REGISTRY.get(opcode)


def registered_opcodes() -> list[str]:
    """Return sorted list of all registered opcodes."""
    return sorted(_REGISTRY.keys())


def dispatch(opcode: str, ctx: "ExecutionContext") -> Dict[str, Any]:
    """
    Dispatch *opcode* via the registry.

    Raises ``RuntimeError`` if the opcode is not registered.
    """
    cls = _REGISTRY.get(opcode)
    if cls is None:
        raise RuntimeError(
            f"Unknown VISA opcode {opcode!r}. "
            f"Registered opcodes: {registered_opcodes()}"
        )
    return cls().execute(ctx)


# ── Execution context ──────────────────────────────────────────────────────────

class ExecutionContext:
    """
    Immutable execution context passed to every VISA instruction.

    Attributes
    ----------
    warp_id   : str   — owning warp (e.g. "W0")
    node_id   : str   — graph node being executed (e.g. "W0:Y")
    dimension : str   — "X", "Y", or "Z"
    instance  : dict  — shared instance state (mutable, global)
    local     : dict  — warp-local state (mutable, per-warp)
    lane_gids : list  — logical compute lane IDs
    args      : dict  — instruction arguments from the node params
    step      : int   — monotonic global step counter
    """

    __slots__ = (
        "warp_id", "node_id", "dimension",
        "instance", "local", "lane_gids",
        "args", "step",
    )

    def __init__(
        self,
        warp_id: str,
        node_id: str,
        dimension: str,
        instance: Dict[str, Any],
        local: Dict[str, Any],
        lane_gids: list,
        args: Dict[str, Any],
        step: int = 0,
    ) -> None:
        self.warp_id = warp_id
        self.node_id = node_id
        self.dimension = dimension
        self.instance = instance
        self.local = local
        self.lane_gids = lane_gids
        self.args = args
        self.step = step


# ── Base instruction class ─────────────────────────────────────────────────────

class VISAInstruction(ABC):
    """
    Abstract base for all VISA instructions.

    Subclass and decorate with ``@visa_instruction("OPCODE")`` to register.
    The ``execute`` method must be pure with respect to the *context* fields
    other than ``local`` and ``instance`` (which it may mutate in-place).
    """

    opcode: str = ""

    @abstractmethod
    def execute(self, ctx: ExecutionContext) -> Dict[str, Any]:
        """Execute and return a result dict."""
        ...
