# dimensional_core/core/dimensions/base.py
"""Abstract base for X / Y / Z dimension operators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DimensionOperator(ABC):
    """
    Base class for per-dimension execution operators.

    Each dimension (X, Y, Z) has exactly one operator that translates a
    *node execution request* into a VISA dispatch call.

    Concrete subclasses do NOT contain algorithm logic — that lives in
    visa/instructions.py.  This layer exists so the engine can dispatch by
    dimension string without any if/elif chains.
    """

    dimension: str = ""  # "X", "Y", or "Z"

    @abstractmethod
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
        """
        Run the dimension-specific operation.

        Must be thread-safe: called from worker threads with no shared
        mutable state other than *instance* and *local* (both owned by the
        caller's warp and protected by its own lock during result apply).

        Returns a result dict.
        """
        ...
