from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OpResult:
    state: Dict[str, Any]
    meta: Dict[str, Any]


class OpInitX:
    def apply(self, state: Dict[str, Any]) -> OpResult:
        s = dict(state)
        s["stage"] = "X_INIT"
        s.setdefault("metrics", {})
        s["metrics"]["init_done"] = True
        return OpResult(state=s, meta={"op": "op_init_x"})


class OpRefineY:
    """
    A refine op with a 'mode' to demonstrate branching and safe-mode reruns.
    mode:
    - "FAST": bigger step (can overshoot)
    - "SAFE": smaller step (more conservative)
    """
    def __init__(self, mode: str = "FAST") -> None:
        self.mode = mode

    def apply(self, state: Dict[str, Any]) -> OpResult:
        s = dict(state)
        s["stage"] = "Y_REFINE"
        s.setdefault("metrics", {})
        old_loss = float(s["metrics"].get("loss", 10.0))

        if self.mode == "FAST":
            new_loss = max(old_loss * 0.80, 0.0)
        else:
            new_loss = max(old_loss * 0.90, 0.0)

        s["metrics"]["loss"] = new_loss
        s["metrics"]["mode"] = self.mode

        return OpResult(
            state=s,
            meta={"op": "op_refine_y", "mode": self.mode, "prev_loss": old_loss, "loss": new_loss},
        )