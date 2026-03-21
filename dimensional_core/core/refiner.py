# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ZState:
    step: int = 0
    best_score: float = float("inf")
    best_params: Optional[Dict[str, Any]] = None


class Refiner:
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon

    def update(self, z: ZState, score: float, params: Dict[str, Any]) -> Dict[str, Any]:
        z.step += 1
        improved = (score + self.epsilon) < z.best_score
        if improved:
            z.best_score = score
            z.best_params = dict(params)
        return {
            "z_step": z.step,
            "score": score,
            "best_score": z.best_score,
            "improved": improved,
            "best_params": z.best_params,
        }