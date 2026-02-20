from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class GraphZ:
    best_score: float = float("inf")
    last_score: float = float("inf")
    improved_steps: int = 0
    total_scored: int = 0
    age: int = 0  # increases each tick where graph is not chosen

    def update(self, score: float) -> Dict[str, Any]:
        self.total_scored += 1
        self.last_score = float(score)
        improved = False
        if score < self.best_score:
            self.best_score = float(score)
            self.improved_steps += 1
            improved = True
        return {
            "score": float(score),
            "best_score": float(self.best_score),
            "improved": improved,
            "improved_steps": self.improved_steps,
            "total_scored": self.total_scored,
        }
