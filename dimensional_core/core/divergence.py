from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import random
import time


@dataclass
class DivergenceConfig:
    """
    Probability that any lane becomes "heavy" on a batch.
    heavy_extra_ms: how much extra latency heavy lanes add.
    """
    p_heavy_lane: float = 0.35
    heavy_extra_ms: int = 60


class DivergenceModel:
    def __init__(self, lanes: int, cfg: DivergenceConfig | None = None, seed: int = 123) -> None:
        self.lanes = int(lanes)
        self.cfg = cfg or DivergenceConfig()
        self.rng = random.Random(seed)

    def sample_lane_costs_ms(self) -> List[int]:
        """
        Returns per-lane additional latency in milliseconds (0 or heavy_extra_ms).
        """
        costs = []
        for _ in range(self.lanes):
            heavy = self.rng.random() < self.cfg.p_heavy_lane
            costs.append(self.cfg.heavy_extra_ms if heavy else 0)
        return costs

    @staticmethod
    def occupancy_from_costs(costs_ms: List[int]) -> Dict[str, Any]:
        """
        If a lane has extra cost, we treat it as active longer.
        For reporting we define:
          active_lanes = lanes that are not "fully idle" (all lanes count as active here)
          idle_lanes   = lanes with 0 extra cost (they finish early relative to heavy lanes)
        """
        lanes = len(costs_ms)
        idle = sum(1 for c in costs_ms if c == 0)
        heavy = lanes - idle
        occ = (heavy / lanes) if lanes else 0.0
        return {
            "lanes": lanes,
            "heavy_lanes": heavy,
            "idle_lanes": idle,
            "occupancy": occ,
        }

    @staticmethod
    def sleep_for_divergence(costs_ms: List[int]) -> None:
        """
        Sleep based on the slowest lane.
        """
        worst = max(costs_ms) if costs_ms else 0
        if worst > 0:
            time.sleep(worst / 1000.0)
