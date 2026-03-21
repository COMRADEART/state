# dimensional_core/core/visa/instructions.py
"""
Built-in VISA instructions for the X / Y / Z execution dimensions.

All classes self-register at import time via @visa_instruction.
Import this module once (done by visa/__init__.py) to activate them.

Design:
  VINIT_X  — Dimension X: initialise per-lane state from instance/checkpoint
  VSTEP_Y  — Dimension Y: gradient-descent optimisation step
  VMUTATE_Y — Dimension Y variant: random perturbation + accept-if-better
  VSCORE_Z  — Dimension Z: verify correctness, determinism, convergence
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Dict, List

from .registry import ExecutionContext, VISAInstruction, visa_instruction


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stable_hash(obj: Any) -> str:
    """16-hex-char SHA-256 of a JSON-serialisable object."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _vec_local(local: Dict[str, Any]) -> Dict[str, Any]:
    local.setdefault("V", {})
    return local["V"]


def _scalar_local(local: Dict[str, Any]) -> Dict[str, Any]:
    local.setdefault("S", {})
    return local["S"]


# ── Dimension X — Initialisation ───────────────────────────────────────────────

@visa_instruction("VINIT_X")
class VInitX(VISAInstruction):
    """
    Dimension X initialisation.

    Loads per-lane ``x`` from the shared instance state, falling back to
    ``default_x`` (default 10.0).  Seeds the lane RNG deterministically from
    *step* and *warp_id* so replay is identical.
    """

    def execute(self, ctx: ExecutionContext) -> Dict[str, Any]:
        V = _vec_local(ctx.local)
        S = _scalar_local(ctx.local)

        default_x = float(ctx.args.get("default_x", 10.0))
        lane_gids = ctx.lane_gids or ["G0"]

        # Continuation mode: if a previous cycle left x_vec in local state,
        # resume from there instead of reinitialising from the instance.
        # This is what lets gradient descent converge across cycles.
        existing_x = V.get("x")
        if existing_x and len(existing_x) == len(lane_gids):
            x_vec = list(existing_x)
        else:
            # Fresh initialisation from instance checkpoint
            x_vec = []
            for gid in lane_gids:
                x = ctx.instance.get(f"x_{gid}", default_x)
                x_vec.append(float(x))
            if not x_vec:
                x_vec = [default_x]

        # Deterministic seed: reproducible across replay
        seed = int(S.get("seed") or (ctx.step * 7919 + abs(hash(ctx.warp_id))) % (2 ** 31))

        V["x"] = x_vec
        V["x_try"] = list(x_vec)
        V.setdefault("score", [float("inf")] * len(x_vec))

        S["seed"] = seed
        S.setdefault("minscore", float("inf"))
        S.setdefault("iterations", 0)

        return {
            "x_vec": x_vec,
            "lanes": len(x_vec),
            "seed": seed,
            "fresh_init": not bool(existing_x),
        }


# ── Dimension Y — Optimisation ─────────────────────────────────────────────────

@visa_instruction("VSTEP_Y")
class VStepY(VISAInstruction):
    """
    Dimension Y gradient-descent step.

    Objective: ``f(x) = (x - 3)^2``  →  gradient = ``2(x - 3)``
    Update:    ``x ← x - lr * grad``

    Converges toward x=3 (global minimum) from any starting point.
    """

    def execute(self, ctx: ExecutionContext) -> Dict[str, Any]:
        V = _vec_local(ctx.local)
        S = _scalar_local(ctx.local)

        lr = float(ctx.args.get("lr", 0.10))
        x_vec = list(V.get("x", [10.0]))

        new_x: List[float] = []
        scores: List[float] = []

        for x in x_vec:
            grad = 2.0 * (x - 3.0)
            x_new = x - lr * grad
            new_x.append(x_new)
            scores.append((x_new - 3.0) ** 2)

        V["x"] = new_x
        V["score"] = scores
        S["iterations"] = S.get("iterations", 0) + 1

        min_score = min(scores)
        prev_min = S.get("minscore", float("inf"))
        S["minscore"] = min(min_score, prev_min)

        return {
            "x_vec": new_x,
            "score_vec": scores,
            "min_score": min_score,
            "lr": lr,
            "iterations": S["iterations"],
        }


@visa_instruction("VMUTATE_Y")
class VMutateY(VISAInstruction):
    """
    Dimension Y mutation/exploration step.

    Perturbs each lane with Gaussian noise, accepts if the perturbed
    solution is better.  Seed evolves deterministically so replay works.
    """

    def execute(self, ctx: ExecutionContext) -> Dict[str, Any]:
        V = _vec_local(ctx.local)
        S = _scalar_local(ctx.local)

        noise = float(ctx.args.get("noise_scale", 0.20))
        seed = int(S.get("seed", ctx.step))
        rng = random.Random(seed)

        x_vec = list(V.get("x", [10.0]))
        x_try = [x + rng.gauss(0.0, noise) for x in x_vec]

        scores_orig = [(x - 3.0) ** 2 for x in x_vec]
        scores_try = [(x - 3.0) ** 2 for x in x_try]

        new_x: List[float] = []
        accepted = 0
        for i in range(len(x_vec)):
            if scores_try[i] < scores_orig[i]:
                new_x.append(x_try[i])
                accepted += 1
            else:
                new_x.append(x_vec[i])

        V["x"] = new_x
        V["x_try"] = x_try
        scores = [(x - 3.0) ** 2 for x in new_x]
        V["score"] = scores
        S["seed"] = seed + 1

        min_score = min(scores)
        S["minscore"] = min(min_score, S.get("minscore", float("inf")))

        return {
            "x_vec": new_x,
            "score_vec": scores,
            "min_score": min_score,
            "accepted": accepted,
            "total": len(x_vec),
        }


# ── Dimension Z — Verification ─────────────────────────────────────────────────

@visa_instruction("VSCORE_Z")
class VScoreZ(VISAInstruction):
    """
    Dimension Z verification operator.

    Checks:
    1. All scores are finite (no NaN / inf divergence).
    2. Best score is within the allowed divergence threshold (not worse than
       ``max_score_threshold``).
    3. Determinism: computes a SHA-256 checksum of (x_vec, scores, step) and
       optionally compares against an expected checksum stored by the engine.

    On failure: ``result["verified"] = False`` and ``result["rollback"] = True``
    so the engine knows to trigger a rollback.

    On success: returns ``score`` (the best lane score) so the scheduler can
    prioritise warps that are converging fastest.
    """

    def execute(self, ctx: ExecutionContext) -> Dict[str, Any]:
        V = _vec_local(ctx.local)

        x_vec: List[float] = V.get("x", [])
        scores: List[float] = V.get("score", [])

        # ── Guard: no scores at all ────────────────────────────────────────────
        if not scores:
            return {
                "verified": False,
                "rollback": True,
                "reason": "no_scores",
            }

        min_score = min(scores)

        # ── Check 1: finiteness ────────────────────────────────────────────────
        if not math.isfinite(min_score) or any(not math.isfinite(s) for s in scores):
            return {
                "verified": False,
                "rollback": True,
                "reason": "non_finite_score",
                "min_score": min_score,
            }

        # ── Check 2: divergence threshold ──────────────────────────────────────
        max_threshold = float(ctx.args.get("max_score_threshold", 1e6))
        if min_score > max_threshold:
            return {
                "verified": False,
                "rollback": True,
                "reason": "score_diverged",
                "min_score": min_score,
                "threshold": max_threshold,
            }

        # ── Check 3: determinism checksum ──────────────────────────────────────
        checksum = _stable_hash({
            "warp": ctx.warp_id,
            "step": ctx.step,
            "x_vec": [round(x, 10) for x in x_vec],
            "scores": [round(s, 10) for s in scores],
        })

        expected = ctx.args.get("expected_checksum")
        if expected and expected != checksum:
            return {
                "verified": False,
                "rollback": True,
                "reason": "checksum_mismatch",
                "checksum": checksum,
                "expected": expected,
                "min_score": min_score,
            }

        return {
            "verified": True,
            "rollback": False,
            "min_score": min_score,
            "score": min_score,       # scheduler key
            "checksum": checksum,
            "x_vec": x_vec,
        }
