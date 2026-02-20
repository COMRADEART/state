# dimensional_core/core/isa_vector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import time

@dataclass
class VInstr:
    op: str
    args: Dict[str, Any]

class L2Cache:
    """Tiny shared cache simulation: line_ids set."""
    def __init__(self):
        self.lines = set()

    def access_lines(self, line_ids: List[int]) -> Dict[str, int]:
        hits = 0
        misses = 0
        for lid in line_ids:
            if lid in self.lines:
                hits += 1
            else:
                misses += 1
                self.lines.add(lid)
        return {"l2_hits": hits, "l2_misses": misses}

class VectorVM:
    def __init__(self):
        pass

    def _V(self, local: Dict[str, Any]) -> Dict[str, Any]:
        local.setdefault("V", {})
        return local["V"]

    def _S(self, local: Dict[str, Any]) -> Dict[str, Any]:
        local.setdefault("S", {})
        return local["S"]

    def _MEM(self, local: Dict[str, Any]) -> Dict[str, Any]:
        local.setdefault("MEM", {})
        return local["MEM"]

    def run(
        self,
        instr: VInstr,
        instance: Dict[str, Any],
        local: Dict[str, Any],
        lane_gids: List[str],
        shared_l2: Optional[L2Cache] = None,
    ) -> Dict[str, Any]:
        op = instr.op
        a = instr.args
        V = self._V(local)
        S = self._S(local)
        MEM = self._MEM(local)

        # ---- C18 ops ----

        if op == "VINIT":
            # Create x vector from instance per lane, default 6.0
            x = []
            for gid in lane_gids:
                x.append(float(instance.get(f"x_{gid}", 6.0)))
                instance.setdefault(f"lr_{gid}", 0.1)
            V["x"] = x
            V["x_try"] = list(x)
            V["score"] = [999999.0] * len(x)
            V["score_try"] = [999999.0] * len(x)
            S.setdefault("minscore", 999999.0)
            S.setdefault("seed", 0.0)
            return {"result": "OK"}

        if op == "VMEM_TOUCH":
            # simulate memory line access
            lines = int(a.get("lines", 4))
            MEM.setdefault("line_ids", [])
            # pick stable pseudo-random lines
            r = random.Random(int(S.get("seed", 0)) + int(time.time() * 10))
            line_ids = [r.randint(0, 63) for _ in range(lines)]
            MEM["line_ids"] = line_ids

            hits = 0
            misses = lines
            if shared_l2 is not None:
                stat = shared_l2.access_lines(line_ids)
                return {
                    "mem": {"hits": hits, "misses": misses, "bytes": lines * 128, "lines": lines},
                    "l2": {"l2_hits": stat["l2_hits"], "l2_misses": stat["l2_misses"], "l2_bytes": stat["l2_misses"] * 128, "l2_lines": lines},
                }
            return {"mem": {"hits": hits, "misses": misses, "bytes": lines * 128, "lines": lines}}

        if op == "VSTEP":
            # one gradient-ish step: x = x - lr * grad
            # demo objective: f(x) = (x-3)^2  => grad = 2(x-3)
            lr = float(a.get("lr", 0.1))
            x = V.get("x", [])
            g = []
            for i, xi in enumerate(x):
                grad = 2.0 * (xi - 3.0)
                g.append(grad)
                x[i] = xi - lr * grad
            V["g"] = g
            V["x"] = x
            return {"result": "OK"}

        if op == "VSCORE":
            # score per lane: (x-3)^2
            x = V.get("x", [])
            score = [(xi - 3.0) ** 2 for xi in x]
            V["score"] = score
            return {"score_vec": score, "min_score": float(min(score))}

        # ---- NEW in C18: candidate refine ----
        if op == "VMUTATE_X":
            # x_try = x + noise, keep in bounds
            sigma = float(a.get("sigma", 0.35))
            lo = float(a.get("lo", -50.0))
            hi = float(a.get("hi", 50.0))
            seed = int(a.get("seed", 0))
            # evolve seed slowly
            S["seed"] = float(S.get("seed", 0.0)) + 1.0
            r = random.Random(seed + int(S["seed"]))

            x = V.get("x", [])
            x_try = []
            for xi in x:
                xi2 = xi + r.uniform(-sigma, sigma)
                if xi2 < lo: xi2 = lo
                if xi2 > hi: xi2 = hi
                x_try.append(xi2)
            V["x_try"] = x_try
            return {"result": "OK", "seed": int(S["seed"])}

        if op == "VSCORE_TRY":
            x_try = V.get("x_try", [])
            score_try = [(xi - 3.0) ** 2 for xi in x_try]
            V["score_try"] = score_try
            return {"score_try": score_try, "min_try": float(min(score_try))}

        if op == "VACCEPT_IF_BETTER":
            min_cur = float(min(V.get("score", [999999.0])))
            min_try = float(min(V.get("score_try", [999999.0])))

            improved = min_try < min_cur
            if improved:
                V["x"] = list(V.get("x_try", V.get("x", [])))
                V["score"] = list(V.get("score_try", V.get("score", [])))
                S["minscore"] = min_try

            return {"improved": improved, "min_cur": min_cur, "min_try": min_try, "best": float(S.get("minscore", min_cur))}

        raise RuntimeError(f"Unknown vector ISA op: {op}")
# --- backwards-compatible names for existing engine imports ---
LANES_DEFAULT = 4
VecVM = VectorVM
