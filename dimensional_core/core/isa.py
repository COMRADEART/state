from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Instr:
    op: str
    a: Optional[str] = None
    b: Optional[str] = None
    out: Optional[str] = None
    imm: Optional[float] = None


class MiniVM:
    def run(self, instr: Instr, instance: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
        op = instr.op.upper()

        def get(name: str) -> float:
            return float(local.get(name, 0.0))

        def set_(name: str, val: float) -> None:
            local[name] = float(val)

        if op == "LOAD_INSTANCE":
            val = float(instance.get(instr.a, 0.0))
            set_(instr.out, val)
            return {"loaded": {instr.out: val}}

        if op == "STORE_INSTANCE":
            val = get(instr.a)
            return {"instance_updates": {instr.out: val}}

        if op == "CONST":
            set_(instr.out, float(instr.imm))
            return {"const": {instr.out: float(instr.imm)}}

        if op == "ADD":
            set_(instr.out, get(instr.a) + get(instr.b))
            return {"reg": {instr.out: get(instr.out)}}

        if op == "SUB":
            set_(instr.out, get(instr.a) - get(instr.b))
            return {"reg": {instr.out: get(instr.out)}}

        if op == "MUL":
            set_(instr.out, get(instr.a) * get(instr.b))
            return {"reg": {instr.out: get(instr.out)}}

        if op == "GRAD_QUAD":
            x = get(instr.a)
            g = 2.0 * (x - 3.0)
            set_(instr.out, g)
            return {"grad": g}

        if op == "SCORE_QUAD":
            x = get(instr.a)
            s = (x - 3.0) * (x - 3.0)
            set_(instr.out, s)
            return {"score": s}

        raise RuntimeError(f"Unknown ISA op: {op}")
