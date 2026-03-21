# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from typing import Dict, Any


def f(x: float) -> float:
    return (x - 3.0) ** 2


def grad_f(x: float) -> float:
    return 2.0 * (x - 3.0)


# Op signature: op(params, instance, local) -> dict result
def ops() -> Dict[str, Any]:
    def compute_grad(params: Dict[str, Any], instance: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
        x = float(params["x"])
        g = grad_f(x)
        local["x"] = x
        local["g"] = g
        return {"grad": g, "x": x}

    def update_x(params: Dict[str, Any], instance: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
        lr = float(params["lr"])
        x_key = params.get("x_key", "x")

        x = float(local.get("x", instance.get(x_key, 20.0)))
        g = float(local.get("g", 0.0))

        x2 = x - lr * g
        local["x2"] = x2

        return {"instance_updates": {x_key: x2}, "x_new": x2}

    def score(params: Dict[str, Any], instance: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
        x_key = params.get("x_key", "x")
        x = float(instance.get(x_key, local.get("x2", local.get("x", 0.0))))
        s = f(x)
        return {"score": s, "params": {"x": x}}

    return {
        "compute_grad": compute_grad,
        "update_x": update_x,
        "score": score,
    }