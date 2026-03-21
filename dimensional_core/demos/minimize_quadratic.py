# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from core.graph import TaskGraph, Node


def build_cycle_graph(instance: dict) -> TaskGraph:
    """
    One iteration cycle:
    grad -> update -> score
    Uses instance['x'] as current value.
    """
    x = float(instance.get("x", 20.0))
    lr = float(instance.get("lr", 0.1))

    g = TaskGraph()
    g.add_node(Node(id="grad", op="compute_grad", params={"x": x}))
    g.add_node(Node(id="update", op="update_x", params={"lr": lr}))
    g.add_edge("grad", "update")
    g.add_node(Node(id="score", op="score_x", params={}))
    g.add_edge("update", "score")
    return g