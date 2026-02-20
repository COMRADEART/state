# dimensional_core/core/warp_factory.py
from __future__ import annotations

from .graph import Node


def build_warp_graph(warp_id: str, lane_gids: list[str], instance: dict, lanes: int = 4):
    """
    Builds a warp graph compatible with your current graph implementation.

    Your graph class is TaskGraph (not Graph) and it DOES NOT have .add().
    So we construct nodes, then call TaskGraph.add_node() / TaskGraph.add_edge()
    (these exist in your earlier versions), and return the TaskGraph.

    If your TaskGraph uses different method names, update ONLY the 3 methods:
      - _add_node(...)
      - _add_edge(...)
      - _set_entry(...)
    """

    # Import TaskGraph from your graph module (your class is named TaskGraph)
    from .graph import TaskGraph

    g = TaskGraph()

    # --------------------------
    # helpers (adapt layer)
    # --------------------------
    def _add_node(node: Node):
        # Most common names in your earlier code
        if hasattr(g, "add_node"):
            return g.add_node(node)
        if hasattr(g, "add"):
            return g.add(node)
        # fallback: direct dict insert
        if hasattr(g, "nodes") and isinstance(g.nodes, dict):
            g.nodes[node.id] = node
            return
        raise AttributeError("TaskGraph has no supported add method (add_node/add/nodes)")

    def _add_edge(a: str, b: str):
        # dependency edge: a -> b
        if hasattr(g, "add_edge"):
            return g.add_edge(a, b)
        if hasattr(g, "edge"):
            return g.edge(a, b)
        # fallback: if graph stores deps in node
        if hasattr(g, "nodes") and isinstance(g.nodes, dict) and b in g.nodes:
            n = g.nodes[b]
            deps = getattr(n, "deps", None)
            if isinstance(deps, list):
                deps.append(a)
                return
        raise AttributeError("TaskGraph has no supported edge method (add_edge/edge/deps)")

    def _set_entry(node_id: str):
        # optional: set entry/start node
        if hasattr(g, "entry"):
            g.entry = node_id
        elif hasattr(g, "start"):
            g.start = node_id
        # otherwise ignore

    # --------------------------
    # nodes
    # --------------------------
    # Vectors: we keep VISA ops, but your engine runs node.run() so Node must carry executable info.
    # Your existing Node.run() likely interprets node.op/params in your VM layer.

    init = Node(
        id=f"{warp_id}:init",
        op="VISA",
        params={
            "lane_gids": list(lane_gids),
            "lanes": lanes,
            "warp": warp_id,
            "stage": "init",
        },
    )

    step = Node(
        id=f"{warp_id}:step",
        op="VISA",
        params={
            "lane_gids": list(lane_gids),
            "lanes": lanes,
            "warp": warp_id,
            "stage": "step",
        },
    )

    score = Node(
        id=f"{warp_id}:score",
        op="VISA",
        params={
            "lane_gids": list(lane_gids),
            "lanes": lanes,
            "warp": warp_id,
            "stage": "score",
        },
    )

    # --------------------------
    # build graph
    # --------------------------
    _add_node(init)
    _add_node(step)
    _add_node(score)

    # init -> step -> score -> step (loop)
    _add_edge(init.id, step.id)
    _add_edge(step.id, score.id)
    _add_edge(score.id, step.id)

    _set_entry(init.id)

    return g
