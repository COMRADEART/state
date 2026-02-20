from __future__ import annotations
from .graph import TaskGraph, Node
from .isa import Instr


def build_opt_graph(gid: str, instance: dict) -> TaskGraph:
    g = TaskGraph()

    x_key = f"x_{gid}"
    lr_key = f"lr_{gid}"

    instance.setdefault(x_key, 20.0)
    instance.setdefault(lr_key, 0.1)

    g.add_node(Node(
        id=f"{gid}:loadx",
        op="ISA",
        params={"instr": Instr(op="LOAD_INSTANCE", a=x_key, out="X").__dict__},
    ))

    g.add_node(Node(
        id=f"{gid}:loadlr",
        op="ISA",
        params={"instr": Instr(op="LOAD_INSTANCE", a=lr_key, out="LR").__dict__},
    ))

    g.add_node(Node(
        id=f"{gid}:grad",
        op="ISA",
        params={"instr": Instr(op="GRAD_QUAD", a="X", out="G").__dict__},
    ))
    g.add_edge(f"{gid}:loadx", f"{gid}:grad")

    g.add_node(Node(
        id=f"{gid}:mul",
        op="ISA",
        params={"instr": Instr(op="MUL", a="LR", b="G", out="TMP").__dict__},
    ))
    g.add_edge(f"{gid}:loadlr", f"{gid}:mul")
    g.add_edge(f"{gid}:grad", f"{gid}:mul")

    g.add_node(Node(
        id=f"{gid}:update",
        op="ISA",
        params={"instr": Instr(op="SUB", a="X", b="TMP", out="X2").__dict__},
    ))
    g.add_edge(f"{gid}:mul", f"{gid}:update")
    g.add_edge(f"{gid}:loadx", f"{gid}:update")

    g.add_node(Node(
        id=f"{gid}:store",
        op="ISA",
        params={"instr": Instr(op="STORE_INSTANCE", a="X2", out=x_key).__dict__},
    ))
    g.add_edge(f"{gid}:update", f"{gid}:store")

    g.add_node(Node(
        id=f"{gid}:score",
        op="ISA",
        params={"instr": Instr(op="SCORE_QUAD", a="X2", out="S").__dict__},
    ))
    g.add_edge(f"{gid}:store", f"{gid}:score")

    return g
