# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from typing import List


def make_warps(graph_ids: List[str], lanes: int = 4) -> List[List[str]]:
    ids = list(graph_ids)
    warps = []
    for i in range(0, len(ids), lanes):
        chunk = ids[i:i+lanes]
        if len(chunk) == lanes:
            warps.append(chunk)
    return warps