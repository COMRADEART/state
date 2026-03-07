from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

Axis = Literal["X", "Y", "Z"]


@dataclass(frozen=True)
class TransferEnvelope:
    run_id: str
    mp_id: str
    axis_from: Axis
    axis_to: Axis
    seq_end: int
    snapshot_ref: str
    state_hash: str
    op_trace: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mp_id": self.mp_id,
            "axis_from": self.axis_from,
            "axis_to": self.axis_to,
            "seq_end": self.seq_end,
            "snapshot_ref": self.snapshot_ref,
            "state_hash": self.state_hash,
            "op_trace": self.op_trace,
        }