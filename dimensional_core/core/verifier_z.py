# EXPERIMENTAL — not used by the production engine
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Status = Literal["PASS", "FAIL_RETRYABLE", "FAIL_FATAL"]


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: Literal["LOW", "MED", "HIGH"]


@dataclass(frozen=True)
class VerifyReport:
    status: Status
    violations: List[Violation]
    rewind_to_mp_id: Optional[str] = None
    suggested_mode: Optional[str] = None


class VerifyRefinedZ:
    """
    Default Z verifier.
    - Validates schema
    - Validates numeric sanity
    - Validates monotonic improvement (if metrics exist)
    """

    def verify(self, state: Dict[str, Any], ctx: Dict[str, Any]) -> VerifyReport:
        violations: List[Violation] = []

        # A) schema checks
        for k in ("job_id", "stage", "metrics"):
            if k not in state:
                violations.append(Violation("SCHEMA_MISSING", f"Missing key: {k}", "HIGH"))

        # B) sanity checks
        metrics = state.get("metrics", {})
        if isinstance(metrics, dict):
            loss = metrics.get("loss")
            if loss is not None:
                try:
                    lf = float(loss)
                    if lf != lf:  # NaN
                        violations.append(Violation("NAN_LOSS", "loss is NaN", "HIGH"))
                    if lf < 0:
                        violations.append(Violation("NEG_LOSS", "loss < 0 unexpected", "MED"))
                except Exception:
                    violations.append(Violation("LOSS_TYPE", "loss is not numeric", "HIGH"))

        # C) monotonic improvement check (optional)
        prev_loss = ctx.get("prev_loss")
        if prev_loss is not None and isinstance(metrics, dict) and metrics.get("loss") is not None:
            try:
                if float(metrics["loss"]) > float(prev_loss) * 1.05:
                    violations.append(Violation("REGRESSION", "loss regressed >5%", "MED"))
            except Exception:
                pass

        if any(v.severity == "HIGH" for v in violations):
            return VerifyReport(
                status="FAIL_RETRYABLE",
                violations=violations,
                rewind_to_mp_id=ctx.get("rewind_default_mp_id"),
                suggested_mode="Y_SAFE",
            )

        if violations:
            # soft violations => still fail retryable by default (you can flip this)
            return VerifyReport(
                status="FAIL_RETRYABLE",
                violations=violations,
                rewind_to_mp_id=ctx.get("rewind_default_mp_id"),
                suggested_mode="Y_SAFE",
            )

        return VerifyReport(status="PASS", violations=[])