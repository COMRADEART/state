# dimensional_core/core/visa
# Self-registering Vector Instruction Set Architecture (VISA)
from .registry import visa_instruction, dispatch, get_instruction, ExecutionContext, VISAInstruction
from .vm import VectorVM, VInstruction
import dimensional_core.core.visa.instructions  # noqa: F401 — trigger auto-registration

__all__ = [
    "visa_instruction",
    "dispatch",
    "get_instruction",
    "ExecutionContext",
    "VISAInstruction",
    "VectorVM",
    "VInstruction",
]
