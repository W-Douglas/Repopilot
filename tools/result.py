"""
Unified Tool Result Format.

All tools return the same structured result, ensuring consistent interface
for downstream processing (Planner, Coder, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from enum import Enum
import time


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ToolResult:
    """
    Unified return format for all tools.

    Design principles:
    1. Always include tool name for traceability
    2. Success/failure status for easy branching
    3. Structured output for programmatic parsing
    4. Latency tracking for performance monitoring
    5. Error details for debugging

    Example:
        ToolResult(
            tool="read_file",
            success=True,
            input={"path": "src/main.py"},
            output="def main():\n    pass",
            latency=0.023
        )
    """

    tool: str                          # Tool identifier (e.g., "read_file")
    success: bool                      # Execution success flag
    input: Dict[str, Any]             # Tool input parameters
    output: Any                        # Tool output (type depends on tool)
    error: Optional[str] = None        # Error message if failed
    latency: float = 0.0               # Execution time in seconds
    status: ToolStatus = ToolStatus.SUCCESS

    # Metadata for downstream processing
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string status to enum if needed."""
        if isinstance(self.status, str):
            self.status = ToolStatus(self.status)

    def to_summary(self) -> str:
        """Generate human-readable summary for LLM context."""
        if self.success:
            preview = str(self.output)[:200]
            if len(str(self.output)) > 200:
                preview += "..."
            return f"[{self.tool}] Success ({self.latency:.3f}s): {preview}"
        else:
            return f"[{self.tool}] Failed: {self.error}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool": self.tool,
            "success": self.success,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "latency": round(self.latency, 3),
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        status = data.get("status", "success")
        if isinstance(status, str):
            try:
                status = ToolStatus(status)
            except ValueError:
                status = ToolStatus.FAILURE

        return cls(
            tool=data["tool"],
            success=data["success"],
            input=data.get("input", {}),
            output=data.get("output"),
            error=data.get("error"),
            latency=data.get("latency", 0.0),
            status=status,
            metadata=data.get("metadata", {}),
        )

    def is_failure(self) -> bool:
        return not self.success

    def is_success(self) -> bool:
        return self.success


def create_success_result(tool: str, input_params: Dict, output: Any,
                          latency: float = 0.0, metadata: Dict = None) -> ToolResult:
    """Factory function for successful results."""
    return ToolResult(
        tool=tool,
        success=True,
        input=input_params,
        output=output,
        latency=latency,
        status=ToolStatus.SUCCESS,
        metadata=metadata or {}
    )


def create_error_result(tool: str, input_params: Dict, error: str,
                        status: ToolStatus = ToolStatus.FAILURE,
                        latency: float = 0.0) -> ToolResult:
    """Factory function for error results."""
    return ToolResult(
        tool=tool,
        success=False,
        input=input_params,
        output=None,
        error=error,
        latency=latency,
        status=status,
    )


@dataclass
class ToolCall:
    """Represents a single tool invocation for trajectory recording."""
    tool: str
    input_params: Dict[str, Any]
    result: ToolResult
    node: str = ""  # Which LangGraph node called this tool
    step: int = 0   # Execution step number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "input_params": self.input_params,
            "success": self.result.success,
            "output_summary": str(self.result.output)[:500] if self.result.output else None,
            "error": self.result.error,
            "latency": self.result.latency,
            "node": self.node,
            "step": self.step,
        }