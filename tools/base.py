"""
Base Tool class - foundation for all tools.

Design principles:
1. All tools inherit from BaseTool
2. Strict input validation via type hints
3. Unified execution interface
4. Built-in error handling and logging
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type
from dataclasses import dataclass
import time
import inspect

from .result import ToolResult, ToolStatus, create_success_result, create_error_result


@dataclass
class ToolParameter:
    """Defines a tool parameter with validation rules."""
    name: str
    type: Type
    required: bool = True
    description: str = ""
    default: Any = None
    validator: Optional[callable] = None


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools should:
    1. Define parameters via `get_parameters()`
    2. Implement `execute()` method
    3. Handle errors gracefully
    4. Return ToolResult objects

    Example:
        class ReadFileTool(BaseTool):
            name = "read_file"
            description = "Read contents of a file"

            @staticmethod
            def get_parameters() -> List[ToolParameter]:
                return [
                    ToolParameter("path", str, required=True, description="File path"),
                    ToolParameter("lines", int, required=False, default=100,
                                  description="Max lines to read")
                ]

            def execute(self, path: str, lines: int = 100) -> ToolResult:
                ...
    """

    name: str = ""
    description: str = ""
    category: str = "general"  # file, search, git, test, system

    # Tool availability flags
    enabled: bool = True
    requires_confirmation: bool = False  # For destructive operations

    def __init__(self, root: Optional[str] = None):
        """
        Initialize tool with repository root.

        Args:
            root: Root directory of the repository
        """
        self.root = root

    @staticmethod
    @abstractmethod
    def get_parameters() -> List[ToolParameter]:
        """
        Define tool parameters.

        Returns:
            List of ToolParameter definitions
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate input parameters.

        Returns:
            (is_valid, error_message)
        """
        params = self.get_parameters()
        for param in params:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

            if param.name in kwargs and param.validator:
                value = kwargs[param.name]
                try:
                    param.validator(value)
                except Exception as e:
                    return False, f"Invalid value for {param.name}: {e}"

        return True, None

    def _run_with_timing(self, func: callable, *args, **kwargs) -> tuple[Any, float]:
        """Execute function and measure time."""
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return result, elapsed
        except Exception as e:
            elapsed = time.time() - start
            raise e

    def _create_result(self, success: bool, output: Any = None,
                       error: str = None, latency: float = 0.0,
                       status: ToolStatus = ToolStatus.SUCCESS) -> ToolResult:
        """Helper to create ToolResult."""
        return ToolResult(
            tool=self.name,
            success=success,
            input=kwargs_to_dict(self.get_parameters(), **({})),
            output=output,
            error=error,
            latency=latency,
            status=status,
        )

    def get_schema(self) -> Dict[str, Any]:
        """
        Generate JSON schema for LLM tool calling.

        This enables:
        1. LLM to know available tools
        2. Input format validation
        3. Structured tool descriptions
        """
        params = self.get_parameters()
        properties = {}
        required = []

        for param in params:
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            properties[param.name] = {
                "type": type_map.get(param.type, "string"),
                "description": param.description,
            }

            if param.default is not None:
                properties[param.name]["default"] = param.default

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


def kwargs_to_dict(params: List[ToolParameter], **kwargs) -> Dict[str, Any]:
    """Extract relevant kwargs to dict based on parameter definitions."""
    result = {}
    for param in params:
        if param.name in kwargs:
            result[param.name] = kwargs[param.name]
    return result


class ReadOnlyTool(BaseTool):
    """Base class for tools that only read, no modifications."""
    requires_confirmation = False

    def can_execute(self) -> bool:
        """Check if tool has required permissions."""
        return self.root is not None


class WriteTool(BaseTool):
    """Base class for tools that modify files."""
    requires_confirmation = True  # Modifications need confirmation

    def can_execute(self) -> bool:
        """Check if tool has write permissions."""
        return self.root is not None