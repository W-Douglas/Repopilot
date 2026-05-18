"""
Tool Manager - orchestrates tool execution with LangGraph integration.

Design for controlled tool access:
1. Tools are NOT exposed directly to LLM
2. LangGraph nodes decide which tools to call
3. Tool results are stored in state for downstream nodes
4. Trajectory is recorded for debugging/evaluation

LangGraph Integration Pattern:
```python
# State contains tool_results
state = {
    "tool_results": [],  # List of ToolCall records
    "current_tool": None,
    ...
}

# Node calls tools via manager
def some_node(state):
    result = tool_manager.execute("read_file", path="main.py")
    state["tool_results"].append(result)
    return {"file_content": result.output}
```

Controlled Access:
- Only registered tools can be executed
- Tools have rate limits (max_calls_per_turn)
- Tools are associated with specific nodes
- Tool execution is logged in trajectory
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .registry import ToolRegistry, ToolCategory
from .result import ToolResult, ToolCall, ToolStatus


class ExecutionMode(Enum):
    """How tools can be executed."""
    DIRECT = "direct"           # Node calls tool directly
    RESTRICTED = "restricted"    # Only allowed tools per node
    REVIEW = "review"           # Require confirmation for writes


@dataclass
class ToolManagerConfig:
    """Configuration for tool manager."""
    execution_mode: ExecutionMode = ExecutionMode.RESTRICTED
    max_calls_per_turn: int = 10
    max_total_calls: int = 100
    enable_trajectory: bool = True
    timeout_default: float = 30.0


class ToolManager:
    """
    Manages tool execution with access control.

    Key features:
    1. Centralized tool execution
    2. Rate limiting per turn
    3. Trajectory recording
    4. Node-based access control

    Usage with LangGraph:
    ```python
    manager = ToolManager(registry=registry)

    # In a node:
    result = manager.execute(
        tool_name="read_file",
        node_name="PlanFixNode",  # Track which node called
        **params
    )
    ```
    """

    def __init__(self, registry: Optional[ToolRegistry] = None,
                 config: Optional[ToolManagerConfig] = None,
                 root: str = "."):
        self.registry = registry or ToolRegistry.create_default_registry(root)
        self.config = config or ToolManagerConfig()
        self.root = root

        # Execution tracking
        self._calls_this_turn = 0
        self._total_calls = 0
        self._trajectory: List[ToolCall] = []
        self._node_tools: Dict[str, List[str]] = {}

        # Initialize node-tool associations
        self._init_node_tools()

    def _init_node_tools(self):
        """Define which tools each node can access."""
        # Design from LangGraph workflow:
        # - BuildRepoMapNode: get_repo_tree, read_file
        # - RetrieveContextNode: read_file, search_code, get_repo_tree
        # - PlanFixNode: read_file (read only)
        # - EditCodeNode: read_file, write_file, get_git_diff
        # - RunTestNode: run_pytest, get_git_diff
        # - DebugFailureNode: read_file, search_code, get_git_diff

        self._node_tools = {
            "AnalyzeIssueNode": ["read_file", "get_repo_tree"],
            "BuildRepoMapNode": ["get_repo_tree", "read_file", "search_code"],
            "RetrieveContextNode": ["read_file", "search_code", "grep", "get_repo_tree"],
            "PlanFixNode": ["read_file"],
            "EditCodeNode": ["read_file", "write_file", "get_git_diff"],
            "RunTestNode": ["run_pytest", "get_git_diff"],
            "DebugFailureNode": ["read_file", "search_code", "grep", "get_git_status"],
            "GenerateReportNode": [],  # No tools needed
        }

    def set_node_tools(self, node_name: str, allowed_tools: List[str]):
        """Set which tools a specific node can access."""
        self._node_tools[node_name] = allowed_tools

    def get_available_tools(self, node_name: Optional[str] = None) -> List[str]:
        """
        Get tools available for execution.

        Args:
            node_name: If provided, return only tools for that node

        Returns:
            List of available tool names
        """
        if node_name and node_name in self._node_tools:
            return self._node_tools[node_name]

        return self.registry.list_tools(enabled_only=True)

    def execute(self, tool_name: str, node_name: Optional[str] = None,
                **kwargs) -> ToolResult:
        """
        Execute a tool with access control.

        Args:
            tool_name: Name of the tool to execute
            node_name: Which node is requesting the tool (for access control)
            **kwargs: Tool parameters

        Returns:
            ToolResult from execution
        """
        # Check rate limits
        if self._calls_this_turn >= self.config.max_calls_per_turn:
            from .result import create_error_result
            return create_error_result(
                tool=tool_name,
                input_params=kwargs,
                error=f"Rate limit exceeded: {self.config.max_calls_per_turn} calls per turn",
                status=ToolStatus.FAILURE
            )

        if self._total_calls >= self.config.max_total_calls:
            from .result import create_error_result
            return create_error_result(
                tool=tool_name,
                input_params=kwargs,
                error=f"Total call limit exceeded: {self.config.max_total_calls}",
                status=ToolStatus.FAILURE
            )

        # Check node access control
        if self.config.execution_mode == ExecutionMode.RESTRICTED:
            if node_name and node_name in self._node_tools:
                allowed = self._node_tools[node_name]
                if allowed and tool_name not in allowed:
                    from .result import create_error_result
                    return create_error_result(
                        tool=tool_name,
                        input_params=kwargs,
                        error=f"Tool '{tool_name}' not allowed for node '{node_name}'",
                        status=ToolStatus.PERMISSION_DENIED
                    )

        # Execute tool
        result = self.registry.execute(tool_name, **kwargs)

        # Update counters
        self._calls_this_turn += 1
        self._total_calls += 1

        # Record trajectory
        if self.config.enable_trajectory:
            call = ToolCall(
                tool=tool_name,
                input_params=kwargs,
                result=result,
                node=node_name or "",
                step=self._total_calls
            )
            self._trajectory.append(call)

        return result

    def reset_turn(self):
        """Reset per-turn call counter (call at start of each turn)."""
        self._calls_this_turn = 0

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get recorded tool execution trajectory."""
        return [call.to_dict() for call in self._trajectory]

    def clear_trajectory(self):
        """Clear recorded trajectory."""
        self._trajectory = []

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_calls = len(self._trajectory)
        successful = sum(1 for c in self._trajectory if c.result.success)
        failed = total_calls - successful

        tool_counts = {}
        for call in self._trajectory:
            tool_counts[call.tool] = tool_counts.get(call.tool, 0) + 1

        return {
            "total_calls": total_calls,
            "successful_calls": successful,
            "failed_calls": failed,
            "tool_usage": tool_counts,
            "current_turn_calls": self._calls_this_turn,
        }


def create_tool_manager_for_workflow(root: str = ".",
                                      execution_mode: ExecutionMode = ExecutionMode.RESTRICTED
                                      ) -> ToolManager:
    """
    Factory function to create tool manager with default setup.

    This creates a tool manager configured for the RepoPilot workflow:
    - Restricted execution mode (tools limited per node)
    - Trajectory recording enabled
    - Standard rate limits
    """
    from .registry import ToolRegistry

    config = ToolManagerConfig(
        execution_mode=execution_mode,
        max_calls_per_turn=10,
        max_total_calls=100,
        enable_trajectory=True,
    )

    registry = ToolRegistry.create_default_registry(root)

    return ToolManager(registry=registry, config=config, root=root)


# LangGraph Helper Functions

def create_tool_calling_node(tool_manager: ToolManager, node_name: str):
    """
    Create a helper function for calling tools in a node.

    Usage in LangGraph:
    ```python
    def PlanFixNode(state):
        call_tool = create_tool_calling_node(manager, "PlanFixNode")

        # Call read_file
        result = call_tool("read_file", path="calculator.py")
        state["file_content"] = result.output

        # Call search_code
        result = call_tool("search_code", pattern="def divide")
        state["divide_usage"] = result.output
    ```
    """
    def call_tool(tool_name: str, **kwargs) -> ToolResult:
        return tool_manager.execute(tool_name, node_name=node_name, **kwargs)

    return call_tool


def create_tool_node(tool_name: str, node_name: str):
    """
    Create a simple node that calls a single tool.

    This creates a reusable node for common operations.

    Usage:
    ```python
    read_file_node = create_tool_node("read_file", "RetrieveContextNode")

    # In workflow:
    # read_file_node will call read_file with params from state
    ```
    """
    def node_fn(state: dict) -> dict:
        manager = state.get("_tool_manager")
        if not manager:
            raise ValueError("Tool manager not in state")

        # Extract params for this tool from state
        params = state.get(f"_{tool_name}_params", {})

        result = manager.execute(tool_name, node_name=node_name, **params)

        # Store result in state
        state[f"_{tool_name}_result"] = result

        return state

    return node_fn