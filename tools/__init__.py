"""
Tool System Module - Agent-Computer Interface for RepoPilot.

This module provides a controlled interface for LLM to interact with the codebase.

Core Components:
- ToolRegistry: Central registry of all tools
- ToolManager: Orchestrates tool execution with access control
- AgentToolSystem: Unified interface for LLM agents
- LLMToolAdapter: Converts tools to LLM function schemas
- LLMAgentNode: LangGraph nodes with LLM tool calling

Usage:
    # Create tool system for an agent
    system = create_agent_tool_system("/repo")

    # Get tools for agent
    schemas = system.get_tools_for_agent("PlannerAgent")

    # Execute tool
    result = system.call_tool(
        agent="PlannerAgent",
        tool="read_file",
        params={"path": "main.py"}
    )

    # In LangGraph workflow
    from tools.llm_agent_node import create_planner_agent
    workflow.add_node("planner", create_planner_agent().execute)
"""

from .result import ToolResult, ToolStatus, create_success_result, create_error_result
from .base import BaseTool, ToolParameter, ReadOnlyTool, WriteTool
from .registry import ToolRegistry, ToolCategory, ToolMetadata
from .tool_manager import (
    ToolManager,
    ToolManagerConfig,
    ExecutionMode,
    create_tool_manager_for_workflow,
    create_tool_calling_node,
    create_tool_node,
)
from .llm_tool_adapter import (
    LLMToolAdapter,
    FunctionCall,
    FunctionSchema,
    AgentToolSystem,
    create_agent_tool_system,
)
from .llm_agent_node import (
    AgentConfig,
    LLMAgentNode,
    create_planner_agent,
    create_coder_agent,
    create_reviewer_agent,
)

__all__ = [
    # Result types
    "ToolResult",
    "ToolStatus",
    "create_success_result",
    "create_error_result",
    # Base
    "BaseTool",
    "ToolParameter",
    "ReadOnlyTool",
    "WriteTool",
    # Registry
    "ToolRegistry",
    "ToolCategory",
    "ToolMetadata",
    # Manager
    "ToolManager",
    "ToolManagerConfig",
    "ExecutionMode",
    "create_tool_manager_for_workflow",
    "create_tool_calling_node",
    "create_tool_node",
    # Adapter
    "LLMToolAdapter",
    "FunctionCall",
    "FunctionSchema",
    "AgentToolSystem",
    "create_agent_tool_system",
    # Agent Node
    "AgentConfig",
    "LLMAgentNode",
    "create_planner_agent",
    "create_coder_agent",
    "create_reviewer_agent",
]