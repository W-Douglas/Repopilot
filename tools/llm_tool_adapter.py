"""
LLM Tool Adapter - Bridges ToolManager with LLM function calling.

This module converts tools from ToolRegistry to LLM-compatible function schemas
and handles the actual tool execution when LLM requests a function call.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json

from .registry import ToolRegistry, ToolCategory
from .tool_manager import ToolManager
from .base import BaseTool
from .result import ToolResult


@dataclass
class FunctionCall:
    """Represents a function call from LLM."""
    name: str
    arguments: Dict[str, Any]  # Parsed JSON arguments


@dataclass
class FunctionSchema:
    """OpenAI-compatible function schema."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class LLMToolAdapter:
    """
    Adapter for LLM function calling.

    Converts tools to LLM-compatible schemas and executes tool calls.

    Usage:
        adapter = LLMToolAdapter(registry, tool_manager)
        schemas = adapter.get_function_schemas()

        # When LLM returns a function call:
        result = adapter.execute_function_call(
            name="read_file",
            arguments={"path": "main.py"}
        )
    """

    def __init__(self, registry: ToolRegistry, tool_manager: ToolManager):
        self.registry = registry
        self.tool_manager = tool_manager

    def get_function_schemas(
        self,
        categories: Optional[List[ToolCategory]] = None,
        enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get function schemas for LLM.

        Returns schemas in OpenAI function calling format.

        Args:
            categories: Filter by categories (None = all)
            enabled_only: Only include enabled tools

        Returns:
            List of function schemas
        """
        schemas = []
        tools = self.registry.list_tools(enabled_only=enabled_only)

        for tool_name in tools:
            tool = self.registry.get(tool_name)
            if tool is None:
                continue

            # Check category filter
            meta = self.registry.get_metadata(tool_name)
            if meta and categories:
                if meta.category not in categories:
                    continue

            # Convert to function schema
            schema = tool.get_schema()
            # Rename 'parameters' key to work with OpenAI format
            if 'parameters' in schema:
                schema['parameters'] = self._convert_to_openai_schema(
                    schema['parameters']
                )
            schemas.append(schema)

        return schemas

    def _convert_to_openai_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our parameter format to OpenAI JSON Schema."""
        properties = params.get('properties', {})
        required = params.get('required', [])

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def execute_function_call(
        self,
        function_call: FunctionCall,
        node_name: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a function call from LLM.

        Args:
            function_call: The function call to execute
            node_name: Which agent/node is making the call

        Returns:
            ToolResult from execution
        """
        return self.tool_manager.execute(
            tool_name=function_call.name,
            node_name=node_name,
            **function_call.arguments
        )

    def execute_raw(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        node_name: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a tool call with raw arguments.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments dict
            node_name: Agent node name

        Returns:
            ToolResult
        """
        return self.tool_manager.execute(
            tool_name=tool_name,
            node_name=node_name,
            **arguments
        )

    def parse_tool_calls(self, llm_response: str) -> List[FunctionCall]:
        """
        Parse function calls from LLM response text.

        This handles various LLM output formats for function calls.

        Args:
            llm_response: Raw LLM response string

        Returns:
            List of FunctionCall objects
        """
        calls = []

        # Try JSON format first
        try:
            # Look for JSON array in response
            if '```json' in llm_response:
                start = llm_response.find('```json') + 7
                end = llm_response.find('```', start)
                json_str = llm_response[start:end].strip()
            elif '"tool_calls"' in llm_response or '"function_call"' in llm_response:
                # Find JSON object
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
            else:
                json_str = None

            if json_str:
                data = json.loads(json_str)
                if isinstance(data, list):
                    for item in data:
                        calls.append(self._parse_single_call(item))
                elif isinstance(data, dict):
                    if 'tool_calls' in data:
                        for tc in data['tool_calls']:
                            calls.append(self._parse_single_call(tc))
                    elif 'function_call' in data:
                        calls.append(self._parse_single_call(data['function_call']))
        except (json.JSONDecodeError, KeyError):
            pass

        # Try markdown format: ```tool_name(...)```
        if not calls:
            import re
            pattern = r'```(\w+)\s*\((.*?)\)```'
            matches = re.findall(pattern, llm_response, re.DOTALL)
            for name, args_str in matches:
                try:
                    args = json.loads(f"{{{args_str}}}")
                    calls.append(FunctionCall(name=name, arguments=args))
                except json.JSONDecodeError:
                    # Try simple key=value parsing
                    kwargs = {}
                    for pair in args_str.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            kwargs[k.strip()] = v.strip().strip('"\'')
                    if kwargs:
                        calls.append(FunctionCall(name=name, arguments=kwargs))

        return calls

    def _parse_single_call(self, data: Dict[str, Any]) -> FunctionCall:
        """Parse a single function call from dict."""
        if 'name' in data:
            name = data['name']
            args = data.get('arguments', data.get('args', {}))
        elif 'function' in data:
            name = data['function'].get('name', data['function'].get('title', ''))
            args = data['function'].get('arguments', data['function'].get('parameters', {}))
        else:
            name = list(data.keys())[0] if data else ''
            args = data.get(name, {})

        # Ensure args is a dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        return FunctionCall(name=name, arguments=args)


class AgentToolSystem:
    """
    Complete tool system for LLM agents.

    Combines ToolRegistry, ToolManager, and LLMToolAdapter
    into a unified interface for agent-based tool calling.

    Usage:
        system = AgentToolSystem(root="/repo")

        # Get available tools for an agent
        schemas = system.get_tools_for_agent("PlannerAgent")

        # Execute a tool call
        result = system.call_tool(
            agent="PlannerAgent",
            tool="read_file",
            params={"path": "main.py"}
        )

        # Get execution trajectory
        trajectory = system.get_trajectory()
    """

    def __init__(self, root: str, execution_mode: str = "restricted"):
        from .tool_manager import ToolManager, ToolManagerConfig, ExecutionMode

        # Create registry
        self.registry = ToolRegistry.create_default_registry(root)

        # Create config
        mode = ExecutionMode[execution_mode.upper()]
        config = ToolManagerConfig(
            execution_mode=mode,
            max_calls_per_turn=10,
            max_total_calls=100,
            enable_trajectory=True,
        )

        # Create tool manager
        self.tool_manager = ToolManager(
            registry=self.registry,
            config=config,
            root=root
        )

        # Create adapter
        self.adapter = LLMToolAdapter(self.registry, self.tool_manager)

        # Agent-specific tool assignments
        self._agent_tools: Dict[str, List[str]] = {
            "PlannerAgent": ["read_file", "get_repo_tree", "search_code"],
            "CoderAgent": ["read_file", "write_file", "get_git_diff"],
            "ReviewAgent": ["read_file", "run_pytest", "get_git_diff"],
            "DebugAgent": ["read_file", "search_code", "grep", "run_pytest"],
        }

        # Register agent tools with tool manager
        for agent_name, tools in self._agent_tools.items():
            self.tool_manager.set_node_tools(agent_name, tools)

    def get_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available for a specific agent."""
        tool_names = self._agent_tools.get(agent_name, [])
        return [self.registry.get(name).get_schema()
                for name in tool_names if self.registry.get(name)]

    def set_agent_tools(self, agent_name: str, tools: List[str]):
        """Set which tools an agent can access."""
        self._agent_tools[agent_name] = tools
        self.tool_manager.set_node_tools(agent_name, tools)

    def call_tool(
        self,
        agent: str,
        tool: str,
        params: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool call on behalf of an agent."""
        return self.tool_manager.execute(tool, node_name=agent, **params)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get tool execution history."""
        return self.tool_manager.get_trajectory()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.tool_manager.get_stats()

    def reset(self):
        """Reset counters and trajectory."""
        self.tool_manager.reset_turn()
        self.tool_manager.clear_trajectory()


# Convenience function
def create_agent_tool_system(
    root: str = ".",
    execution_mode: str = "restricted"
) -> AgentToolSystem:
    """Create a new agent tool system."""
    return AgentToolSystem(root=root, execution_mode=execution_mode)