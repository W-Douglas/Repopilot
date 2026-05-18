"""
Tool Registry - manages available tools and their access control.

Design principles:
1. Central registry of all tools
2. Category-based organization
3. Tool enable/disable controls
4. LangGraph integration for controlled access
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseTool
from .result import ToolResult


class ToolCategory(Enum):
    """Tool categories for organization."""
    FILE = "file"           # File operations
    SEARCH = "search"       # Search operations
    GIT = "git"            # Git operations
    TEST = "test"          # Test execution
    TREE = "tree"          # Repository structure
    SYSTEM = "system"      # System operations


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    tool: BaseTool
    category: ToolCategory
    enabled: bool = True
    max_calls_per_turn: int = 5  # Limit calls per LangGraph turn
    timeout: float = 30.0        # Max execution time
    requires_context: List[str] = field(default_factory=list)  # Required state fields
    provides_context: List[str] = field(default_factory=list)  # State fields this tool populates


class ToolRegistry:
    """
    Central registry for all tools.

    Usage:
        registry = ToolRegistry()
        registry.register(ReadFileTool(root="/repo"))

        # Get tools for a specific node
        node_tools = registry.get_tools_for_node("BuildRepoMapNode")

        # Check if tool is available
        if registry.is_available("read_file"):
            result = registry.execute("read_file", path="main.py")
    """

    def __init__(self, root: Optional[str] = None):
        self.root = root
        self._tools: Dict[str, ToolMetadata] = {}
        self._category_map: Dict[ToolCategory, List[str]] = {}

    def register(self, tool: BaseTool, category: ToolCategory = ToolCategory.SYSTEM,
                 **metadata) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
            category: Tool category
            **metadata: Additional metadata (max_calls, timeout, etc.)
        """
        meta = ToolMetadata(
            tool=tool,
            category=category,
            **metadata
        )

        self._tools[tool.name] = meta

        # Update category map
        if category not in self._category_map:
            self._category_map[category] = []
        if tool.name not in self._category_map[category]:
            self._category_map[category].append(tool.name)

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            meta = self._tools.pop(tool_name)
            if meta.category in self._category_map:
                if tool_name in self._category_map[meta.category]:
                    self._category_map[meta.category].remove(tool_name)
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        meta = self._tools.get(tool_name)
        return meta.tool if meta else None

    def get_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get tool metadata."""
        return self._tools.get(tool_name)

    def list_tools(self, category: Optional[ToolCategory] = None,
                   enabled_only: bool = True) -> List[str]:
        """
        List available tools.

        Args:
            category: Filter by category
            enabled_only: Only return enabled tools

        Returns:
            List of tool names
        """
        if category:
            tool_names = self._category_map.get(category, [])
            if enabled_only:
                return [n for n in tool_names if self._tools[n].enabled]
            return tool_names

        if enabled_only:
            return [n for n, m in self._tools.items() if m.enabled]
        return list(self._tools.keys())

    def is_available(self, tool_name: str) -> bool:
        """Check if tool is available."""
        meta = self._tools.get(tool_name)
        return meta is not None and meta.enabled

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool parameters

        Returns:
            ToolResult from execution
        """
        meta = self._tools.get(tool_name)
        if not meta:
            from .result import create_error_result
            return create_error_result(
                tool=tool_name,
                input_params=kwargs,
                error=f"Tool not found: {tool_name}",
                status="failure"
            )

        if not meta.enabled:
            from .result import create_error_result
            return create_error_result(
                tool=tool_name,
                input_params=kwargs,
                error=f"Tool disabled: {tool_name}",
                status="permission_denied"
            )

        tool = meta.tool
        return tool.execute(**kwargs)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all available tools."""
        schemas = []
        for name, meta in self._tools.items():
            if meta.enabled:
                schemas.append(meta.tool.get_schema())
        return schemas

    def create_default_registry(root: str = ".") -> "ToolRegistry":
        """
        Create registry with default tools.

        This creates all tools defined in the design document:
        - read_file, write_file, search_code
        - get_repo_tree, run_pytest
        - get_git_diff, rollback_file
        """
        from .file_tools import ReadFileTool, WriteFileTool
        from .search_tools import SearchCodeTool
        from .tree_tools import GetRepoTreeTool
        from .test_tools import RunPytestTool
        from .git_tools import GetGitDiffTool, RollbackFileTool

        registry = ToolRegistry(root=root)

        # File tools
        registry.register(ReadFileTool(root=root), ToolCategory.FILE)
        registry.register(WriteFileTool(root=root), ToolCategory.FILE)

        # Search tools
        registry.register(SearchCodeTool(root=root), ToolCategory.SEARCH)

        # Tree tools
        registry.register(GetRepoTreeTool(root=root), ToolCategory.TREE)

        # Test tools
        registry.register(RunPytestTool(root=root), ToolCategory.TEST)

        # Git tools
        registry.register(GetGitDiffTool(root=root), ToolCategory.GIT)
        registry.register(RollbackFileTool(root=root), ToolCategory.GIT)

        return registry