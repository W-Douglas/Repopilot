"""
Repository tree tools: get_repo_tree.
"""

import time
from pathlib import Path
from typing import Optional, List

from .base import BaseTool, ReadOnlyTool, ToolParameter
from .result import ToolResult, create_success_result, create_error_result


class GetRepoTreeTool(ReadOnlyTool):
    """
    Get directory tree structure of the repository.

    Designed for:
    - LLM to understand project layout
    - Finding file locations
    - Understanding module organization

    Parameters:
        max_depth: Maximum tree depth (default: 3)
        include_files: Include files in tree
        filter_pattern: Optional glob pattern to filter
    """

    name = "get_repo_tree"
    description = "Get directory tree structure of the repository. Use to understand project layout."
    category = "tree"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("max_depth", int, required=False, default=3,
                          description="Maximum tree depth"),
            ToolParameter("include_files", bool, required=False, default=True,
                          description="Include files in tree"),
            ToolParameter("filter", str, required=False, default=None,
                          description="Only show files matching pattern"),
        ]

    def execute(self, max_depth: int = 3,
                include_files: bool = True,
                filter: Optional[str] = None) -> ToolResult:
        start = time.time()

        try:
            root = Path(self.root) if self.root else Path(".")

            if not root.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"max_depth": max_depth},
                    error=f"Path does not exist: {root}",
                    latency=time.time() - start
                )

            tree_lines = []
            self._build_tree(root, root, 0, max_depth, include_files, filter, tree_lines)

            tree_output = "\n".join(tree_lines)

            return create_success_result(
                tool=self.name,
                input_params={"max_depth": max_depth, "include_files": include_files},
                output=tree_output,
                latency=time.time() - start,
                metadata={
                    "depth": max_depth,
                    "root": str(root),
                    "lines": len(tree_lines),
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"max_depth": max_depth},
                error=str(e),
                latency=time.time() - start
            )

    def _build_tree(self, base_path: Path, current_path: Path,
                    depth: int, max_depth: int,
                    include_files: bool, filter_pattern: Optional[str],
                    lines: list):
        """Recursively build tree representation."""

        if depth > max_depth:
            return

        # Get entries
        try:
            entries = list(current_path.iterdir())
        except PermissionError:
            return

        # Separate dirs and files
        dirs = []
        files = []
        for entry in entries:
            name = entry.name

            # Skip hidden and common non-essential dirs
            skip_patterns = {
                '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
                '.tox', '.venv', 'venv', 'node_modules', '.egg-info',
                '.coverage', 'htmlcov', '.hypothesis', '.serverless',
                '.next', '.nuxt', '.cache', '.parcel-cache', '.serverless',
            }
            if name in skip_patterns or name.startswith('.'):
                if entry.is_dir():
                    continue

            # Apply filter
            if filter_pattern and not self._matches_filter(name, filter_pattern):
                continue

            if entry.is_dir():
                dirs.append(entry)
            elif include_files:
                files.append(entry)

        # Sort
        dirs.sort(key=lambda x: x.name.lower())
        files.sort(key=lambda x: x.name.lower())

        # Build tree lines
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1 and len(files) == 0)
            connector = "└── " if is_last else "├── "
            prefix = "│   " * depth

            if depth == 0:
                lines.append(f"📁 {d.name}/")
            else:
                lines.append(f"{prefix}{connector}📁 {d.name}/")

            # Recurse into directory
            new_prefix = "    " if is_last else "│   "
            self._build_tree(
                base_path, d,
                depth + 1, max_depth,
                include_files, filter_pattern,
                lines
            )

        for i, f in enumerate(files):
            is_last = (i == len(files) - 1)
            connector = "└── " if is_last else "├── "
            prefix = "│   " * depth if depth > 0 else ""

            # Determine file icon
            ext = f.suffix
            icon = self._get_file_icon(ext)

            lines.append(f"{prefix}{connector}{icon} {f.name}")

    def _matches_filter(self, name: str, pattern: str) -> bool:
        """Check if filename matches filter pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def _get_file_icon(self, ext: str) -> str:
        """Get icon for file extension."""
        icons = {
            '.py': '🐍',
            '.js': '📜',
            '.ts': '📘',
            '.json': '📋',
            '.md': '📝',
            '.txt': '📃',
            '.yaml': '⚙️',
            '.yml': '⚙️',
            '.toml': '📦',
            '.cfg': '🔧',
            '.conf': '🔧',
        }
        return icons.get(ext, '📄')