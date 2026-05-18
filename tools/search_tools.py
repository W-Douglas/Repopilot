"""
Search tools: search_code - search for code patterns.
"""

import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BaseTool, ReadOnlyTool, ToolParameter
from .result import ToolResult, create_success_result, create_error_result


class SearchCodeTool(ReadOnlyTool):
    """
    Search for code patterns in files.

    Designed for:
    - Finding function/class definitions
    - Locating import statements
    - Finding usage of specific identifiers

    Parameters:
        pattern: Regex or simple string pattern to search
        file_pattern: Optional glob pattern to filter files (e.g., "*.py")
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    """

    name = "search_code"
    description = "Search for code patterns in repository files. Use to find functions, classes, or specific code."
    category = "search"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("pattern", str, required=True,
                          description="Search pattern (regex or string)"),
            ToolParameter("file_pattern", str, required=False, default="*.py",
                          description="File pattern to search (glob)"),
            ToolParameter("case_sensitive", bool, required=False, default=False,
                          description="Case-sensitive search"),
            ToolParameter("max_results", int, required=False, default=50,
                          description="Maximum number of matches"),
            ToolParameter("search_in", List, required=False, default=None,
                          description="Specific files/directories to search"),
        ]

    def execute(self, pattern: str, file_pattern: str = "*.py",
                case_sensitive: bool = False,
                max_results: int = 50,
                search_in: Optional[List[str]] = None) -> ToolResult:
        start = time.time()

        try:
            results = []
            root = Path(self.root) if self.root else Path(".")

            # Determine which directories to search
            if search_in:
                paths_to_search = []
                for p in search_in:
                    path = root / p if not Path(p).is_absolute() else Path(p)
                    paths_to_search.append(path)
            else:
                paths_to_search = [root]

            # Compile regex
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error:
                # Fall back to literal search
                pattern = re.escape(pattern)
                regex = re.compile(pattern, flags)

            # Search files
            for base_path in paths_to_search:
                if not base_path.exists():
                    continue

                for file_path in base_path.rglob(file_pattern):
                    # Skip certain directories
                    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules',
                                 '.venv', 'venv', '.tox', '.mypy_cache'}
                    if any(skip in file_path.parts for skip in skip_dirs):
                        continue

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                match = regex.search(line)
                                if match:
                                    rel_path = str(file_path.relative_to(root))
                                    results.append({
                                        "file": rel_path,
                                        "line": line_num,
                                        "content": line.rstrip(),
                                        "match_start": match.start(),
                                        "match_end": match.end(),
                                    })

                                    if len(results) >= max_results:
                                        break

                    except (PermissionError, IsADirectoryError):
                        continue

                    if len(results) >= max_results:
                        break

                if len(results) >= max_results:
                    break

            output = {
                "total_matches": len(results),
                "results": results,
                "pattern": pattern,
            }

            return create_success_result(
                tool=self.name,
                input_params={
                    "pattern": pattern,
                    "file_pattern": file_pattern,
                    "max_results": max_results,
                },
                output=output,
                latency=time.time() - start,
                metadata={
                    "files_searched": len(list(root.rglob(file_pattern))) if search_in is None else len(search_in),
                    "truncated": len(results) >= max_results,
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"pattern": pattern, "file_pattern": file_pattern},
                error=f"Search error: {str(e)}",
                latency=time.time() - start
            )


class GrepTool(ReadOnlyTool):
    """
    Grep-like search for text in files.

    Simpler than SearchCodeTool - plain text search without regex.
    """

    name = "grep"
    description = "Plain text search in files (like grep)."
    category = "search"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("pattern", str, required=True,
                          description="Text to search for"),
            ToolParameter("path", str, required=False, default=".",
                          description="Directory to search in"),
            ToolParameter("file_type", str, required=False, default=None,
                          description="File extension filter (e.g., 'py')"),
            ToolParameter("max_lines", int, required=False, default=100,
                          description="Maximum output lines"),
        ]

    def execute(self, pattern: str, path: str = ".",
                file_type: Optional[str] = None,
                max_lines: int = 100) -> ToolResult:
        start = time.time()

        try:
            results = []
            search_path = Path(self.root) / path if self.root else Path(path)

            if not search_path.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"pattern": pattern, "path": path},
                    error=f"Path not found: {path}",
                    latency=time.time() - start
                )

            if file_type:
                pattern_glob = f"*.{file_type}"
            else:
                pattern_glob = "*"

            for file_path in search_path.rglob(pattern_glob):
                # Skip hidden directories
                if any(p.startswith('.') for p in file_path.parts):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern in line:
                                rel_path = str(file_path.relative_to(search_path.parent))
                                results.append(f"{rel_path}:{line_num}:{line.rstrip()}")

                                if len(results) >= max_lines:
                                    break

                except (PermissionError, IsADirectoryError):
                    continue

                if len(results) >= max_lines:
                    break

            return create_success_result(
                tool=self.name,
                input_params={"pattern": pattern, "path": path, "file_type": file_type},
                output="\n".join(results) if results else "No matches found",
                latency=time.time() - start,
                metadata={
                    "total_matches": len(results),
                    "truncated": len(results) >= max_lines,
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"pattern": pattern, "path": path},
                error=str(e),
                latency=time.time() - start
            )