"""
File operation tools: read_file, write_file.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BaseTool, ReadOnlyTool, WriteTool, ToolParameter
from .result import ToolResult, ToolStatus, create_success_result, create_error_result


class ReadFileTool(ReadOnlyTool):
    """
    Read contents of a file.

    Designed for:
    - LLM to understand file structure
    - Retrieve function/class definitions
    - Read test files for context

    Parameters:
        path: Relative path to file from repo root
        max_lines: Maximum lines to read (default: 500)
        start_line: Starting line number (default: 0)
        highlight: Line numbers to highlight (for showing relevant code)
    """

    name = "read_file"
    description = "Read contents of a file. Use this to examine code, understand functions, or read test files."
    category = "file"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("path", str, required=True,
                          description="Relative file path from repository root"),
            ToolParameter("max_lines", int, required=False, default=500,
                          description="Maximum number of lines to read"),
            ToolParameter("start_line", int, required=False, default=0,
                          description="Starting line number (0-indexed)"),
        ]

    def execute(self, path: str, max_lines: int = 500,
                start_line: int = 0) -> ToolResult:
        start = time.time()

        try:
            # Resolve path
            file_path = Path(self.root) / path if self.root else Path(path)
            file_path = file_path.resolve()

            # Security check: ensure path is within root
            if self.root:
                root_path = Path(self.root).resolve()
                if not str(file_path).startswith(str(root_path)):
                    return create_error_result(
                        tool=self.name,
                        input_params={"path": path, "max_lines": max_lines},
                        error="Access denied: Path outside repository root",
                        status=ToolStatus.PERMISSION_DENIED,
                        latency=time.time() - start
                    )

            # Check file exists
            if not file_path.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"path": path},
                    error=f"File not found: {path}",
                    status=ToolStatus.FAILURE,
                    latency=time.time() - start
                )

            # Check if regular file
            if not file_path.is_file():
                return create_error_result(
                    tool=self.name,
                    input_params={"path": path},
                    error=f"Not a regular file: {path}",
                    status=ToolStatus.FAILURE,
                    latency=time.time() - start
                )

            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Apply pagination
            total_lines = len(lines)
            end_line = min(start_line + max_lines, total_lines)
            content = ''.join(lines[start_line:end_line])

            metadata = {
                "total_lines": total_lines,
                "read_lines": end_line - start_line,
                "truncated": end_line < total_lines,
                "start_line": start_line,
                "end_line": end_line,
            }

            return create_success_result(
                tool=self.name,
                input_params={"path": path, "max_lines": max_lines, "start_line": start_line},
                output=content,
                latency=time.time() - start,
                metadata=metadata
            )

        except UnicodeDecodeError:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error="Cannot read file: Invalid encoding",
                latency=time.time() - start
            )
        except PermissionError:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error="Access denied: Permission error",
                status=ToolStatus.PERMISSION_DENIED,
                latency=time.time() - start
            )
        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error=f"Error reading file: {str(e)}",
                latency=time.time() - start
            )


class WriteFileTool(WriteTool):
    """
    Write or modify a file.

    Designed for:
    - Saving LLM-generated code changes
    - Creating new files
    - Modifying existing files

    Parameters:
        path: Relative path to file
        content: Full file content (whole-file write)
        create_backup: Whether to create backup before writing
        create_if_missing: Create file if it doesn't exist
    """

    name = "write_file"
    description = "Write or modify a file. Use this to save code changes. WARNING: This overwrites the entire file."
    category = "file"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("path", str, required=True,
                          description="Relative file path from repository root"),
            ToolParameter("content", str, required=True,
                          description="Full file content to write"),
            ToolParameter("create_backup", bool, required=False, default=True,
                          description="Create backup before writing"),
            ToolParameter("create_if_missing", bool, required=False, default=True,
                          description="Create file if it doesn't exist"),
        ]

    def execute(self, path: str, content: str,
                create_backup: bool = True,
                create_if_missing: bool = True) -> ToolResult:
        start = time.time()

        try:
            file_path = Path(self.root) / path if self.root else Path(path)
            file_path = file_path.resolve()

            # Security check
            if self.root:
                root_path = Path(self.root).resolve()
                if not str(file_path).startswith(str(root_path)):
                    return create_error_result(
                        tool=self.name,
                        input_params={"path": path, "content_length": len(content)},
                        error="Access denied: Path outside repository root",
                        status=ToolStatus.PERMISSION_DENIED,
                        latency=time.time() - start
                    )

            # Check if file exists
            exists = file_path.exists()

            if not exists and not create_if_missing:
                return create_error_result(
                    tool=self.name,
                    input_params={"path": path},
                    error=f"File does not exist: {path}",
                    latency=time.time() - start
                )

            # Create backup if file exists and backup requested
            backup_path = None
            if exists and create_backup:
                backup_dir = file_path.parent / ".backups"
                backup_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{file_path.name}.{timestamp}.bak"

                import shutil
                shutil.copy2(file_path, backup_path)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            metadata = {
                "backup_created": backup_path is not None,
                "backup_path": str(backup_path) if backup_path else None,
                "bytes_written": len(content.encode('utf-8')),
                "lines_written": content.count('\n') + 1,
                "created": not exists,
            }

            return create_success_result(
                tool=self.name,
                input_params={"path": path, "content_length": len(content)},
                output=f"File {'created' if not exists else 'updated'}: {path}",
                latency=time.time() - start,
                metadata=metadata
            )

        except PermissionError:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error="Access denied: Permission error",
                status=ToolStatus.PERMISSION_DENIED,
                latency=time.time() - start
            )
        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error=f"Error writing file: {str(e)}",
                latency=time.time() - start
            )


class ReadLinesTool(ReadOnlyTool):
    """
    Read specific lines from a file (for showing function body).

    Designed for:
    - Getting specific function/method content
    - Reading a class definition
    - Accessing specific sections of large files
    """

    name = "read_lines"
    description = "Read specific lines from a file by line numbers."
    category = "file"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("path", str, required=True,
                          description="Relative file path"),
            ToolParameter("start", int, required=True,
                          description="Start line number (1-indexed)"),
            ToolParameter("end", int, required=True,
                          description="End line number (inclusive)"),
        ]

    def execute(self, path: str, start: int, end: int) -> ToolResult:
        start_time = time.time()

        try:
            file_path = Path(self.root) / path if self.root else Path(path)

            if not file_path.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"path": path, "start": start, "end": end},
                    error=f"File not found: {path}",
                    latency=time.time() - start_time
                )

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Convert to 0-indexed
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)

            if start_idx >= len(lines):
                return create_error_result(
                    tool=self.name,
                    input_params={"path": path, "start": start, "end": end},
                    error=f"Start line {start} beyond file length ({len(lines)})",
                    latency=time.time() - start_time
                )

            content = ''.join(lines[start_idx:end_idx])

            return create_success_result(
                tool=self.name,
                input_params={"path": path, "start": start, "end": end},
                output=content,
                latency=time.time() - start_time,
                metadata={
                    "lines_read": end_idx - start_idx,
                    "actual_start": start + 1,
                    "actual_end": end,
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"path": path},
                error=str(e),
                latency=time.time() - start_time
            )