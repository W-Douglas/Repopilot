"""
Git tools: get_git_diff, rollback_file.
"""

import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional, List

from .base import BaseTool, ReadOnlyTool, WriteTool, ToolParameter
from .result import ToolResult, ToolStatus, create_success_result, create_error_result


class GetGitDiffTool(ReadOnlyTool):
    """
    Get git diff for changed files.

    Designed for:
    - Showing what files have been modified
    - Reviewing changes before committing
    - Debugging by showing recent changes

    Parameters:
        file_path: Specific file to diff (or None for all changes)
        staged: Show staged changes only
        untracked: Include untracked files
    """

    name = "get_git_diff"
    description = "Get git diff showing file changes. Use to review modifications."
    category = "git"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("file_path", str, required=False, default=None,
                          description="Specific file path (None for all changes)"),
            ToolParameter("staged", bool, required=False, default=False,
                          description="Show only staged changes"),
            ToolParameter("untracked", bool, required=False, default=True,
                          description="Include untracked files"),
        ]

    def execute(self, file_path: Optional[str] = None,
                staged: bool = False,
                untracked: bool = True) -> ToolResult:
        start = time.time()

        try:
            root = Path(self.root) if self.root else Path(".")

            # Check if git repo
            git_dir = root / ".git"
            if not git_dir.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"file_path": file_path, "staged": staged},
                    error="Not a git repository",
                    latency=time.time() - start
                )

            # Build git command
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")

            if file_path:
                cmd.append("--")
                cmd.append(file_path)

            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
            )

            diff_output = result.stdout

            # Get status for summary
            status_cmd = ["git", "status", "--porcelain"]
            status_result = subprocess.run(
                status_cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
            )

            changed_files = [
                line.strip() for line in status_result.stdout.split('\n')
                if line.strip()
            ]

            return create_success_result(
                tool=self.name,
                input_params={"file_path": file_path, "staged": staged},
                output={
                    "diff": diff_output,
                    "changed_files": changed_files,
                    "has_changes": len(changed_files) > 0,
                },
                latency=time.time() - start,
                metadata={
                    "num_changed": len(changed_files),
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"file_path": file_path},
                error=f"Git error: {str(e)}",
                latency=time.time() - start
            )


class RollbackFileTool(WriteTool):
    """
    Rollback a file to its git version.

    Designed for:
    - Discarding bad changes
    - Reverting to last known good state
    - Undoing accidental modifications

    Parameters:
        file_path: File to rollback
        create_backup: Create backup before rollback
    """

    name = "rollback_file"
    description = "Rollback a file to its git version. Use to undo unwanted changes."
    category = "git"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("file_path", str, required=True,
                          description="File path to rollback"),
            ToolParameter("create_backup", bool, required=False, default=True,
                          description="Create backup before rollback"),
        ]

    def execute(self, file_path: str,
                create_backup: bool = True) -> ToolResult:
        start = time.time()

        try:
            root = Path(self.root) if self.root else Path(".")
            target_path = root / file_path

            # Check if git repo
            git_dir = root / ".git"
            if not git_dir.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"file_path": file_path},
                    error="Not a git repository",
                    latency=time.time() - start
                )

            # Check if file exists
            if not target_path.exists():
                return create_error_result(
                    tool=self.name,
                    input_params={"file_path": file_path},
                    error=f"File not found: {file_path}",
                    latency=time.time() - start
                )

            # Create backup first
            backup_path = None
            if create_backup:
                backup_dir = target_path.parent / ".backups"
                backup_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{target_path.name}.{timestamp}.bak"
                shutil.copy2(target_path, backup_path)

            # Get original content from git
            result = subprocess.run(
                ["git", "checkout", "--", file_path],
                cwd=str(root),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return create_error_result(
                    tool=self.name,
                    input_params={"file_path": file_path},
                    error=f"Git checkout failed: {result.stderr}",
                    latency=time.time() - start
                )

            return create_success_result(
                tool=self.name,
                input_params={"file_path": file_path, "create_backup": create_backup},
                output=f"Rolled back: {file_path}",
                latency=time.time() - start,
                metadata={
                    "backup_created": backup_path is not None,
                    "backup_path": str(backup_path) if backup_path else None,
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"file_path": file_path},
                error=str(e),
                latency=time.time() - start
            )


class GetGitStatusTool(ReadOnlyTool):
    """
    Get git status of the repository.
    """

    name = "get_git_status"
    description = "Get git status showing modified, staged, and untracked files."
    category = "git"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return []

    def execute(self) -> ToolResult:
        start = time.time()

        try:
            root = Path(self.root) if self.root else Path(".")

            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                cwd=str(root),
                capture_output=True,
                text=True,
            )

            lines = result.stdout.split('\n')
            status = {
                "branch": "",
                "modified": [],
                "staged": [],
                "untracked": [],
                "clean": True,
            }

            for line in lines:
                if not line.strip():
                    continue

                # Parse porcelain format: XY filename
                if len(line) < 3:
                    continue

                state = line[:2]
                filename = line[3:]

                status["clean"] = False

                # Staged changes
                if state[0] != ' ' and state[0] != '?':
                    status["staged"].append(filename)

                # modifications
                if state[1] != ' ' and state[1] != '?':
                    status["modified"].append(filename)

                # Untracked
                if state == '??':
                    status["untracked"].append(filename)

                # Branch info
                if line.startswith('##'):
                    if '...' in line:
                        parts = line.split('...')
                        status["branch"] = parts[0].replace('## ', '').split('...')[0]
                    else:
                        status["branch"] = line.replace('## ', '').split('...')[0]

            return create_success_result(
                tool=self.name,
                input_params={},
                output=status,
                latency=time.time() - start,
                metadata={
                    "num_modified": len(status["modified"]),
                    "num_staged": len(status["staged"]),
                    "num_untracked": len(status["untracked"]),
                }
            )

        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={},
                error=str(e),
                latency=time.time() - start
            )