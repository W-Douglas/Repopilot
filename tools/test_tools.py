"""
Test execution tools: run_pytest.
"""

import subprocess
import time
import json
from pathlib import Path
from typing import Optional, List

from .base import BaseTool, ReadOnlyTool, ToolParameter
from .result import ToolResult, ToolStatus, create_success_result, create_error_result


class RunPytestTool(ReadOnlyTool):
    """
    Execute pytest tests.

    Designed for:
    - Verifying code changes
    - Running test suites
    - Getting test results for debugging

    Parameters:
        test_path: Specific test file or directory to run
        markers: Optional pytest markers to filter (e.g., "unit", "integration")
        verbose: Enable verbose output
        capture_output: Capture stdout/stderr
    """

    name = "run_pytest"
    description = "Run pytest tests to verify code changes. Returns detailed test results."
    category = "test"

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter("test_path", str, required=False, default=".",
                          description="Test path (file, directory, or test name pattern)"),
            ToolParameter("verbose", bool, required=False, default=True,
                          description="Verbose output"),
            ToolParameter("capture_output", bool, required=False, default=True,
                          description="Capture stdout/stderr"),
            ToolParameter("timeout", int, required=False, default=300,
                          description="Max execution time in seconds"),
        ]

    def execute(self, test_path: str = ".",
                verbose: bool = True,
                capture_output: bool = True,
                timeout: int = 300) -> ToolResult:
        start = time.time()

        try:
            root = Path(self.root) if self.root else Path(".")

            # Build pytest command
            cmd = ["python", "-m", "pytest", test_path]

            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            # Add output format
            cmd.extend(["--tb=short"])

            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )

            elapsed = time.time() - start

            # Parse output
            output_data = {
                "exit_code": result.returncode,
                "passed": result.returncode == 0,
                "stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
            }

            # Try to parse pytest summary
            summary = self._parse_pytest_output(result.stdout + result.stderr)
            output_data.update(summary)

            success = result.returncode == 0
            status = ToolStatus.SUCCESS if success else ToolStatus.FAILURE

            return ToolResult(
                tool=self.name,
                success=success,
                input={"test_path": test_path, "verbose": verbose},
                output=output_data,
                latency=elapsed,
                status=status,
            )

        except subprocess.TimeoutExpired:
            return create_error_result(
                tool=self.name,
                input_params={"test_path": test_path},
                error=f"Test timeout after {timeout} seconds",
                status=ToolStatus.TIMEOUT,
                latency=time.time() - start
            )
        except FileNotFoundError:
            return create_error_result(
                tool=self.name,
                input_params={"test_path": test_path},
                error="pytest not found. Install with: pip install pytest",
                latency=time.time() - start
            )
        except Exception as e:
            return create_error_result(
                tool=self.name,
                input_params={"test_path": test_path},
                error=f"Test execution error: {str(e)}",
                latency=time.time() - start
            )

    def _parse_pytest_output(self, output: str) -> dict:
        """Parse pytest output to extract summary."""
        summary = {
            "summary": "",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_tests": [],
        }

        # Look for summary line like "2 passed, 1 failed in 0.5s"
        import re

        # Pattern: "X passed, Y failed, Z skipped"
        match = re.search(r'(\d+)\s+passed', output)
        if match:
            summary["passed"] = int(match.group(1))
            summary["total_tests"] += int(match.group(1))

        match = re.search(r'(\d+)\s+failed', output)
        if match:
            summary["failed"] = int(match.group(1))
            summary["total_tests"] += int(match.group(1))

        match = re.search(r'(\d+)\s+skipped', output)
        if match:
            summary["skipped"] = int(match.group(1))

        # Extract failed test names
        failed_section = re.findall(r'FAILED\s+(.+)', output)
        summary["failed_tests"] = failed_section

        # Build summary string
        parts = []
        if summary["passed"] > 0:
            parts.append(f"{summary['passed']} passed")
        if summary["failed"] > 0:
            parts.append(f"{summary['failed']} failed")
        if summary["skipped"] > 0:
            parts.append(f"{summary['skipped']} skipped")

        if parts:
            summary["summary"] = ", ".join(parts)

        return summary


class RunTestTool(RunPytestTool):
    """Alias for RunPytestTool with different name for clarity."""
    pass