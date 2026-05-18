"""
Coder - Generates executable edit instructions from fix plans.

Following the design in design_documents/Coder.md:
- Receives Planner's fix_plan + target_file_contents + context
- Outputs structured edit instructions (replace/whole_file)
- Constrains modifications to Planner's target files
- Prefers minimal edits (replace over whole_file)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import re

from .edit import EditInstruction, EditType, EditCollection

# Import ChatMessage - try relative first, then absolute
try:
    from ..llm import ChatMessage
except ImportError:
    from llm import ChatMessage

if TYPE_CHECKING:
    from ..llm import BaseLLM
    from ..planner.state import PlanState


@dataclass
class CoderConfig:
    """Configuration for the Coder module."""
    # Preferred edit type: "replace" (recommended) or "whole_file"
    preferred_edit_type: str = "replace"

    # Maximum number of edits per file
    max_edits_per_file: int = 10

    # Allow modifying test files (default: no)
    allow_test_modification: bool = False

    # Allow modifying files outside Planner's target list
    allow_extra_files: bool = False

    # LLM temperature for code generation
    temperature: float = 0.3

    # Include context in LLM prompts
    include_context: bool = True


class Coder:
    """
    Generates edit instructions from Planner's fix plan.

    The Coder's role (per design_documents/Coder.md):
    1. Convert fix_plan + target_file_contents → edit instructions
    2. Control modification scope (only target_files)
    3. Prefer minimal edits (replace over whole_file)
    4. Output structured, machine-executable format

    Example input:
        {
            "issue": "pytest test_divide failed",
            "fix_plan": {
                "bug_reason": "divide uses integer division",
                "target_files": ["calculator.py"],
                "fix_strategy": "change // to /"
            },
            "target_file_contents": {"calculator.py": "..."},
            "context": [...]
        }

    Example output:
        {
            "edits": [{
                "edit_type": "replace",
                "file_path": "calculator.py",
                "old_text": "return a // b",
                "new_text": "return a / b",
                "reason": "Fix integer division"
            }],
            "summary": "...",
            "risk_level": "low"
        }
    """

    name = "coder"
    description = "Generates edit instructions from fix plans"

    def __init__(self, config: Optional[CoderConfig] = None, llm: Optional["BaseLLM"] = None):
        """
        Initialize the Coder.

        Args:
            config: Coder configuration
            llm: Optional LLM instance for complex edits
        """
        self.config = config or CoderConfig()
        self.llm = llm

    def generate_edits(
        self,
        issue: str,
        fix_plan: Dict[str, Any],
        target_file_contents: Dict[str, str],
        context: Optional[List[Dict[str, Any]]] = None,
        previous_diff: Optional[str] = None,
        test_error: Optional[str] = None,
        retry_count: int = 0,
    ) -> EditCollection:
        """
        Generate edit instructions from a fix plan.

        This is the main entry point for the Coder module.

        Args:
            issue: The original issue/bug description
            fix_plan: Planner's fix plan containing bug_reason, target_files, fix_strategy
            target_file_contents: Dict mapping file_path → file_content
            context: Retrieved code context (optional)
            previous_diff: Previous diff if this is a retry
            test_error: Test error message if retry
            retry_count: Number of retry attempts

        Returns:
            EditCollection with all generated edit instructions
        """
        collection = EditCollection()

        # Extract key information from fix_plan
        bug_reason = fix_plan.get("bug_reason", "")
        target_files = fix_plan.get("target_files", [])
        fix_strategy = fix_plan.get("fix_strategy", "")
        target_symbols = fix_plan.get("target_symbols", [])

        # Set summary
        collection.summary = f"Fix: {bug_reason[:100]}"
        collection.risk_level = fix_plan.get("risk_level", "low")

        # Validate target files
        valid_files = self._filter_valid_files(target_files, target_file_contents)

        if not valid_files:
            collection.metadata["error"] = "No valid target files"
            return collection

        # Use LLM if available, otherwise use rule-based generation
        if self.llm:
            edits = self._generate_with_llm(
                issue, fix_plan, target_file_contents, context,
                previous_diff, test_error, retry_count
            )
            for edit in edits:
                try:
                    collection.add_edit(edit)
                except ValueError as e:
                    collection.metadata.setdefault("skipped_edits", []).append(str(e))
        else:
            edits = self._generate_rule_based(
                bug_reason, fix_strategy, valid_files, target_file_contents, target_symbols
            )
            for edit in edits:
                try:
                    collection.add_edit(edit)
                except ValueError as e:
                    collection.metadata.setdefault("skipped_edits", []).append(str(e))

        collection.metadata["target_files_from_planner"] = target_files
        collection.metadata["edits_generated"] = len(collection.edits)

        return collection

    def _filter_valid_files(
        self,
        target_files: List[str],
        file_contents: Dict[str, str]
    ) -> List[str]:
        """Filter to only files that exist in file_contents."""
        valid = []
        for f in target_files:
            if f in file_contents:
                valid.append(f)
            elif self.config.allow_extra_files:
                # Check if file exists on disk
                pass  # Could add disk check here
        return valid

    def _generate_rule_based(
        self,
        bug_reason: str,
        fix_strategy: str,
        target_files: List[str],
        file_contents: Dict[str, str],
        target_symbols: List[str],
    ) -> List[EditInstruction]:
        """
        Generate edits using rule-based pattern matching.

        This is used when no LLM is available.
        It analyzes the bug reason and strategy to find the exact code to change.
        """
        edits = []

        # Normalize the fix strategy
        strategy_lower = fix_strategy.lower()

        for file_path in target_files:
            content = file_contents.get(file_path, "")
            if not content:
                continue

            # Pattern-based fixes for common security issues
            edits.extend(self._generate_security_fixes(file_path, content, bug_reason, strategy_lower))

        return edits

    def _generate_security_fixes(
        self,
        file_path: str,
        content: str,
        bug_reason: str,
        strategy: str,
    ) -> List[EditInstruction]:
        """
        Generate edit instructions for security-related fixes.

        This implements rule-based fixes for the security issues
        identified in the Planner's output.
        """
        edits = []

        # Check content for security issues (not just bug_reason)
        has_sha256 = 'sha256' in content.lower()
        has_password_comparison = 'self.password == password' in content or 'password == self.password' in content
        has_permission_check = 'def check_permission' in content and 'return True' in content

        # SHA256 password hashing → PBKDF2
        if has_sha256 and 'hashlib.sha256(password.encode()).hexdigest()' in content:
            new_hash = '''import hashlib
import secrets
salt = secrets.token_hex(16)
hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ':' + salt'''
            edits.append(EditInstruction(
                edit_type=EditType.REPLACE,
                file_path=file_path,
                old_text='hashlib.sha256(password.encode()).hexdigest()',
                new_text=new_hash,
                reason="Replace SHA256 with PBKDF2 for secure password hashing",
                risk_level="medium"
            ))

        # Plain text password comparison
        if has_password_comparison:
            old_text = "self.password == password"
            if old_text in content:
                edits.append(EditInstruction(
                    edit_type=EditType.REPLACE,
                    file_path=file_path,
                    old_text=old_text,
                    new_text="self._verify_password(password)",
                    reason="Use secure password verification method",
                    risk_level="medium"
                ))

        # Check permission always returns True
        if has_permission_check:
            # Try to find and replace return True in check_permission
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'def check_permission' in line:
                    # Find next return True
                    for j in range(i, min(i+10, len(lines))):
                        if 'return True' in lines[j] and not lines[j].strip().startswith('#'):
                            edits.append(EditInstruction(
                                edit_type=EditType.REPLACE,
                                file_path=file_path,
                                old_text='return True',
                                new_text='return self.role == "admin"',
                                reason="Fix permission check to verify admin role",
                                risk_level="medium"
                            ))
                            break
                    break

        return edits

    def _generate_with_llm(
        self,
        issue: str,
        fix_plan: Dict[str, Any],
        file_contents: Dict[str, str],
        context: Optional[List[Dict[str, Any]]],
        previous_diff: Optional[str],
        test_error: Optional[str],
        retry_count: int,
    ) -> List[EditInstruction]:
        """Generate edits using LLM for complex fixes."""
        # Build prompt for LLM
        prompt = self._build_llm_prompt(
            issue, fix_plan, file_contents, context,
            previous_diff, test_error, retry_count
        )

        messages = [
            ChatMessage(role="system", content=self._get_system_prompt()),
            ChatMessage(role="user", content=prompt),
        ]

        try:
            response = self.llm.chat(messages, temperature=self.config.temperature)
            return self._parse_llm_response(response.content, file_contents)
        except Exception as e:
            # Fall back to rule-based if LLM fails
            return self._generate_rule_based(
                fix_plan.get("bug_reason", ""),
                fix_plan.get("fix_strategy", ""),
                list(file_contents.keys()),
                file_contents,
                fix_plan.get("target_symbols", [])
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM-based edit generation."""
        return """You are an expert code editor. Your task is to convert fix plans into precise edit instructions.

You must output valid JSON in this exact format:
{
    "edits": [
        {
            "edit_type": "replace" or "whole_file",
            "file_path": "relative/path/to/file.py",
            "old_text": "exact text to replace (for replace mode)",
            "new_text": "new text (for replace mode)",
            "new_content": "full file content (for whole_file mode)",
            "reason": "why this edit is needed",
            "risk_level": "low/medium/high"
        }
    ],
    "summary": "brief summary of changes"
}

Rules:
1. Use "replace" mode for small, targeted changes
2. Use "whole_file" mode only for large refactoring
3. old_text must match EXACTLY (watch for whitespace)
4. Do NOT modify files not in the target list
5. Prefer minimal edits over wholesale replacement
6. Output ONLY JSON, no explanatory text before or after"""

    def _build_llm_prompt(
        self,
        issue: str,
        fix_plan: Dict[str, Any],
        file_contents: Dict[str, str],
        context: Optional[List[Dict[str, Any]]],
        previous_diff: Optional[str],
        test_error: Optional[str],
        retry_count: int,
    ) -> str:
        """Build the prompt for LLM edit generation."""
        parts = [f"# Issue\n{issue}\n"]
        parts.append(f"\n# Fix Plan\n")

        if retry_count > 0:
            parts.append(f"(Retry #{retry_count})\n")

        parts.append(f"- Bug Reason: {fix_plan.get('bug_reason', 'N/A')}")
        parts.append(f"- Target Files: {', '.join(fix_plan.get('target_files', []))}")
        parts.append(f"- Strategy: {fix_plan.get('fix_strategy', 'N/A')}")

        if previous_diff:
            parts.append(f"\n# Previous Attempt (failed)\n{previous_diff}")

        if test_error:
            parts.append(f"\n# Test Error\n{test_error}")

        parts.append("\n# Target File Contents")
        for file_path, content in file_contents.items():
            parts.append(f"\n## {file_path}\n```python\n{content}\n```")

        if context and self.config.include_context:
            parts.append("\n# Retrieved Context")
            for ctx in context:
                parts.append(f"\n- {ctx.get('file_path', 'unknown')}: {ctx.get('symbols', [])}")

        return "\n".join(parts)

    def _parse_llm_response(
        self,
        response: str,
        file_contents: Dict[str, str]
    ) -> List[EditInstruction]:
        """Parse LLM response into EditInstruction objects."""
        import json

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_str = response.strip()
            if not json_str.startswith("{"):
                # Find the first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        edits = []
        for edit_data in data.get("edits", []):
            try:
                edit = EditInstruction.from_dict(edit_data)

                # Validate the edit refers to a valid target file
                if edit.file_path in file_contents:
                    valid, error = edit.validate()
                    if valid:
                        edits.append(edit)
            except Exception:
                continue

        return edits

    def generate_from_plan_state(
        self,
        plan_state: "PlanState",
        file_contents: Dict[str, str],
    ) -> EditCollection:
        """
        Generate edits from a PlanState object.

        Convenience method that wraps plan_state data for the Coder.

        Args:
            plan_state: Planner's PlanState object
            file_contents: Dict mapping file_path → file_content

        Returns:
            EditCollection with all generated edit instructions
        """
        # Convert PlanState to fix_plan format
        fix_plan = {
            "bug_reason": plan_state.thinking[:500] if plan_state.thinking else "See plan",
            "target_files": list(set(step.target for step in plan_state.plan)),
            "fix_strategy": "; ".join(s.description for s in plan_state.plan[:3]),
            "target_symbols": [],
            "risk_level": "medium",
        }

        return self.generate_edits(
            issue=plan_state.task,
            fix_plan=fix_plan,
            target_file_contents=file_contents,
            context=plan_state.context,
        )