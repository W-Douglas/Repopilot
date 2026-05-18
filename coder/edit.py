"""
Edit Instruction Types - Defines how edits are represented and generated.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class EditType(Enum):
    """Types of edit operations."""
    WHOLE_FILE = "whole_file"  # Replace entire file content
    REPLACE = "replace"       # Replace specific text snippet
    LINE_EDIT = "line_edit"   # Edit specific line range
    PATCH = "patch"           # Unified diff patch (future)


@dataclass
class EditInstruction:
    """
    A single edit instruction to be executed by Editor.

    Attributes:
        edit_type: Type of edit (replace, whole_file, etc.)
        file_path: Target file path (relative to repo root)
        old_text: Text to be replaced (for replace mode)
        new_text: New text to insert (for replace/line_edit)
        new_content: Full new file content (for whole_file mode)
        start_line: Start line for line_edit mode
        end_line: End line for line_edit mode
        reason: Why this edit is needed
        risk_level: low/medium/high
        hash: Hash of old_text for verification
    """
    edit_type: EditType
    file_path: str
    reason: str
    risk_level: str = "low"

    # For replace mode
    old_text: Optional[str] = None
    new_text: Optional[str] = None

    # For whole_file mode
    new_content: Optional[str] = None

    # For line_edit mode
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    # For patch mode (future)
    patch: Optional[str] = None

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate the edit instruction.

        Returns:
            (is_valid, error_message)
        """
        if not self.file_path:
            return False, "file_path is required"

        if self.edit_type == EditType.REPLACE:
            if not self.old_text:
                return False, "old_text is required for replace mode"
            if self.new_text is None:
                return False, "new_text is required for replace mode"
            if self.old_text == self.new_text:
                return False, "old_text and new_text are identical (no-op)"

        elif self.edit_type == EditType.WHOLE_FILE:
            if not self.new_content:
                return False, "new_content is required for whole_file mode"

        elif self.edit_type == EditType.LINE_EDIT:
            if self.start_line is None or self.end_line is None:
                return False, "start_line and end_line are required for line_edit mode"
            if self.start_line < 1:
                return False, "start_line must be >= 1"
            if self.end_line < self.start_line:
                return False, "end_line must be >= start_line"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "edit_type": self.edit_type.value,
            "file_path": self.file_path,
            "reason": self.reason,
            "risk_level": self.risk_level,
        }

        if self.old_text is not None:
            result["old_text"] = self.old_text
        if self.new_text is not None:
            result["new_text"] = self.new_text
        if self.new_content is not None:
            result["new_content"] = self.new_content
        if self.start_line is not None:
            result["start_line"] = self.start_line
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.patch is not None:
            result["patch"] = self.patch

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditInstruction":
        """Create from dictionary."""
        edit_type = data.get("edit_type", "replace")
        if isinstance(edit_type, str):
            edit_type = EditType(edit_type)

        return cls(
            edit_type=edit_type,
            file_path=data["file_path"],
            old_text=data.get("old_text"),
            new_text=data.get("new_text"),
            new_content=data.get("new_content"),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            reason=data.get("reason", ""),
            risk_level=data.get("risk_level", "low"),
            patch=data.get("patch"),
        )


@dataclass
class EditCollection:
    """
    Collection of edit instructions that together fix a bug.

    The Coder outputs this structure containing all edits
    needed to fix the issues identified by the Planner.
    """
    edits: List[EditInstruction] = field(default_factory=list)
    summary: str = ""
    risk_level: str = "low"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_edit(self, edit: EditInstruction) -> None:
        """Add an edit instruction to the collection."""
        valid, error = edit.validate()
        if not valid:
            raise ValueError(f"Invalid edit instruction: {error}")
        self.edits.append(edit)

    def is_valid(self) -> tuple[bool, List[str]]:
        """
        Validate all edits in the collection.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        for i, edit in enumerate(self.edits):
            valid, error = edit.validate()
            if not valid:
                errors.append(f"Edit {i+1}: {error}")

        return len(errors) == 0, errors

    def get_target_files(self) -> List[str]:
        """Get list of all files that will be modified."""
        return list(set(e.file_path for e in self.edits))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edits": [e.to_dict() for e in self.edits],
            "summary": self.summary,
            "risk_level": self.risk_level,
            "metadata": self.metadata,
            "target_files": self.get_target_files(),
            "edit_count": len(self.edits),
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)