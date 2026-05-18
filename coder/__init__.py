"""
Coder Module - Generates executable edit instructions from fix plans.

The Coder receives:
- Planner's fix plan (bug reason, target files, fix strategy)
- Target file contents
- Retrieved context
- Previous diff (for retry scenarios)

The Coder outputs:
- Structured edit instructions (replace/whole_file/line_edit)
- Ready for Editor to execute
"""

from .coder import Coder, CoderConfig
from .edit import EditInstruction, EditType, EditCollection
from .pipeline import CoderPipeline, CoderInput, CoderOutput
from .editor import Editor, EditorConfig, EditorResult

__all__ = [
    "Coder",
    "CoderConfig",
    "EditInstruction",
    "EditType",
    "EditCollection",
    "CoderPipeline",
    "CoderInput",
    "CoderOutput",
    "Editor",
    "EditorConfig",
    "EditorResult",
]