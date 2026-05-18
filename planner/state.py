"""
Plan State Management - tracks planning state across turns
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


class TaskType(Enum):
    """Types of tasks the planner can handle."""
    MODIFY = "modify"  # Modify existing code
    CREATE = "create"  # Create new files/components
    REFACTOR = "refactor"  # Refactor existing code
    DEBUG = "debug"  # Debug/analyze issues
    REVIEW = "review"  # Code review
    EXPLAIN = "explain"  # Explain code
    TEST = "test"  # Generate tests
    UNKNOWN = "unknown"


class PlanStatus(Enum):
    """Status of the current plan."""
    INITIAL = "initial"  # Plan not yet started
    THINKING = "thinking"  # Currently thinking/planning
    PLANNED = "planned"  # Plan completed, ready to execute
    EXECUTING = "executing"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    REVISED = "revised"  # Plan was revised


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_id: int
    description: str
    action: str  # "read_file", "write_file", "search_code", etc.
    target: str  # File path or search query
    dependencies: List[int] = field(default_factory=list)  # IDs of dependent steps
    status: str = "pending"  # pending, in_progress, completed, skipped, failed
    result: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 1.0  # 0-1, how confident we are about this step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "target": self.target,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(**data)


@dataclass
class PlanState:
    """
    Maintains the state of the planning and execution process.

    Attributes:
        task: The original user task description
        task_type: Classified type of the task
        context: Relevant code context retrieved from RAG
        plan: List of planned steps
        status: Current planning/execution status
        history: Conversation history for context
        metadata: Additional metadata about the session
    """
    task: str
    task_type: TaskType = TaskType.UNKNOWN
    context: List[Dict[str, Any]] = field(default_factory=list)  # Retrieved code chunks
    plan: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.INITIAL
    thinking: str = ""  # The planner's reasoning/thinking
    history: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user", "content": "..."}]
    current_step: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_context(self, chunks: List[Dict[str, Any]]) -> None:
        """Add retrieved code chunks as context."""
        self.context.extend(chunks)

    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.plan.append(step)

    def add_history(self, role: str, content: str) -> None:
        """Add to conversation history."""
        self.history.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()

    def update_step(self, step_id: int, **kwargs) -> None:
        """Update a step's attributes."""
        for step in self.plan:
            if step.step_id == step_id:
                for key, value in kwargs.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                break
        self.updated_at = datetime.now().isoformat()

    def get_next_pending_step(self) -> Optional[PlanStep]:
        """Get the next step that hasn't been executed."""
        for step in self.plan:
            if step.status == "pending":
                # Check dependencies
                deps_done = all(
                    self.plan[s-1].status == "completed"
                    for s in step.dependencies
                    if s <= len(self.plan)
                )
                if deps_done:
                    return step
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the plan status."""
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "skipped": 0, "failed": 0}
        for step in self.plan:
            status_counts[step.status] = status_counts.get(step.status, 0) + 1

        return {
            "task": self.task[:100],
            "task_type": self.task_type.value,
            "status": self.status.value,
            "total_steps": len(self.plan),
            "step_status": status_counts,
            "current_step": self.current_step,
            "error": self.error_message,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "task_type": self.task_type.value,
            "context": self.context,
            "plan": [s.to_dict() for s in self.plan],
            "status": self.status.value,
            "thinking": self.thinking,
            "history": self.history,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanState":
        """Create from dictionary."""
        if isinstance(data.get("task_type"), str):
            data["task_type"] = TaskType(data["task_type"])
        if isinstance(data.get("status"), str):
            data["status"] = PlanStatus(data["status"])
        if "plan" in data:
            data["plan"] = [PlanStep.from_dict(s) for s in data["plan"]]
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "PlanState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def reset(self) -> None:
        """Reset the plan for a new task."""
        self.plan.clear()
        self.context.clear()
        self.history.clear()
        self.status = PlanStatus.INITIAL
        self.thinking = ""
        self.current_step = 0
        self.error_message = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()