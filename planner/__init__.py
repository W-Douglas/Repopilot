"""
Planner Module - Think and Execute Separation

This module implements the planning pattern inspired by aider's architect-coder,
separating the planning (thinking) and execution phases:

1. Planner: Analyzes the task, generates a plan
2. Executor: Executes the plan with tool calls

Key features:
- State management for multi-turn conversations
- Integration with Retriever for code context
- Structured output parsing
- Streaming support
"""

from .state import PlanState, PlanStep, TaskType, PlanStatus
from .planner import Planner, PlanResult
from .executor import Executor, ExecutionResult
from .pipeline import PlannerPipeline, PipelineConfig, PipelineResult

__all__ = [
    "PlanState",
    "PlanStep",
    "TaskType",
    "PlanStatus",
    "Planner",
    "PlanResult",
    "Executor",
    "ExecutionResult",
    "PlannerPipeline",
    "PipelineConfig",
    "PipelineResult",
]