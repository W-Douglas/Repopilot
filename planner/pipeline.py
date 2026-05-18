"""
Planner Pipeline - Combines Planner and Executor for end-to-end task execution

This module provides the main entry point that orchestrates:
1. Planning (think phase)
2. Context retrieval
3. Execution (execute phase)
4. Error handling and recovery

Supports seamless model switching via the LLM module.
"""
from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import time

from .state import PlanState, PlanStep, TaskType, PlanStatus
from .planner import Planner, PlanResult
from .executor import Executor, ExecutionResult, ToolExecutor
from .prompts import PlannerPrompts, default_prompts


@dataclass
class PipelineConfig:
    """Configuration for the planner pipeline."""
    # Planning settings
    auto_plan: bool = True
    max_plan_steps: int = 20
    allow_revision: bool = True
    max_revisions: int = 3

    # Execution settings
    auto_execute: bool = True
    stop_on_error: bool = True
    retry_on_failure: bool = True
    max_retries: int = 2

    # Context settings
    retrieval_top_k: int = 10
    max_context_tokens: int = 4000

    # Model settings - supports multiple providers
    model: str = "gpt-4o"  # Model string like "openai/gpt-4o"
    provider: str = "openai"  # Provider name
    api_key: Optional[str] = None  # API key
    base_url: Optional[str] = None  # Custom API base URL
    language: str = "English"

    # Callbacks
    on_plan_complete: Optional[Callable] = None
    on_step_complete: Optional[Callable] = None
    on_step_error: Optional[Callable] = None
    on_pipeline_complete: Optional[Callable] = None


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    success: bool
    task: str
    task_type: TaskType
    plan: List[PlanStep]
    executions: List[ExecutionResult]
    thinking: str
    final_output: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task": self.task,
            "task_type": self.task_type.value,
            "plan": [s.to_dict() for s in self.plan],
            "executions": [e.to_dict() for e in self.executions],
            "thinking": self.thinking,
            "final_output": self.final_output,
            "error": self.error,
            "duration": round(self.duration, 3),
            "metadata": self.metadata,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Task: {self.task[:80]}...",
            f"Type: {self.task_type.value}",
            f"Status: {'Success' if self.success else 'Failed'}",
            f"Steps: {len(self.plan)}, Completed: {sum(1 for e in self.executions if e.success)}",
            f"Duration: {self.duration:.2f}s",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class PlannerPipeline:
    """
    End-to-end planning and execution pipeline.

    This pipeline orchestrates:
    1. Task analysis and planning (Planner)
    2. Code context retrieval (Retriever integration)
    3. Plan execution (Executor)
    4. Error handling and recovery

    Supports multiple LLM providers for model switching:
    - OpenAI (GPT-4o, GPT-4, etc.)
    - Anthropic (Claude 3, Claude 3.5)
    - Google (Gemini)
    - DeepSeek
    - MiniMax
    - LiteLLM (unified wrapper)

    Usage:
        # With default model
        pipeline = PlannerPipeline(root, retriever, tools)
        result = pipeline.run("Add authentication to the login endpoint")

        # With specific model
        pipeline = PlannerPipeline(root, config=PipelineConfig(
            model="anthropic/claude-3-5-sonnet",
            api_key="sk-..."
        ))
        result = pipeline.run("Analyze code quality")
    """

    name = "planner_pipeline"
    description = "End-to-end planning and execution"

    def __init__(
        self,
        root: str,
        retriever=None,
        tools: Optional[Dict[str, Callable]] = None,
        config: Optional[PipelineConfig] = None,
        llm=None,
    ):
        """
        Initialize the pipeline.

        Args:
            root: Repository root path
            retriever: Retriever instance for code context
            tools: Dict of tool_name -> callable
            config: Pipeline configuration
            llm: Optional pre-configured LLM instance
        """
        self.root = root
        self.config = config or PipelineConfig()

        # Initialize LLM
        self._llm = llm
        if self._llm is None and self.config.model:
            self._init_llm()

        # Initialize components
        self.retriever = retriever
        self.planner = Planner(
            root=root,
            retriever=retriever,
            model=self.config.model,
            llm=self._llm,
            language=self.config.language,
        )
        self.executor = Executor(root=root, tools=tools or {})

        # State
        self.current_plan: Optional[PlanState] = None
        self.execution_history: List[ExecutionResult] = []
        self.revision_count = 0

    def _init_llm(self) -> None:
        """Initialize LLM from config."""
        try:
            from llm import LLMFactory
            self._llm = LLMFactory.create(
                model=self.config.model,
                provider=self.config.provider,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")
            self._llm = None

    def set_model(self, model: str, provider: str = None, **kwargs) -> None:
        """
        Switch to a different model.

        Args:
            model: Model string like "openai/gpt-4o" or "claude-3-5-sonnet"
            provider: Optional explicit provider
            **kwargs: Additional LLM config (api_key, base_url, etc.)
        """
        self.config.model = model
        if provider:
            self.config.provider = provider
        if "api_key" in kwargs:
            self.config.api_key = kwargs["api_key"]
        if "base_url" in kwargs:
            self.config.base_url = kwargs["base_url"]

        self._init_llm()
        self.planner._llm = self._llm
        self.planner.model_name = model

    def switch_provider(self, provider: str, model: str = None, **kwargs) -> None:
        """
        Switch to a different provider.

        Args:
            provider: Provider name (openai, anthropic, google, deepseek, minimax)
            model: Model name (auto-selected if not provided)
            **kwargs: Additional config
        """
        if model is None:
            # Auto-select model for provider
            model_defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "google": "gemini-1.5-flash",
                "deepseek": "deepseek-chat",
                "minimax": "MiniMax-Text-01",
            }
            model = model_defaults.get(provider, "gpt-4o")

        self.set_model(model, provider=provider, **kwargs)

    def set_tools(self, tools: Dict[str, Callable]) -> None:
        """Set or update tools."""
        self.executor.set_tools(tools)

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a single tool."""
        self.executor.set_tools({name: func})

    def run(self, task: str, context: Optional[List[Dict[str, Any]]] = None) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            task: User's task description
            context: Optional pre-retrieved context

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()
        self.execution_history.clear()
        self.revision_count = 0

        # Phase 1: Planning
        plan_result = self.planner.plan(task, context)
        if not plan_result.success:
            return PipelineResult(
                success=False,
                task=task,
                task_type=self.current_plan.task_type if self.current_plan else TaskType.UNKNOWN,
                plan=[],
                executions=[],
                thinking=plan_result.thinking,
                error=plan_result.error,
                duration=time.time() - start_time,
            )

        self.current_plan = plan_result.state

        # Callback: plan complete
        if self.config.on_plan_complete:
            self.config.on_plan_complete(plan_result)

        # Phase 2: Execution
        if self.config.auto_execute:
            self.executor.load_plan(self.current_plan)

            while True:
                step = self.executor.get_next_step()
                if not step:
                    break

                # Execute step
                result = self.executor.execute_step(step)
                self.execution_history.append(result)

                # Callbacks
                if result.success and self.config.on_step_complete:
                    self.config.on_step_complete(step, result)
                elif not result.success and self.config.on_step_error:
                    self.config.on_step_error(step, result)

                    # Handle error/retry
                    if self.config.stop_on_error:
                        break

                    # Retry logic
                    if self.config.retry_on_failure and result.error:
                        retry_result = self.executor.retry_step(step.step_id)
                        if retry_result.success:
                            self.execution_history.append(retry_result)
                            continue

                # Check revision limit
                if self.revision_count >= self.config.max_revisions:
                    break

        # Calculate final status
        all_completed = all(s.status == "completed" for s in self.current_plan.plan)
        success = all_completed and not self.current_plan.error_message

        # Final callback
        if success and self.config.on_pipeline_complete:
            self.config.on_pipeline_complete(self)

        return PipelineResult(
            success=success,
            task=task,
            task_type=self.current_plan.task_type,
            plan=self.current_plan.plan,
            executions=self.execution_history,
            thinking=self.current_plan.thinking,
            final_output=self._generate_final_output(),
            error=self.current_plan.error_message,
            duration=time.time() - start_time,
            metadata={
                "revisions": self.revision_count,
                "context_files": len(self.current_plan.context),
            },
        )

    def run_with_llm(
        self,
        task: str,
        llm_func: Callable,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> PipelineResult:
        """
        Run pipeline with LLM for plan generation.

        This uses an LLM function to generate more sophisticated plans.

        Args:
            task: User's task description
            llm_func: Function that takes a prompt and returns text
            context: Optional pre-retrieved context

        Returns:
            PipelineResult
        """
        start_time = time.time()
        self.execution_history.clear()

        # Build context for LLM
        context_text = ""
        if context:
            context_text = "Relevant code context:\n"
            for c in context[:5]:
                context_text += f"\n{c['file_path']} (lines {c['lines']}):\n"
                context_text += c['content'][:500] + "\n..."

        # Generate plan with LLM
        prompt = default_prompts.get_think_prompt(task, context_text, self.config.language)
        thinking = llm_func(prompt)

        # Parse plan from LLM output
        # In production, this would parse structured output
        # For now, use the basic planner
        plan_result = self.planner.plan(task, context)

        if plan_result.success:
            self.current_plan = plan_result.state
            self.current_plan.thinking = thinking

        return self._execute_plan(start_time, task)

    def _execute_plan(self, start_time: float, task: str) -> PipelineResult:
        """Execute the current plan."""
        if not self.current_plan:
            return PipelineResult(
                success=False,
                task=task,
                task_type=TaskType.UNKNOWN,
                plan=[],
                executions=[],
                thinking="",
                error="No plan to execute",
                duration=time.time() - start_time,
            )

        # Load plan into executor
        self.executor.load_plan(self.current_plan)

        # Execute all steps
        while True:
            step = self.executor.get_next_step()
            if not step:
                break

            result = self.executor.execute_step(step)
            self.execution_history.append(result)

            if not result.success and self.config.stop_on_error:
                break

        # Build result
        all_completed = all(s.status == "completed" for s in self.current_plan.plan)
        success = all_completed and not self.current_plan.error_message

        return PipelineResult(
            success=success,
            task=task,
            task_type=self.current_plan.task_type,
            plan=self.current_plan.plan,
            executions=self.execution_history,
            thinking=self.current_plan.thinking,
            final_output=self._generate_final_output(),
            error=self.current_plan.error_message,
            duration=time.time() - start_time,
        )

    def _generate_final_output(self) -> str:
        """Generate the final output summary."""
        if not self.current_plan:
            return ""

        lines = ["## Execution Summary\n"]

        for step in self.current_plan.plan:
            status_icon = {
                "completed": "✓",
                "failed": "✗",
                "skipped": "-",
                "pending": "○",
            }.get(step.status, "?")

            lines.append(f"{status_icon} Step {step.step_id}: {step.description}")

        lines.append(f"\nTotal: {len(self.current_plan.plan)} steps")

        completed = sum(1 for s in self.current_plan.plan if s.status == "completed")
        failed = sum(1 for s in self.current_plan.plan if s.status == "failed")
        lines.append(f"Completed: {completed}, Failed: {failed}")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        if not self.current_plan:
            return {"status": "idle", "plan": None}

        return {
            "status": self.current_plan.status.value,
            "task_type": self.current_plan.task_type.value,
            "total_steps": len(self.current_plan.plan),
            "completed": sum(1 for s in self.current_plan.plan if s.status == "completed"),
            "failed": sum(1 for s in self.current_plan.plan if s.status == "failed"),
            "pending": sum(1 for s in self.current_plan.plan if s.status == "pending"),
            "context_files": len(self.current_plan.context),
        }

    def reset(self) -> None:
        """Reset the pipeline for a new task."""
        self.current_plan = None
        self.execution_history.clear()
        self.revision_count = 0


def quick_pipeline(task: str, root: str = ".") -> PipelineResult:
    """
    Quick pipeline run without external dependencies.

    Args:
        task: Task description
        root: Repository root

    Returns:
        PipelineResult
    """
    pipeline = PlannerPipeline(root)
    return pipeline.run(task)