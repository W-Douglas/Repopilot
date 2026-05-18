"""
Planner - Analyzes tasks and generates execution plans

The Planner is responsible for:
1. Understanding the user's task
2. Retrieving relevant code context
3. Classifying the task type
4. Generating a structured execution plan
5. Reasoning through the approach

Supports seamless model switching via LLM module.
"""
import re
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

from .state import PlanState, PlanStep, TaskType, PlanStatus
from .prompts import PlannerPrompts, default_prompts


@dataclass
class PlanResult:
    """Result of the planning process."""
    success: bool
    state: PlanState
    thinking: str = ""
    plan: List[PlanStep] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "thinking": self.thinking,
            "plan": [s.to_dict() for s in self.plan],
            "error": self.error,
            "state": self.state.to_dict(),
        }


class Planner:
    """
    Plans and coordinates task execution.

    The planner follows a think-then-execute pattern:
    1. Think: Analyze the task, retrieve context, generate plan
    2. Execute: Follow the plan step by step

    The Planner does NOT execute tools - it delegates to the Executor.

    Supports multiple LLM providers via the LLM module:
    - OpenAI (GPT-4o, GPT-4, etc.)
    - Anthropic (Claude 3, Claude 3.5)
    - Google (Gemini)
    - DeepSeek
    - MiniMax
    - LiteLLM (unified wrapper)
    """

    name = "planner"
    description = "Task planning and coordination"

    def __init__(
        self,
        root: str,
        retriever=None,
        model: str = "gpt-4o",
        llm=None,
        language: str = "English",
        prompts: PlannerPrompts = None,
    ):
        """
        Initialize the planner.

        Args:
            root: Repository root path
            retriever: Optional retriever instance for code context
            model: LLM model string (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet")
                  Can also be just model name like "gpt-4o" (auto-detects provider)
            llm: Optional pre-configured LLM instance (takes precedence over model)
            language: Language for prompts/responses
            prompts: Custom prompts instance
        """
        self.root = root
        self.retriever = retriever
        self.model_name = model
        self.language = language
        self.prompts = prompts or default_prompts

        # Initialize LLM
        self._llm = None
        if llm is not None:
            self._llm = llm
        elif model:
            self._init_llm(model)

        # Current state
        self.state = None

    def _init_llm(self, model: str) -> None:
        """Initialize LLM from model string."""
        try:
            from llm import LLMFactory
            self._llm = LLMFactory.create(model)
        except ImportError:
            # LLM module not available, planner will work without LLM
            pass
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")

    def set_model(self, model: str) -> None:
        """
        Switch to a different model.

        Args:
            model: Model string like "openai/gpt-4o" or "claude-3-5-sonnet"
        """
        self.model_name = model
        self._init_llm(model)

    def switch_model(self, provider: str, model: str, **kwargs) -> None:
        """
        Switch to a different provider/model.

        Args:
            provider: Provider name (openai, anthropic, google, etc.)
            model: Model name
            **kwargs: Additional LLM config options
        """
        try:
            from llm import LLMFactory
            self._llm = LLMFactory.create(model, provider=provider, **kwargs)
            self.model_name = f"{provider}/{model}"
        except Exception as e:
            raise RuntimeError(f"Failed to switch model: {e}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of {"role": "...", "content": "..."}
            **kwargs: Additional parameters for LLM

        Returns:
            Response content or None
        """
        if not self._llm:
            return None

        try:
            from llm import ChatMessage
            chat_messages = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in messages
            ]
            response = self._llm.chat(chat_messages, **kwargs)
            return response.content
        except Exception as e:
            print(f"LLM chat error: {e}")
            return None

    def plan(self, task: str, context: Optional[List[Dict[str, Any]]] = None) -> PlanResult:
        """
        Main planning entry point.

        Args:
            task: The user's task description
            context: Optional pre-retrieved code context

        Returns:
            PlanResult with the generated plan and state
        """
        # Initialize state
        self.state = PlanState(task=task)

        # Add task to history
        self.state.add_history("user", task)

        try:
            # Classify the task
            task_type = self._classify_task(task)
            self.state.task_type = task_type

            # Retrieve relevant context if not provided
            if context is None and self.retriever:
                context = self._retrieve_context(task)
                self.state.add_context(context)

            # Generate the plan
            thinking, steps = self._generate_plan(task, context)
            self.state.thinking = thinking

            for step in steps:
                self.state.add_step(step)

            # Update status
            self.state.status = PlanStatus.PLANNED

            # Add planning response to history
            self.state.add_history("assistant", f"Plan generated:\n{self._plan_to_text(steps)}")

            return PlanResult(
                success=True,
                state=self.state,
                thinking=thinking,
                plan=steps,
            )

        except Exception as e:
            self.state.status = PlanStatus.FAILED
            self.state.error_message = str(e)
            return PlanResult(
                success=False,
                state=self.state,
                error=str(e),
            )

    def _classify_task(self, task: str) -> TaskType:
        """Classify the task type based on keywords."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["add", "implement", "create", "new"]):
            if any(kw in task_lower for kw in ["function", "file", "class", "module"]):
                return TaskType.CREATE

        if any(kw in task_lower for kw in ["fix", "bug", "error", "crash", "issue"]):
            return TaskType.DEBUG

        if any(kw in task_lower for kw in ["refactor", "rename", "move", "restructure"]):
            return TaskType.REFACTOR

        if any(kw in task_lower for kw in ["test", "spec", "verify"]):
            return TaskType.TEST

        if any(kw in task_lower for kw in ["review", "check", "analyze"]):
            return TaskType.REVIEW

        if any(kw in task_lower for kw in ["explain", "what", "how", "why"]):
            return TaskType.EXPLAIN

        if any(kw in task_lower for kw in ["modify", "change", "update", "edit"]):
            return TaskType.MODIFY

        return TaskType.UNKNOWN

    def _retrieve_context(self, task: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant code context from the retriever."""
        if not self.retriever:
            return []

        from retriever import RetrievalQuery

        query = RetrievalQuery.from_text(task, top_k=top_k)
        results = self.retriever.retrieve(query)

        # Convert to context format
        context = []
        for r in results:
            context.append({
                "file_path": r.chunk.file_path,
                "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
                "content": r.chunk.content,
                "symbols": r.chunk.symbols,
                "score": r.score,
                "type": r.chunk.chunk_type.value,
            })

        return context

    def _generate_plan(
        self,
        task: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[PlanStep]]:
        """
        Generate execution plan from task and context.

        Returns:
            Tuple of (thinking, steps)
        """
        # For now, generate a simple plan based on task type and context
        # In production, this would call an LLM

        thinking_parts = []
        steps = []

        # Analyze task and context
        if context:
            files = list(set(c["file_path"] for c in context))
            thinking_parts.append(f"Found {len(files)} relevant files:")
            for f in files[:5]:
                thinking_parts.append(f"  - {f}")

        # Generate steps based on task type
        step_id = 1

        if self.state.task_type == TaskType.CREATE:
            # Create new files/components
            thinking_parts.append("This is a create task. Steps:")

            # Usually need to read existing files first
            if context:
                first_file = context[0]["file_path"] if context else None
                if first_file:
                    steps.append(PlanStep(
                        step_id=step_id,
                        description=f"Read existing file for reference: {first_file}",
                        action="read_file",
                        target=first_file,
                        confidence=0.9,
                    ))
                    step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                description=f"Create new implementation based on task",
                action="write_file",
                target=self._infer_target_file(task),
                confidence=0.7,
            ))
            step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                description="Verify the implementation is correct",
                action="read_file",
                target=self._infer_target_file(task),
                confidence=0.8,
            ))

        elif self.state.task_type == TaskType.MODIFY:
            thinking_parts.append("This is a modify task. Steps:")

            # Identify target files from context
            target_files = self._extract_target_files(task, context)
            for tf in target_files:
                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"Read file to understand current implementation",
                    action="read_file",
                    target=tf,
                    confidence=0.9,
                ))
                step_id += 1

                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"Modify the file to implement the change",
                    action="write_file",
                    target=tf,
                    dependencies=[step_id - 1],
                    confidence=0.8,
                ))
                step_id += 1

        elif self.state.task_type == TaskType.DEBUG:
            thinking_parts.append("This is a debug task. Steps:")

            # Find relevant files
            target_files = self._extract_target_files(task, context)
            for tf in target_files:
                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"Read file to identify the issue",
                    action="read_file",
                    target=tf,
                    confidence=0.9,
                ))
                step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                description="Identify the root cause and fix the issue",
                action="write_file",
                target=target_files[0] if target_files else "",
                dependencies=[step_id - 1] if target_files else [],
                confidence=0.7,
            ))

        elif self.state.task_type == TaskType.REFACTOR:
            thinking_parts.append("This is a refactor task. Steps:")

            target_files = self._extract_target_files(task, context)
            for tf in target_files:
                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"Read and analyze current structure",
                    action="read_file",
                    target=tf,
                    confidence=0.9,
                ))
                step_id += 1

                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"Refactor the code",
                    action="write_file",
                    target=tf,
                    dependencies=[step_id - 1],
                    confidence=0.7,
                ))
                step_id += 1

        else:
            # Generic approach: read context files then modify
            thinking_parts.append("Using generic approach:")

            if context:
                files = list(set(c["file_path"] for c in context))[:3]
                for tf in files:
                    steps.append(PlanStep(
                        step_id=step_id,
                        description=f"Read file: {tf}",
                        action="read_file",
                        target=tf,
                        confidence=0.8,
                    ))
                    step_id += 1

                # Add a modify step
                target = context[0]["file_path"]
                steps.append(PlanStep(
                    step_id=step_id,
                    description="Make the required changes",
                    action="write_file",
                    target=target,
                    dependencies=[step_id - len(files)] if files else [],
                    confidence=0.7,
                ))
            else:
                # No context - infer target file from task
                target = self._infer_target_file(task)
                thinking_parts.append(f"No context found. Target file: {target}")

                steps.append(PlanStep(
                    step_id=step_id,
                    description="Read target file to understand current state",
                    action="read_file",
                    target=target,
                    confidence=0.6,
                ))
                step_id += 1

                steps.append(PlanStep(
                    step_id=step_id,
                    description="Make the required changes",
                    action="write_file",
                    target=target,
                    dependencies=[step_id - 1],
                    confidence=0.5,
                ))

        thinking = "\n".join(thinking_parts)
        return thinking, steps

    def _infer_target_file(self, task: str) -> str:
        """Infer the target file from the task description."""
        # Simple heuristics - in production, use LLM
        words = task.split()
        for word in words:
            if "/" in word and (word.endswith(".py") or ".py" in word):
                return word
            if word.endswith(".py"):
                return word
        return "src/main.py"  # Default

    def _extract_target_files(
        self,
        task: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Extract target file paths from task or context."""
        files = []

        # From task
        words = task.split()
        for word in words:
            if "/" in word and ".py" in word:
                files.append(word)
            elif word.endswith(".py"):
                files.append(word)

        # From context
        if context:
            for c in context:
                fp = c.get("file_path", "")
                if fp and fp not in files:
                    files.append(fp)

        return files[:5]  # Limit to 5 files

    def _plan_to_text(self, steps: List[PlanStep]) -> str:
        """Convert plan to readable text."""
        lines = ["## Plan"]
        for step in steps:
            deps = f", depends on {step.dependencies}" if step.dependencies else ""
            lines.append(f"Step {step.step_id}: {step.description}")
            lines.append(f"  Action: {step.action} -> {step.target}{deps}")
        return "\n".join(lines)

    def revise_plan(
        self,
        error: str,
        current_step: int,
    ) -> PlanResult:
        """
        Revise the current plan after an error.

        Args:
            error: The error message
            current_step: Which step failed

        Returns:
            Revised plan
        """
        if not self.state:
            raise ValueError("No active plan to revise")

        self.state.status = PlanStatus.THINKING

        # Mark the failed step
        self.state.update_step(current_step, status="failed", error=error)

        # Generate revised plan
        thinking = f"Analyzing error from step {current_step}: {error}\n"
        thinking += "Revising plan to work around the error.\n"

        # Add recovery steps
        recovery_step_id = len(self.state.plan) + 1
        self.state.add_step(PlanStep(
            step_id=recovery_step_id,
            description=f"Handle error from previous step",
            action="skip",  # Skip the failing step
            target="",
            dependencies=[current_step],
            confidence=0.5,
        ))

        self.state.thinking = thinking
        self.state.status = PlanStatus.REVISED

        return PlanResult(
            success=True,
            state=self.state,
            thinking=thinking,
            plan=self.state.plan,
        )

    def update_progress(self, step_id: int, status: str, result: str = None) -> None:
        """Update the status of a plan step."""
        if self.state:
            self.state.update_step(step_id, status=status, result=result)
            if status == "completed":
                self.state.current_step = step_id + 1

    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next pending step to execute."""
        if self.state:
            return self.state.get_next_pending_step()
        return None

    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        if not self.state:
            return True
        return all(s.status == "completed" for s in self.state.plan)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current plan."""
        if self.state:
            return self.state.get_status_summary()
        return {"error": "No active plan"}


def quick_plan(task: str, root: str = ".") -> PlanResult:
    """
    Quick planning without external dependencies.

    Args:
        task: Task description
        root: Repository root

    Returns:
        PlanResult
    """
    planner = Planner(root)
    return planner.plan(task)