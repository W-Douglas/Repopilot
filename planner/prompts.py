"""
Planner Prompts - System prompts for the planner model
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlannerPrompts:
    """System prompts for the planner agent."""

    # Main system prompt for planning
    main_system = """You are an expert software architect and planner. Your role is to:
1. Analyze the user's task and requirements
2. Understand the codebase context provided
3. Create a clear, step-by-step plan to accomplish the task
4. Consider dependencies and ordering of steps

Output format:
- Use numbered steps (1, 2, 3, ...)
- For each step, specify: action, target file/query, and description
- Identify dependencies between steps
- Estimate confidence for each step (high/medium/low)

Available actions:
- read_file: Read file contents
- write_file: Write or modify file
- search_code: Search for code patterns
- grep: Search text in files
- run_command: Execute shell command
- create_file: Create new file

Always consider:
- Start with reading existing relevant files
- Build dependencies properly
- Verify results before proceeding to next steps
- Error handling and fallback plans

Reply in the user's language ({language}).
"""

    # Task classification prompt
    classify_system = """Analyze the following task and classify its type.

Task: {task}

Options:
- modify: Modify existing code
- create: Create new files/components
- refactor: Refactor existing code
- debug: Debug/analyze issues
- review: Code review
- explain: Explain code
- test: Generate tests

Also identify key entities mentioned (file names, function names, classes, etc.)
and estimate complexity (simple/medium/complex).

Output as JSON: {{"type": "...", "entities": [...], "complexity": "..."}}
"""

    # Plan generation prompt
    plan_system = """Based on the task and context, generate a detailed execution plan.

Task: {task}
Context files: {context_files}

For each step, provide:
- step_id: Sequential number
- description: What this step does
- action: Which tool/action to use
- target: File path or search query
- dependencies: Which steps must complete first (by step_id)

Output format:
```
Step 1:
  action: read_file
  target: src/utils/helper.py
  description: Read the helper file to understand current implementation
  dependencies: []

Step 2:
  action: write_file
  target: src/utils/helper.py
  description: Add the new calculate_discount function
  dependencies: [1]
...
```

Be specific and concise. Do not show actual code in the plan.
"""

    # Thinking/reasoning prompt
    think_system = """Think through this task step by step. Consider:

1. What is the goal?
2. What files need to be modified?
3. What is the current state of those files?
4. What changes need to be made?
5. Are there any dependencies or side effects?
6. How can we verify the changes work?

Think silently, then output your final plan.

Task: {task}

Existing context:
{context}

Your thinking:
"""

    # Execution prompt for executor
    execute_system = """Execute the following plan step by step.

Plan:
{plan}

Current step: {current_step}
Step description: {step_description}

Use the appropriate tool to complete this step.
If the step fails, analyze the error and determine if you should:
- Retry with modifications
- Skip and continue to next step
- Revise the plan

Report the result of each step clearly.
"""

    # Error handling prompt
    error_system = """An error occurred during execution:

Step: {step}
Error: {error}

Current plan status:
{plan_status}

Analyze the error and determine:
1. Was the error due to incorrect plan or external factors?
2. Should we retry, skip, or revise?
3. What is the impact on remaining steps?

Output your decision and any revised approach.
"""

    # Final review prompt
    review_system = """Review the completed execution:

Task: {task}
Steps completed: {steps_completed}
Errors encountered: {errors}

Verify that:
1. All required changes were made
2. No unintended side effects occurred
3. The solution is complete and correct

If verification passes, summarize what was done.
If verification fails, identify what remains to be done.
"""

    # Output format for parsing
    output_format = """
IMPORTANT: Always output your response in this format:

## Thinking
[Your reasoning and analysis here]

## Plan
Step 1:
  action: [tool name]
  target: [file path or query]
  description: [what to do]
  dependencies: [list of step IDs, empty if none]

Step 2:
  ...

## Confidence
[high/medium/low] - {reason}
"""

    @classmethod
    def get_classify_prompt(cls, task: str, language: str = "English") -> str:
        return cls.classify_system.format(task=task).replace("{language}", language)

    @classmethod
    def get_plan_prompt(cls, task: str, context_files: List[str], language: str = "English") -> str:
        ctx = "\n".join(f"- {f}" for f in context_files) if context_files else "No context files."
        return cls.plan_system.format(task=task, context_files=ctx).replace("{language}", language)

    @classmethod
    def get_think_prompt(cls, task: str, context: str, language: str = "English") -> str:
        return cls.think_system.format(task=task, context=context).replace("{language}", language)

    @classmethod
    def get_execute_prompt(cls, plan: str, current_step: int, step_desc: str, language: str = "English") -> str:
        return cls.execute_system.format(
            plan=plan,
            current_step=current_step,
            step_description=step_desc
        ).replace("{language}", language)

    @classmethod
    def get_error_prompt(cls, step: str, error: str, plan_status: str, language: str = "English") -> str:
        return cls.error_system.format(
            step=step,
            error=error,
            plan_status=plan_status
        ).replace("{language}", language)

    @classmethod
    def get_review_prompt(cls, task: str, steps: List[str], errors: List[str], language: str = "English") -> str:
        return cls.review_system.format(
            task=task,
            steps_completed="\n".join(f"- {s}" for s in steps),
            errors="\n".join(f"- {e}" for e in errors) if errors else "None"
        ).replace("{language}", language)


# Default prompts instance
default_prompts = PlannerPrompts()