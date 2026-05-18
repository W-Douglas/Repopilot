"""
Planner Module CLI Entry Point

Usage:
    python -m planner.main --task "Add authentication to login"
    python -m planner.main --task "Fix the bug" --execute
    python -m planner.main --interactive
"""

import argparse
import json
import sys
from pathlib import Path

from planner import (
    PlannerPipeline,
    Planner,
    Executor,
    PlanState,
    PlanStep,
    PipelineConfig,
)


def create_tools():
    """Create mock tools for CLI."""
    tools = {}

    # Read file tool
    def read_file(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

    # Write file tool
    def write_file(path, content=None):
        # In CLI, content would be provided separately
        return f"Would write to {path}"

    # Search tool
    def search_code(query):
        return f"Would search for: {query}"

    tools["read_file"] = read_file
    tools["write_file"] = write_file
    tools["search_code"] = search_code

    return tools


def cmd_plan(args):
    """Run planning only."""
    planner = Planner(root=args.root)

    if args.context:
        context = json.load(open(args.context))
    else:
        context = None

    result = planner.plan(args.task, context)

    print(f"\n{'='*60}")
    print("PLAN RESULT")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Task type: {result.state.task_type.value}")
    print(f"Success: {result.success}")

    if result.thinking:
        print(f"\n## Thinking\n{result.thinking}")

    print(f"\n## Plan ({len(result.plan)} steps)")
    for step in result.plan:
        deps = f" [depends on {step.dependencies}]" if step.dependencies else ""
        print(f"  {step.step_id}. {step.description}")
        print(f"     Action: {step.action} -> {step.target}{deps}")

    return 0 if result.success else 1


def cmd_execute(args):
    """Run planning and execution."""
    tools = create_tools()

    config = PipelineConfig(
        auto_execute=True,
        stop_on_error=args.stop_on_error,
    )

    pipeline = PlannerPipeline(
        root=args.root,
        tools=tools,
        config=config,
    )

    result = pipeline.run(args.task)

    print(f"\n{'='*60}")
    print("EXECUTION RESULT")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")

    print(f"\n## Plan")
    for step in result.plan:
        status = "✓" if step.status == "completed" else "✗" if step.status == "failed" else "○"
        print(f"  {status} Step {step.step_id}: {step.description}")

    print(f"\n## Executions")
    for exec in result.executions:
        status = "✓" if exec.success else "✗"
        print(f"  {status} {exec.action} on {exec.target} ({exec.duration:.3f}s)")

    if result.error:
        print(f"\n## Error\n{result.error}")

    if args.verbose:
        print(f"\n## Full Output\n{result.final_output}")

    return 0 if result.success else 1


def cmd_status(args):
    """Check status of current plan."""
    # For now, just show available tools
    print("\nPlanner Module")
    print("=" * 40)
    print("Available commands:")
    print("  plan     - Generate a plan for a task")
    print("  execute  - Plan and execute a task")
    print("  status   - Show current status")
    print("\nAvailable tools:")
    tools = create_tools()
    for name in tools:
        print(f"  - {name}")


def cmd_interactive(args):
    """Interactive mode."""
    print("\nPlanner Interactive Mode")
    print("=" * 40)
    print("Type your task and press Enter.")
    print("Type 'exit' or 'quit' to exit.")
    print()

    tools = create_tools()
    pipeline = PlannerPipeline(root=args.root, tools=tools)

    while True:
        try:
            task = input("Task> ").strip()

            if not task:
                continue

            if task.lower() in ['exit', 'quit', 'q']:
                break

            result = pipeline.run(task)

            print(f"\nResult: {'Success' if result.success else 'Failed'}")
            print(f"Steps: {len(result.plan)}, Duration: {result.duration:.2f}s")

            if result.error:
                print(f"Error: {result.error}")

            print()

        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Planner Module CLI - Task planning and execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # plan command
    plan_parser = subparsers.add_parser("plan", help="Generate a plan")
    plan_parser.add_argument("--task", "-t", required=True, help="Task description")
    plan_parser.add_argument("--context", "-c", help="JSON file with context")

    # execute command
    exec_parser = subparsers.add_parser("execute", help="Plan and execute")
    exec_parser.add_argument("--task", "-t", required=True, help="Task description")
    exec_parser.add_argument("--no-stop", dest="stop_on_error", action="store_false", default=True,
                            help="Don't stop on errors")

    # status command
    subparsers.add_parser("status", help="Show status")

    # interactive
    subparsers.add_parser("interactive", help="Interactive mode")
    subparsers.add_parser("i", help="Interactive mode (shortcut)")

    args = parser.parse_args()

    if not args.command:
        # Default: plan command
        if args.task:
            return cmd_plan(args)
        else:
            parser.print_help()
            return 1

    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "execute":
        return cmd_execute(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command in ["interactive", "i"]:
        return cmd_interactive(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())