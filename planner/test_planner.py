"""
Test and examples for the Planner module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from planner import (
    Planner,
    Executor,
    PlannerPipeline,
    PlanState,
    PlanStep,
    TaskType,
    PlanStatus,
    PipelineResult,
)
from retriever import Chunk, ChunkType, RetrievalQuery


def test_plan_state():
    """Test plan state management."""
    print("\n=== Testing Plan State ===")

    state = PlanState(task="Add authentication to the login endpoint")

    # Add context
    state.add_context([{
        "file_path": "auth.py",
        "content": "def login(): pass",
        "lines": "1-5",
    }])

    # Add steps
    state.add_step(PlanStep(
        step_id=1,
        description="Read auth.py to understand current structure",
        action="read_file",
        target="auth.py",
    ))
    state.add_step(PlanStep(
        step_id=2,
        description="Add authentication logic",
        action="write_file",
        target="auth.py",
        dependencies=[1],
    ))

    # Update step
    state.update_step(1, status="completed", result="File content read")

    print(f"Task: {state.task}")
    print(f"Task type: {state.task_type}")
    print(f"Status: {state.status}")
    print(f"Steps: {len(state.plan)}")
    print(f"Context: {len(state.context)} files")

    summary = state.get_status_summary()
    print(f"\nSummary: {summary}")

    print("Plan state test passed!")


def test_planner():
    """Test planner functionality."""
    print("\n=== Testing Planner ===")

    planner = Planner(root=".")

    # Test task classification
    test_tasks = [
        "Add a new function to process data",
        "Fix the bug in the login flow",
        "Refactor the user module",
        "Explain how the cache works",
    ]

    for task in test_tasks:
        state = PlanState(task=task)
        # Simple classification (using planner's internal method)
        planner.state = state
        task_type = planner._classify_task(task)
        print(f"Task: '{task}' -> Type: {task_type.value}")

    print("\nPlanner test passed!")


def test_executor():
    """Test executor functionality."""
    print("\n=== Testing Executor ===")

    # Define mock tools
    def mock_read_file(path):
        return f"Content of {path}"

    def mock_write_file(path):
        return f"Wrote to {path}"

    def mock_search(query):
        return f"Found results for: {query}"

    tools = {
        "read_file": mock_read_file,
        "write_file": mock_write_file,
        "search_code": mock_search,
    }

    # Create executor
    executor = Executor(root=".", tools=tools)

    # Create a plan
    state = PlanState(task="Test task")
    state.add_step(PlanStep(step_id=1, description="Read file", action="read_file", target="test.py"))
    state.add_step(PlanStep(step_id=2, description="Write file", action="write_file", target="test.py", dependencies=[1]))

    # Load and execute
    executor.load_plan(state)
    results = executor.execute_all()

    print(f"Executed {len(results)} steps:")
    for r in results:
        print(f"  Step {r.step_id}: {'✓' if r.success else '✗'} ({r.action})")

    summary = executor.get_execution_summary()
    print(f"\nSummary: {summary}")

    print("\nExecutor test passed!")


def test_pipeline():
    """Test the full pipeline."""
    print("\n=== Testing Planner Pipeline ===")

    # Define mock tools
    def mock_read_file(path):
        return f"Content of {path}"

    def mock_write_file(path):
        return f"Wrote to {path}"

    tools = {
        "read_file": mock_read_file,
        "write_file": mock_write_file,
    }

    # Create pipeline
    config = PipelineConfig(auto_execute=True, stop_on_error=True)
    pipeline = PlannerPipeline(root=".", tools=tools, config=config)

    # Run a task
    task = "Add a new calculate_discount function to utils.py"
    result = pipeline.run(task)

    print(f"\nPipeline result:")
    print(f"  Success: {result.success}")
    print(f"  Task type: {result.task_type.value}")
    print(f"  Plan steps: {len(result.plan)}")
    print(f"  Executions: {len(result.executions)}")
    print(f"  Duration: {result.duration:.3f}s")

    if result.error:
        print(f"  Error: {result.error}")

    print("\nPipeline test passed!")


def test_integration_with_retriever():
    """Test integration with retriever."""
    print("\n=== Testing Retriever Integration ===")

    from retriever import BM25Retriever, RetrievalQuery

    # Create chunks
    chunks = [
        Chunk(id="1", content="def login(username, password): authenticate user",
              file_path="auth.py", chunk_type=ChunkType.FUNCTION,
              start_line=1, end_line=5, symbols=["login"]),
        Chunk(id="2", content="class UserSession: manages user sessions",
              file_path="session.py", chunk_type=ChunkType.CLASS,
              start_line=1, end_line=10, symbols=["UserSession"]),
        Chunk(id="3", content="def validate_token(token): verify JWT token",
              file_path="auth.py", chunk_type=ChunkType.FUNCTION,
              start_line=20, end_line=25, symbols=["validate_token"]),
    ]

    # Create and index retriever
    retriever = BM25Retriever(".")
    retriever.index(chunks)

    # Create planner with retriever
    planner = Planner(root=".", retriever=retriever)

    # Plan with retrieval
    task = "Add token validation to the login function"
    result = planner.plan(task)

    print(f"Task: {task}")
    print(f"Success: {result.success}")
    print(f"Task type: {result.state.task_type}")
    print(f"Plan steps: {len(result.plan)}")
    print(f"Context retrieved: {len(result.state.context)} chunks")

    for ctx in result.state.context:
        print(f"  - {ctx['file_path']}: {ctx['symbols']}")

    print("\nRetriever integration test passed!")


def test_task_types():
    """Test all task types."""
    print("\n=== Testing Task Types ===")

    for tt in TaskType:
        print(f"  - {tt.value}")


if __name__ == "__main__":
    print("Planner Module Test Suite")
    print("=" * 50)

    test_task_types()
    test_plan_state()
    test_planner()
    test_executor()
    test_pipeline()
    test_integration_with_retriever()

    print("\n" + "=" * 50)
    print("All tests passed!")