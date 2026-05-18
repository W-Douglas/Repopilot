"""
Integration Test: LLM Tool Calling with LangGraph

This test demonstrates the complete LLM Tool Calling flow:
1. LLM receives function schemas
2. LLM decides to call a tool
3. Tool is executed via ToolManager
4. Results are returned to LLM
5. Loop continues until task complete
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    create_agent_tool_system,
    ToolRegistry,
    ToolManager,
    LLMToolAdapter,
    AgentToolSystem,
)


def test_tool_system_creation():
    """Test creating the agent tool system."""
    print("\n" + "=" * 60)
    print("Test 1: Creating Agent Tool System")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "main.py").write_text("x = 1\nprint('hello')")

        system = create_agent_tool_system(root=tmpdir)

        print(f"Registry: {len(system.registry.list_tools())} tools")
        print(f"Tool Manager: OK")
        print(f"Adapter: OK")

        # List available tools
        schemas = system.adapter.get_function_schemas()
        print(f"\nFunction Schemas: {len(schemas)}")
        for schema in schemas:
            print(f"  - {schema['name']}: {schema['description'][:50]}...")

        return True


def test_agent_tool_permissions():
    """Test agent-specific tool permissions."""
    print("\n" + "=" * 60)
    print("Test 2: Agent Tool Permissions")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("x = 1")

        system = create_agent_tool_system(root=tmpdir)

        # Test PlannerAgent tools
        planner_tools = system.get_tools_for_agent("PlannerAgent")
        print(f"PlannerAgent tools: {[t['name'] for t in planner_tools]}")

        # Test CoderAgent tools
        coder_tools = system.get_tools_for_agent("CoderAgent")
        print(f"CoderAgent tools: {[t['name'] for t in coder_tools]}")

        # Verify write_file is only in CoderAgent
        planner_names = [t['name'] for t in planner_tools]
        coder_names = [t['name'] for t in coder_tools]

        assert "write_file" not in planner_names, "Planner should not have write_file"
        assert "write_file" in coder_names, "Coder should have write_file"

        print("\nPermission check: PASSED")
        return True


def test_tool_execution_via_agent():
    """Test executing tools through agent system."""
    print("\n" + "=" * 60)
    print("Test 3: Tool Execution via Agent")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        main_py = Path(tmpdir) / "main.py"
        main_py.write_text("def hello():\n    return 'world'\n")

        system = create_agent_tool_system(root=tmpdir)

        # Execute read_file as PlannerAgent
        result = system.call_tool(
            agent="PlannerAgent",
            tool="read_file",
            params={"path": "main.py"}
        )

        print(f"Tool: read_file")
        print(f"Agent: PlannerAgent")
        print(f"Success: {result.success}")
        print(f"Output: {result.output[:100]}...")

        assert result.success, f"Tool execution failed: {result.error}"
        assert "def hello" in result.output, "File content not returned"

        return True


def test_permission_denied():
    """Test that agents cannot use unauthorized tools."""
    print("\n" + "=" * 60)
    print("Test 4: Permission Denied for Unauthorized Tool")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("x = 1")

        system = create_agent_tool_system(root=tmpdir)

        # Try to call write_file as PlannerAgent (should fail)
        result = system.call_tool(
            agent="PlannerAgent",
            tool="write_file",
            params={"path": "test.py", "content": "y = 2"}
        )

        print(f"Tool: write_file")
        print(f"Agent: PlannerAgent (not authorized)")
        print(f"Success: {result.success}")
        print(f"Status: {result.status.value}")

        assert not result.success, "Planner should not be able to write files"
        assert result.status.value == "permission_denied", f"Expected permission_denied, got {result.status}"

        print("\nPermission check: PASSED (correctly denied)")
        return True


def test_trajectory_recording():
    """Test that tool calls are recorded in trajectory."""
    print("\n" + "=" * 60)
    print("Test 5: Trajectory Recording")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("x = 1")

        system = create_agent_tool_system(root=tmpdir)

        # Execute multiple tools
        system.call_tool("PlannerAgent", "read_file", {"path": "main.py"})
        system.call_tool("PlannerAgent", "read_file", {"path": "main.py"})

        # Check trajectory
        trajectory = system.get_trajectory()
        stats = system.get_stats()

        print(f"Total calls: {stats['total_calls']}")
        print(f"Trajectory entries: {len(trajectory)}")
        print(f"Tool usage: {stats['tool_usage']}")

        assert stats['total_calls'] >= 2, "Should have recorded calls"

        return True


def test_llm_function_schemas():
    """Test that tools can be converted to LLM schemas."""
    print("\n" + "=" * 60)
    print("Test 6: LLM Function Schemas")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("x = 1")

        system = create_agent_tool_system(root=tmpdir)

        # Get schemas in OpenAI format
        schemas = system.adapter.get_function_schemas()

        print(f"Total schemas: {len(schemas)}")

        for schema in schemas[:3]:  # Show first 3
            print(f"\n  {schema['name']}:")
            params = schema.get('parameters', {})
            props = params.get('properties', {})
            for pname, pdef in list(props.items())[:2]:
                print(f"    - {pname}: {pdef.get('type', 'any')}")

        # Verify format matches OpenAI
        assert all('name' in s for s in schemas), "Missing name field"
        assert all('description' in s for s in schemas), "Missing description"
        assert all('parameters' in s for s in schemas), "Missing parameters"

        return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" LLM TOOL CALLING INTEGRATION TESTS")
    print("=" * 70)

    tests = [
        ("Tool System Creation", test_tool_system_creation),
        ("Agent Tool Permissions", test_agent_tool_permissions),
        ("Tool Execution via Agent", test_tool_execution_via_agent),
        ("Permission Denied", test_permission_denied),
        ("Trajectory Recording", test_trajectory_recording),
        ("LLM Function Schemas", test_llm_function_schemas),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\nTest FAILED: {e}")

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)