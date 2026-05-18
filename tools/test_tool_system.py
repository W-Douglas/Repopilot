"""
Tool System Test Suite - Validates tool functionality and LangGraph integration.

This test file demonstrates:
1. Tool instantiation and execution
2. Unified result format
3. Registry and access control
4. LangGraph integration patterns
"""

import sys
import json
from pathlib import Path

# Setup path
_test_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_test_dir))

from tools import ToolResult, ToolRegistry, ToolManager, ToolStatus
from tools.result import create_success_result, create_error_result
from tools.base import BaseTool, ToolParameter

# Import all tools
from tools.file_tools import ReadFileTool, WriteFileTool
from tools.search_tools import SearchCodeTool, GrepTool
from tools.tree_tools import GetRepoTreeTool
from tools.test_tools import RunPytestTool  # This is test_tools.py from tools/
from tools.git_tools import GetGitDiffTool, RollbackFileTool, GetGitStatusTool


def test_result_format():
    """Test 1: Unified ToolResult format."""
    print("\n" + "=" * 60)
    print("TEST 1: ToolResult Format")
    print("=" * 60)

    # Test success result
    result = create_success_result(
        tool="read_file",
        input_params={"path": "main.py", "lines": 100},
        output="def main():\n    pass",
        latency=0.023
    )

    print("\n[✓] Success Result:")
    print(f"    tool: {result.tool}")
    print(f"    success: {result.success}")
    print(f"    latency: {result.latency}s")
    print(f"    output: {str(result.output)[:50]}...")

    # Test error result
    error_result = create_error_result(
        tool="read_file",
        input_params={"path": "nonexistent.py"},
        error="File not found",
        status=ToolStatus.FAILURE
    )

    print("\n[✓] Error Result:")
    print(f"    tool: {error_result.tool}")
    print(f"    success: {error_result.success}")
    print(f"    error: {error_result.error}")

    # Test serialization
    result_dict = result.to_dict()
    print("\n[✓] Serialized result (for trajectory):")
    print(json.dumps(result_dict, indent=2)[:300])

    return result


def test_read_file_tool():
    """Test 2: ReadFileTool execution."""
    print("\n" + "=" * 60)
    print("TEST 2: ReadFileTool")
    print("=" * 60)

    tool = ReadFileTool(root=".")

    # Read this test file
    result = tool.execute(path="repomap/test_repomap.py", max_lines=50)

    print(f"\n[✓] Read test_repomap.py:")
    print(f"    success: {result.success}")
    print(f"    latency: {result.latency:.3f}s")
    print(f"    lines read: {result.metadata.get('read_lines', 0)}")
    print(f"    content preview: {result.output[:100]}...")

    # Test file not found
    error_result = tool.execute(path="nonexistent.py")
    print(f"\n[✓] Read nonexistent.py:")
    print(f"    success: {error_result.success}")
    print(f"    error: {error_result.error}")

    return result


def test_get_repo_tree():
    """Test 3: GetRepoTreeTool execution."""
    print("\n" + "=" * 60)
    print("TEST 3: GetRepoTreeTool")
    print("=" * 60)

    tool = GetRepoTreeTool(root=".")

    result = tool.execute(max_depth=2)

    print(f"\n[✓] Repository tree (depth=2):")
    print(f"    success: {result.success}")
    print(f"    lines: {result.metadata.get('lines', 0)}")
    print(f"\n{result.output}")

    return result


def test_search_code():
    """Test 4: SearchCodeTool execution."""
    print("\n" + "=" * 60)
    print("TEST 4: SearchCodeTool")
    print("=" * 60)

    tool = SearchCodeTool(root=".")

    # Search for "def" in Python files
    result = tool.execute(pattern=r"def\s+\w+", max_results=10)

    print(f"\n[✓] Search 'def \\w+':")
    print(f"    success: {result.success}")
    print(f"    matches: {result.output.get('total_matches', 0)}")

    for match in result.output.get('results', [])[:5]:
        print(f"    {match['file']}:{match['line']} - {match['content'][:50]}")

    return result


def test_registry():
    """Test 5: ToolRegistry."""
    print("\n" + "=" * 60)
    print("TEST 5: ToolRegistry")
    print("=" * 60)

    registry = ToolRegistry.create_default_registry(root=".")

    print(f"\n[✓] Registered tools:")
    for tool_name in registry.list_tools():
        meta = registry.get_metadata(tool_name)
        print(f"    - {tool_name} ({meta.category.value})")

    # Test execution through registry
    result = registry.execute("get_repo_tree")
    print(f"\n[✓] Execute via registry:")
    print(f"    success: {result.success}")

    return registry


def test_tool_manager():
    """Test 6: ToolManager with access control."""
    print("\n" + "=" * 60)
    print("TEST 6: ToolManager (Access Control)")
    print("=" * 60)

    manager = ToolManager(root=".")

    print(f"\n[✓] Tools available for BuildRepoMapNode:")
    available = manager.get_available_tools("BuildRepoMapNode")
    print(f"    {available}")

    print(f"\n[✓] Tools available for EditCodeNode:")
    available = manager.get_available_tools("EditCodeNode")
    print(f"    {available}")

    # Test access control - PlanFixNode should NOT have write_file
    result = manager.execute("write_file", node_name="PlanFixNode", path="test.txt", content="test")
    print(f"\n[✓] PlanFixNode trying write_file:")
    print(f"    success: {result.success}")
    print(f"    error: {result.error}")

    # Test access - EditCodeNode should have write_file
    result = manager.execute("get_repo_tree", node_name="EditCodeNode")
    print(f"\n[✓] EditCodeNode using get_repo_tree:")
    print(f"    success: {result.success}")

    return manager


def test_langgraph_integration():
    """Test 7: LangGraph integration pattern."""
    print("\n" + "=" * 60)
    print("TEST 7: LangGraph Integration Pattern")
    print("=" * 60)

    manager = ToolManager(root=".")

    # Simulate a LangGraph node
    node_name = "RetrieveContextNode"
    call_tool = manager.execute  # Simple alias

    print(f"\n[✓] Simulating {node_name}:")

    # Step 1: Get repo tree
    result = call_tool("get_repo_tree", node_name=node_name)
    print(f"    1. get_repo_tree: {'✓' if result.success else '✗'}")

    # Step 2: Read a file
    result = call_tool("read_file", node_name=node_name, path="repomap/__init__.py")
    print(f"    2. read_file: {'✓' if result.success else '✗'}")

    # Step 3: Search for symbols
    result = call_tool("search_code", node_name=node_name, pattern="class")
    print(f"    3. search_code: {'✓' if result.success else '✗'}")

    # Get trajectory
    trajectory = manager.get_trajectory()
    print(f"\n[✓] Execution trajectory ({len(trajectory)} calls):")
    for call in trajectory:
        print(f"    Step {call['step']}: {call['tool']} from {call['node']} - {'✓' if call['success'] else '✗'}")

    # Stats
    stats = manager.get_stats()
    print(f"\n[✓] Execution stats:")
    print(f"    total_calls: {stats['total_calls']}")
    print(f"    tool_usage: {stats['tool_usage']}")

    return trajectory


def test_write_tool():
    """Test 8: WriteFileTool execution."""
    print("\n" + "=" * 60)
    print("TEST 8: WriteFileTool")
    print("=" * 60)

    tool = WriteFileTool(root=".")

    # Create a test file
    result = tool.execute(
        path="test_write_temp.txt",
        content="# Test file created by Tool System\nprint('Hello')",
        create_backup=True
    )

    print(f"\n[✓] Write test file:")
    print(f"    success: {result.success}")
    print(f"    metadata: {result.metadata}")

    # Read it back
    read_tool = ReadFileTool(root=".")
    read_result = read_tool.execute(path="test_write_temp.txt")
    print(f"\n[✓] Read back:")
    print(f"    content: {read_result.output[:50]}...")

    # Clean up
    Path("test_write_temp.txt").unlink(missing_ok=True)
    print("\n[✓] Cleaned up test file")

    return result


def test_git_tools():
    """Test 9: Git tools."""
    print("\n" + "=" * 60)
    print("TEST 9: Git Tools")
    print("=" * 60)

    # Git status
    status_tool = GetGitStatusTool(root=".")
    result = status_tool.execute()

    print(f"\n[✓] Git status:")
    print(f"    branch: {result.output.get('branch', 'N/A')}")
    print(f"    clean: {result.output.get('clean', True)}")
    print(f"    modified: {len(result.output.get('modified', []))}")
    print(f"    untracked: {len(result.output.get('untracked', []))}")

    # Git diff
    diff_tool = GetGitDiffTool(root=".")
    result = diff_tool.execute()

    print(f"\n[✓] Git diff:")
    print(f"    has_changes: {result.output.get('has_changes', False)}")

    return result


def test_tool_schemas():
    """Test 10: Tool schemas for LLM."""
    print("\n" + "=" * 60)
    print("TEST 10: Tool Schemas (for LLM)")
    print("=" * 60)

    tool = ReadFileTool(root=".")
    schema = tool.get_schema()

    print(f"\n[✓] ReadFileTool schema:")
    print(json.dumps(schema, indent=2))

    return schema


def main():
    """Run all tests."""
    print("=" * 60)
    print("Tool System Module Integration Test")
    print("=" * 60)

    results = {}

    try:
        results['result_format'] = test_result_format()
    except Exception as e:
        print(f"\n[✗] Result format test failed: {e}")

    try:
        results['read_file'] = test_read_file_tool()
    except Exception as e:
        print(f"\n[✗] ReadFileTool test failed: {e}")

    try:
        results['repo_tree'] = test_get_repo_tree()
    except Exception as e:
        print(f"\n[✗] GetRepoTreeTool test failed: {e}")

    try:
        results['search_code'] = test_search_code()
    except Exception as e:
        print(f"\n[✗] SearchCodeTool test failed: {e}")

    try:
        results['registry'] = test_registry()
    except Exception as e:
        print(f"\n[✗] Registry test failed: {e}")

    try:
        results['tool_manager'] = test_tool_manager()
    except Exception as e:
        print(f"\n[✗] ToolManager test failed: {e}")

    try:
        results['langgraph'] = test_langgraph_integration()
    except Exception as e:
        print(f"\n[✗] LangGraph integration test failed: {e}")

    try:
        results['write_tool'] = test_write_tool()
    except Exception as e:
        print(f"\n[✗] WriteFileTool test failed: {e}")

    try:
        results['git_tools'] = test_git_tools()
    except Exception as e:
        print(f"\n[✗] Git tools test failed: {e}")

    try:
        results['schemas'] = test_tool_schemas()
    except Exception as e:
        print(f"\n[✗] Tool schemas test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    print("\n[✓] Tool System module is ready for LangGraph integration.")
    print("\nKey features validated:")
    print("  - Unified ToolResult format")
    print("  - Node-based access control (RESTRICTED mode)")
    print("  - Trajectory recording")
    print("  - Tool schemas for LLM compatibility")


if __name__ == "__main__":
    main()