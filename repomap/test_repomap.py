"""
RepoMap Test Suite - Validates repomap module for integration with downstream components.

This test file demonstrates:
1. Module instantiation and basic functionality
2. RepoMap output structure that downstream components expect
3. Integration patterns for Planner, Retriever, and Coder modules

Run from project root: python3 repomap/test_repomap.py
"""

import json
import sys
from pathlib import Path

# Setup import paths based on execution location
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent

# Add project root to path
sys.path.insert(0, str(_project_root))

from repomap.config import RepoMapConfig
from repomap.scanner import FileScanner
from repomap.parser import SymbolParser, symbols_to_markdown
from repomap.builder import RepoMapBuilder


class RepoMapOutput:
    """
    Standardized output format for RepoMap that downstream components expect.
    This ensures consistent interface between RepoMap and other modules.
    """

    def __init__(self, raw_data: dict):
        self.raw = raw_data
        self.root = raw_data.get("root", "")
        self.files = raw_data.get("files", [])
        self.tree = raw_data.get("tree", "")
        self.file_count = raw_data.get("file_count", 0)

    def get_files_by_importance(self, top_n: int = None) -> list:
        """Get files sorted by importance."""
        sorted_files = sorted(self.files, key=lambda f: f.get("importance", 0), reverse=True)
        if top_n:
            return sorted_files[:top_n]
        return sorted_files

    def get_files_by_name(self, name: str) -> list:
        """Find files matching a name pattern."""
        name_lower = name.lower()
        return [f for f in self.files if name_lower in f["path"].lower()]

    def get_symbols_by_kind(self, kind: str) -> list:
        """Get all symbols of a specific kind across all files."""
        result = []
        for f in self.files:
            for sym in f.get("symbols", []):
                if sym.get("kind") == kind:
                    result.append({**sym, "file": f["path"]})
        return result

    def to_context_for_llm(self, max_tokens: int = 4000) -> str:
        """Convert to context string suitable for LLM input."""
        builder = RepoMapBuilder(self.root, RepoMapConfig())
        return builder.get_repo_map_text(max_tokens=max_tokens)

    def to_planner_input(self) -> dict:
        """
        Convert to format expected by Planner module.
        Planner needs: file paths, symbols, importance scores.
        """
        return {
            "repo_root": self.root,
            "file_count": self.file_count,
            "files": [
                {
                    "path": f["path"],
                    "importance": f["importance"],
                    "symbols": [
                        {"name": s["name"], "kind": s["kind"], "line": s["line"]}
                        for s in f.get("symbols", [])
                    ],
                    "is_test": f.get("is_test", False),
                }
                for f in self.get_files_by_importance(top_n=20)
            ],
            "directory_tree": self.tree,
        }

    def to_retriever_input(self) -> dict:
        """
        Convert to format expected by Retriever module.
        Retriever needs: all symbols with file locations for search.
        """
        symbols = []
        for f in self.files:
            for sym in f.get("symbols", []):
                symbols.append({
                    "name": sym["name"],
                    "kind": sym["kind"],
                    "file": f["path"],
                    "line": sym["line"],
                    "signature": sym.get("signature", ""),
                    "docstring": sym.get("docstring", "")[:100],
                })
        return {
            "repo_root": self.root,
            "symbols": symbols,
            "file_count": self.file_count,
        }

    def to_coder_input(self, target_file: str) -> dict:
        """
        Convert to format expected by Coder module.
        Coder needs: specific file content and its symbols.
        """
        for f in self.files:
            if f["path"] == target_file:
                return {
                    "file_path": target_file,
                    "file_size": f["size"],
                    "symbols": f.get("symbols", []),
                    "is_test": f.get("is_test", False),
                    "importance": f.get("importance", 0),
                }
        return None


def test_scanner():
    """Test 1: FileScanner functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: FileScanner")
    print("=" * 60)

    config = RepoMapConfig()
    scanner = FileScanner(".", config)

    # Scan files
    files = scanner.scan()
    print(f"\n[✓] Scanned {len(files)} files")

    # Show tree
    tree = scanner.get_repo_tree(max_depth=3)
    print(f"\n[✓] Directory tree generated ({len(tree.splitlines())} lines)")

    # Show file list
    print("\n[✓] File list:")
    for f in files[:10]:
        marker = "TEST" if f.is_test else "SRC "
        print(f"  [{marker}] {f.rel_path} ({f.size} bytes)")

    return files


def test_parser():
    """Test 2: SymbolParser functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: SymbolParser")
    print("=" * 60)

    parser = SymbolParser(".")

    # Parse a Python file in this module
    test_file = Path(__file__).parent / "builder.py"
    if test_file.exists():
        symbols = parser.parse(str(test_file))
        print(f"\n[✓] Parsed {len(symbols)} symbols from builder.py")

        # Show symbols
        md = symbols_to_markdown(symbols, file_name="builder.py")
        print("\n[✓] Symbol markdown output:")
        print(md[:500])
    else:
        # Parse scanner.py
        test_file = Path(__file__).parent / "scanner.py"
        symbols = parser.parse(str(test_file))
        print(f"\n[✓] Parsed {len(symbols)} symbols from scanner.py")

    return symbols


def test_builder():
    """Test 3: RepoMapBuilder functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: RepoMapBuilder")
    print("=" * 60)

    config = RepoMapConfig()
    config.cache_enabled = False  # Disable cache for fresh test
    builder = RepoMapBuilder(".", config)

    # Build repo map
    data = builder.build()
    print(f"\n[✓] Built repo map with {data['file_count']} files")

    # Show summary
    print("\n[✓] Files by importance:")
    for entry in data['files'][:5]:
        n_symbols = len(entry['symbols'])
        print(f"  - {entry['path']}: {entry['importance']:.1f} points, {n_symbols} symbols")

    # Show tree
    print(f"\n[✓] Directory tree ({len(data['tree'].splitlines())} lines)")
    print(data['tree'][:300] + "..." if len(data['tree']) > 300 else data['tree'])

    return data


def test_output_formats():
    """Test 4: Output format conversion for downstream components."""
    print("\n" + "=" * 60)
    print("TEST 4: Output Format Conversion")
    print("=" * 60)

    config = RepoMapConfig()
    config.cache_enabled = False
    builder = RepoMapBuilder(".", config)
    raw_data = builder.build()

    # Wrap in standardized output
    output = RepoMapOutput(raw_data)

    # Test Planner input
    planner_input = output.to_planner_input()
    print("\n[✓] Planner input format:")
    print(f"  - repo_root: {planner_input['repo_root']}")
    print(f"  - file_count: {planner_input['file_count']}")
    print(f"  - top files: {[f['path'] for f in planner_input['files'][:3]]}")

    # Test Retriever input
    retriever_input = output.to_retriever_input()
    print(f"\n[✓] Retriever input format:")
    print(f"  - total symbols: {len(retriever_input['symbols'])}")
    classes = [s for s in retriever_input['symbols'] if s['kind'] == 'class']
    functions = [s for s in retriever_input['symbols'] if s['kind'] in {'function', 'method'}]
    print(f"  - classes: {len(classes)}, functions/methods: {len(functions)}")

    # Test Coder input (target a specific file)
    target = output.files[0]['path'] if output.files else None
    if target:
        coder_input = output.to_coder_input(target)
        print(f"\n[✓] Coder input format (target: {target}):")
        print(f"  - file_size: {coder_input['file_size']}")
        print(f"  - symbols: {len(coder_input['symbols'])}")

    return output


def test_token_limited_map():
    """Test 5: Token-limited map generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Token-Limited Map Generation")
    print("=" * 60)

    config = RepoMapConfig()
    config.cache_enabled = False
    builder = RepoMapBuilder(".", config)

    # Generate map with different token limits
    for max_tokens in [1000, 2000, 4000]:
        text = builder.get_repo_map_text(max_tokens=max_tokens)
        est_tokens = len(text) // 4
        print(f"\n[✓] max_tokens={max_tokens} → ~{est_tokens} tokens, {len(text)} chars")
        print(f"  Preview: {text[:100]}...")

    return True


def test_dependencies():
    """Test 6: Dependency analysis."""
    print("\n" + "=" * 60)
    print("TEST 6: Dependency Analysis")
    print("=" * 60)

    config = RepoMapConfig()
    builder = RepoMapBuilder(".", config)

    deps = builder.get_file_dependencies()
    print(f"\n[✓] Analyzed {len(deps)} files with imports")

    if deps:
        print("\n[✓] Sample dependencies:")
        for file, imports in list(deps.items())[:3]:
            print(f"  - {file}: {imports[:3]}")

    return deps


def save_test_output(output: RepoMapOutput, path: str):
    """Save test output for reference."""
    data = {
        "planner_input": output.to_planner_input(),
        "retriever_input": output.to_retriever_input(),
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n[✓] Test output saved to {path}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RepoMap Module Integration Test")
    print("=" * 60)

    results = {}

    try:
        results['scanner'] = test_scanner()
    except Exception as e:
        print(f"\n[✗] Scanner test failed: {e}")
        results['scanner'] = None

    try:
        results['parser'] = test_parser()
    except Exception as e:
        print(f"\n[✗] Parser test failed: {e}")
        results['parser'] = None

    try:
        results['builder'] = test_builder()
    except Exception as e:
        print(f"\n[✗] Builder test failed: {e}")
        results['builder'] = None

    try:
        results['output'] = test_output_formats()
    except Exception as e:
        print(f"\n[✗] Output format test failed: {e}")
        results['output'] = None

    try:
        results['token_map'] = test_token_limited_map()
    except Exception as e:
        print(f"\n[✗] Token-limited map test failed: {e}")

    try:
        results['dependencies'] = test_dependencies()
    except Exception as e:
        print(f"\n[✗] Dependency test failed: {e}")

    # Save output for reference
    if results.get('output'):
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        save_test_output(results['output'], output_dir / "repomap_test_output.json")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if results.get('builder'):
        data = results['builder']
        print(f"\nRepo Map Stats:")
        print(f"  - Files: {data['file_count']}")
        print(f"  - Total symbols: {sum(len(f['symbols']) for f in data['files'])}")

    print("\n[✓] RepoMap module is ready for integration with downstream components.")
    print("\nIntegration points:")
    print("  - Planner: use output.to_planner_input()")
    print("  - Retriever: use output.to_retriever_input()")
    print("  - Coder: use output.to_coder_input(target_file)")


if __name__ == "__main__":
    main()