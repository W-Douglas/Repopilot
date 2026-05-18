#!/usr/bin/env python3
"""
RepoMap CLI - Command line interface for repository mapping.
"""
import argparse
import json
import sys
from pathlib import Path

# Handle both module and direct execution
_cwd = Path.cwd()
_repomap_dir = _cwd / "repomap"
if _repomap_dir.exists() and not any(p.name == "repomap" for p in _cwd.parents):
    # Running from parent directory with repomap as subdir
    sys.path.insert(0, str(_cwd))

try:
    from repomap.config import RepoMapConfig
    from repomap.scanner import FileScanner
    from repomap.parser import SymbolParser, symbols_to_markdown
    from repomap.builder import RepoMapBuilder
except ImportError:
    # Fallback for direct execution
    from config import RepoMapConfig
    from scanner import FileScanner
    from parser import SymbolParser, symbols_to_markdown
    from builder import RepoMapBuilder


def cmd_scan(args):
    """Scan and list files in repository."""
    config = RepoMapConfig()
    scanner = FileScanner(args.root, config)

    if args.tree:
        print(scanner.get_repo_tree(max_depth=args.depth))
    else:
        files = scanner.scan()
        print(f"Found {len(files)} files:\n")
        for f in files:
            marker = "TEST" if f.is_test else "SRC "
            print(f"  [{marker}] {f.rel_path} ({f.size:,} bytes)")


def cmd_parse(args):
    """Parse a file and show its symbols."""
    config = RepoMapConfig()
    parser = SymbolParser(args.root)

    symbols = parser.parse(args.file)
    if not symbols:
        print(f"No symbols found in {args.file}")
        return

    print(f"\n=== Symbols in {args.file} ===\n")
    print(symbols_to_markdown(symbols))


def cmd_build(args):
    """Build repository map."""
    config = RepoMapConfig.from_env()
    config.cache_enabled = not args.no_cache

    builder = RepoMapBuilder(args.root, config)

    if args.output:
        data = builder.build(force_refresh=args.refresh)
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Repo map saved to {args.output}")
    else:
        data = builder.build(force_refresh=args.refresh)
        print(f"\n=== Repo Map Summary ===\n")
        print(f"Root: {data['root']}")
        print(f"Total files: {data['file_count']}")
        print("\nTop files by importance:")

        for entry in data['files'][:args.top]:
            n_symbols = len(entry['symbols'])
            n_classes = sum(1 for s in entry['symbols'] if s.get('kind') == 'class')
            n_funcs = sum(1 for s in entry['symbols'] if s.get('kind') in {'function', 'method'})
            print(f"  {entry['path']}")
            print(f"    - importance: {entry['importance']:.1f}")
            print(f"    - {n_classes} classes, {n_funcs} functions, {n_symbols} total symbols")


def cmd_map(args):
    """Generate token-limited repo map text."""
    config = RepoMapConfig.from_env()
    builder = RepoMapBuilder(args.root, config)

    text = builder.get_repo_map_text(
        max_tokens=args.max_tokens,
        mentioned_names=set(args.mentions) if args.mentions else None,
    )

    print(text)


def cmd_understand(args):
    """Generate LLM-powered global understanding."""
    from analyzer import GlobalAnalyzer, AnalyzerConfig

    analyzer_config = AnalyzerConfig.from_env()

    if not analyzer_config.api_key:
        print("Error: DEEPSEEK_API_KEY not set.")
        print("Set it with: export DEEPSEEK_API_KEY=your_key")
        sys.exit(1)

    builder = RepoMapBuilder(args.root, RepoMapConfig.from_env())
    analyzer = GlobalAnalyzer(analyzer_config)

    print("Building repo map...")
    repo_data = builder.build()

    print("Generating global understanding...")
    understanding = analyzer.generate_understanding(
        repo_tree=repo_data.get("tree", ""),
        file_symbols="",
        summary=repo_data,
        query=args.query,
    )

    if args.output:
        with open(args.output, 'w') as f:
            f.write(understanding.to_markdown())
        print(f"Saved to {args.output}")
    else:
        print(understanding.to_markdown())


def cmd_suggest(args):
    """Suggest files for a given issue."""
    from analyzer import GlobalAnalyzer, AnalyzerConfig

    analyzer_config = AnalyzerConfig.from_env()

    if not analyzer_config.api_key:
        print("Error: DEEPSEEK_API_KEY not set.")
        sys.exit(1)

    builder = RepoMapBuilder(args.root, RepoMapConfig.from_env())
    analyzer = GlobalAnalyzer(analyzer_config)

    print("Building repo map...")
    repo_data = builder.build()

    print(f"Analyzing issue: {args.issue[:100]}...")
    suggestions = analyzer.suggest_files_for_issue(args.issue, repo_data)

    print("\nSuggested files to examine:")
    for i, f in enumerate(suggestions, 1):
        print(f"  {i}. {f}")


def cmd_dependencies(args):
    """Analyze file dependencies."""
    config = RepoMapConfig()
    builder = RepoMapBuilder(args.root, config)

    deps = builder.get_file_dependencies()

    if args.file:
        if args.file in deps:
            print(f"\n=== Dependencies of {args.file} ===\n")
            for dep in deps[args.file]:
                print(f"  - {dep}")
        else:
            print(f"No dependencies found for {args.file}")
    else:
        print(f"\n=== File Dependencies ({len(deps)} files) ===\n")
        for file, imports in sorted(deps.items()):
            if imports:
                print(f"{file}:")
                for imp in imports[:5]:
                    print(f"  - {imp}")


def main():
    parser = argparse.ArgumentParser(
        description="RepoMap - Repository structure understanding tool (static analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=".", help="Repository root path")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # scan command
    p_scan = subparsers.add_parser("scan", help="Scan repository files")
    p_scan.add_argument("--tree", action="store_true", help="Show directory tree")
    p_scan.add_argument("--depth", type=int, default=4, help="Tree depth")
    p_scan.set_defaults(func=cmd_scan)

    # parse command
    p_parse = subparsers.add_parser("parse", help="Parse a file for symbols")
    p_parse.add_argument("file", help="File to parse")
    p_parse.set_defaults(func=cmd_parse)

    # build command
    p_build = subparsers.add_parser("build", help="Build repository map")
    p_build.add_argument("-o", "--output", help="Output file (JSON)")
    p_build.add_argument("--no-cache", action="store_true", help="Disable cache")
    p_build.add_argument("--refresh", action="store_true", help="Force refresh")
    p_build.add_argument("--top", type=int, default=10, help="Top N files to show")
    p_build.set_defaults(func=cmd_build)

    # map command
    p_map = subparsers.add_parser("map", help="Generate token-limited repo map")
    p_map.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    p_map.add_argument("--mentions", nargs="+", help="Names to boost importance")
    p_map.set_defaults(func=cmd_map)

    # understand command (uses LLM via analyzer module)
    p_understand = subparsers.add_parser("understand", help="Generate global understanding via LLM")
    p_understand.add_argument("-o", "--output", help="Output file")
    p_understand.add_argument("-q", "--query", help="Context query")
    p_understand.set_defaults(func=cmd_understand)

    # suggest command (uses LLM via analyzer module)
    p_suggest = subparsers.add_parser("suggest", help="Suggest files for an issue")
    p_suggest.add_argument("issue", help="Issue description")
    p_suggest.set_defaults(func=cmd_suggest)

    # dependencies command
    p_deps = subparsers.add_parser("deps", help="Analyze dependencies")
    p_deps.add_argument("-f", "--file", help="Specific file")
    p_deps.set_defaults(func=cmd_dependencies)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()