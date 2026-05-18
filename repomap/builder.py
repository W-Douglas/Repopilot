"""
RepoMap Builder - builds structured repository maps from parsed symbols.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Handle both module and direct execution
_cwd = Path.cwd()
_repomap_dir = _cwd / "repomap"
if _repomap_dir.exists() and not any(p.name == "repomap" for p in _cwd.parents):
    sys.path.insert(0, str(_cwd))
    from repomap.scanner import FileScanner, ScannedFile
    from repomap.parser import SymbolParser, Symbol, symbols_to_markdown
else:
    from scanner import FileScanner, ScannedFile
    from parser import SymbolParser, Symbol, symbols_to_markdown


@dataclass
class FileEntry:
    """Represents a file in the repo map."""
    path: str
    size: int
    is_test: bool
    symbols: List[Dict[str, Any]]
    importance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "is_test": self.is_test,
            "symbols": self.symbols,
            "importance": self.importance,
        }


class RepoMapBuilder:
    """Builds a structured repository map."""

    def __init__(self, root: str, config):
        self.root = Path(root).resolve()
        self.config = config
        self.scanner = FileScanner(root, config)
        self.parser = SymbolParser(root)
        self.cache = {}

    def _get_cache_key(self) -> str:
        """Generate cache key based on file mtimes."""
        files = self.scanner.scan()
        key_parts = [f"{f.rel_path}:{f.size}" for f in files]
        key_parts.sort()
        return str(hash(tuple(key_parts)))

    def _should_use_cache(self) -> bool:
        """Check if cached repo map exists."""
        if not self.config.cache_enabled:
            return False

        cache_dir = self.root / self.config.cache_dir
        cache_file = cache_dir / "repo_map.json"
        return cache_file.exists()

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load repo map from cache."""
        cache_file = self.root / self.config.cache_dir / "repo_map.json"
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        """Save repo map to cache."""
        cache_dir = self.root / self.config.cache_dir
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "repo_map.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _calculate_importance(self, entry: FileEntry, mentioned_names: set = None) -> float:
        """Calculate file importance based on various signals."""
        score = 0.0

        # Base score from symbol count
        score += len(entry.symbols) * 0.5

        # Higher score for non-test files (usually more important)
        if not entry.is_test:
            score += 2.0

        # Higher score for files with classes (often core logic)
        has_classes = any(s.get("kind") == "class" for s in entry.symbols)
        if has_classes:
            score += 3.0

        # Higher score for files with many methods
        n_methods = sum(1 for s in entry.symbols if s.get("kind") in {"method", "function"})
        if n_methods > 5:
            score += 2.0

        # Mentioned names boost
        if mentioned_names:
            path_lower = entry.path.lower()
            for name in mentioned_names:
                if name.lower() in path_lower:
                    score += 5.0
                    break

        return score

    def build(self, mentioned_names: set = None, force_refresh: bool = False) -> Dict[str, Any]:
        """Build the complete repository map."""
        # Try cache first
        if not force_refresh and self._should_use_cache():
            cached = self._load_from_cache()
            if cached:
                return cached

        files = self.scanner.scan()
        entries = []

        for f in files:
            entry = FileEntry(
                path=f.rel_path,
                size=f.size,
                is_test=f.is_test,
                symbols=[],
            )

            # Parse symbols
            symbols = self.parser.parse(f.path)
            entry.symbols = [s.to_dict() for s in symbols]

            # Calculate importance
            entry.importance = self._calculate_importance(entry, mentioned_names)

            entries.append(entry)

        # Sort by importance (most important first)
        entries.sort(key=lambda e: e.importance, reverse=True)

        result = {
            "root": str(self.root),
            "file_count": len(entries),
            "files": [e.to_dict() for e in entries],
            "tree": self.scanner.get_repo_tree(max_depth=4),
        }

        # Cache result
        if self.config.cache_enabled:
            self._save_to_cache(result)

        return result

    def get_repo_map_text(self, max_tokens: int = 4096, mentioned_names: set = None) -> str:
        """Generate a token-limited repo map text."""
        data = self.build(mentioned_names=mentioned_names)

        lines = ["# Repository Structure\n"]
        lines.append(f"Root: {data['root']}")
        lines.append(f"Total files: {data['file_count']}\n")
        lines.append("## Directory Tree\n")
        lines.append(data['tree'])

        lines.append("\n\n## File Symbols\n")

        # Add file symbols in order of importance
        for entry in data['files']:
            symbols = entry['symbols']
            if not symbols:
                continue

            marker = "[TEST]" if entry['is_test'] else ""
            lines.append(f"\n### {entry['path']} {marker}\n")

            for sym in symbols:
                kind = sym['kind']
                name = sym['name']
                line = sym['line']
                sig = sym.get('signature', '')

                if kind in {'class'}:
                    lines.append(f"- **{name}** [L{line}]")
                    lines.append(f"  - Signature: `{sig}`")
                elif kind in {'function', 'method'}:
                    lines.append(f"- {sig} [L{line}]")
                else:
                    lines.append(f"- {name} ({kind}) [L{line}]")

                if sym.get('docstring'):
                    doc = sym['docstring'][:60]
                    lines.append(f"  - # {doc}")

        # Estimate token count and truncate if needed
        text = "\n".join(lines)
        estimated_tokens = len(text) // 4  # Rough estimate

        if estimated_tokens > max_tokens:
            # Truncate to approximate token limit
            lines = text.split('\n')
            truncated_lines = []
            current_tokens = 0
            for line in lines:
                current_tokens += len(line) // 4 + 1
                if current_tokens > max_tokens:
                    break
                truncated_lines.append(line)
            text = "\n".join(truncated_lines)
            text += f"\n\n... (truncated, showing most important files)\n"

        return text

    def get_file_dependencies(self) -> Dict[str, List[str]]:
        """Analyze import dependencies between files."""
        deps = {}
        files = self.scanner.scan()

        for f in files:
            if f.extension != ".py":
                continue

            try:
                with open(f.path, 'r', encoding='utf-8') as fp:
                    content = fp.read()
            except Exception:
                continue

            imports = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('import '):
                    mod = line.replace('import ', '').split(' as ')[0].split('.')[0].strip()
                    imports.append(mod)
                elif line.startswith('from '):
                    mod = line.replace('from ', '').split(' import')[0].strip()
                    imports.append(mod)

            if imports:
                deps[f.rel_path] = imports

        return deps

    def find_related_files(self, target_file: str, max_results: int = 5) -> List[str]:
        """Find files related to a target file based on imports."""
        deps = self.get_file_dependencies()
        related = []

        target_name = Path(target_file).stem

        for file, imports in deps.items():
            if file == target_file:
                continue
            if any(target_name in imp for imp in imports):
                related.append(file)

        return related[:max_results]


if __name__ == "__main__":
    from config import RepoMapConfig

    config = RepoMapConfig()
    builder = RepoMapBuilder(".", config)

    print("=== Building Repo Map ===\n")
    data = builder.build()

    print(f"Files: {data['file_count']}")
    print("\nTop files by importance:")
    for entry in data['files'][:5]:
        print(f"  {entry['path']} (importance: {entry['importance']:.1f})")

    print("\n=== Repo Map Text (first 3000 chars) ===\n")
    text = builder.get_repo_map_text(max_tokens=2000)
    print(text[:3000])
