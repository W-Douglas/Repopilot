"""
File scanner - discovers source files in a repository.
"""
import os
import sys
import fnmatch
from pathlib import Path
from typing import List, Set, Iterator
from dataclasses import dataclass

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ScannedFile:
    """Represents a discovered source file."""
    path: str  # Absolute path
    rel_path: str  # Relative to repo root
    size: int
    extension: str
    is_test: bool


class FileScanner:
    """Scans repository to discover source files."""

    def __init__(self, root: str, config):
        self.root = Path(root).resolve()
        self.config = config
        self._exclude_set: Set[str] = set(config.exclude_patterns)

    def is_excluded(self, name: str) -> bool:
        """Check if file/directory should be excluded."""
        for pattern in self._exclude_set:
            if fnmatch.fnmatch(name, pattern):
                return True
            if f"/{name}" in pattern or f"{name}/" in pattern:
                return True
        return False

    def is_included(self, path: Path) -> bool:
        """Check if file should be included based on extension."""
        name = path.name
        for pattern in self.config.include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _should_skip_file(self, path: Path) -> bool:
        """Determine if file should be skipped."""
        if not path.is_file():
            return True
        if self.is_excluded(path.name):
            return True
        if not self.is_included(path):
            return True
        try:
            size = path.stat().st_size
            if size > self.config.max_file_size:
                return True
            if size == 0:
                return True
        except OSError:
            return True
        return False

    def _is_test_file(self, path: Path) -> bool:
        """Check if file appears to be a test file."""
        name = path.name.lower()
        return (
            "test_" in name
            or "_test." in name
            or name.startswith("test.")
            or "/tests/" in str(path)
            or "/test/" in str(path)
            or "/__tests__/" in str(path)
            or name.endswith("_test.py")
        )

    def scan(self) -> List[ScannedFile]:
        """Scan repository and return list of discovered files."""
        files = []

        for root, dirs, filenames in os.walk(self.root):
            root_path = Path(root)

            # Check depth
            try:
                depth = len(root_path.relative_to(self.root).parts)
                if depth > self.config.max_depth:
                    dirs.clear()
                    continue
            except ValueError:
                continue

            # Filter excluded directories
            dirs[:] = [d for d in dirs if not self.is_excluded(d)]

            for filename in filenames:
                file_path = root_path / filename

                if self._should_skip_file(file_path):
                    continue

                try:
                    rel_path = file_path.relative_to(self.root)
                    size = file_path.stat().st_size

                    files.append(ScannedFile(
                        path=str(file_path),
                        rel_path=str(rel_path),
                        size=size,
                        extension=file_path.suffix,
                        is_test=self._is_test_file(file_path),
                    ))
                except (ValueError, OSError):
                    continue

        return files

    def get_repo_tree(self, max_depth: int = 3) -> str:
        """Generate a tree view of the repository structure."""
        lines = [f"/ ({self.root.name})"]

        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)

            try:
                depth = len(root_path.relative_to(self.root).parts)
                if depth > max_depth:
                    dirs.clear()
                    continue
            except ValueError:
                continue

            dirs[:] = [d for d in dirs if not self.is_excluded(d)]
            dirs.sort()
            files = [f for f in files if not self.is_excluded(f)]
            files.sort()

            prefix = "  " * depth
            for d in dirs:
                lines.append(f"{prefix}├── 📁 {d}/")

            for i, f in enumerate(files):
                is_last = (i == len(files) - 1 and len(dirs) == 0)
                connector = "└──" if is_last else "├──"
                marker = "🔵" if self.is_included(Path(f)) else "⚪"
                lines.append(f"{prefix}{connector} {marker} {f}")

        return "\n".join(lines)


if __name__ == "__main__":
    from config import RepoMapConfig

    config = RepoMapConfig()
    scanner = FileScanner(".", config)

    print("=== Repository Tree ===")
    print(scanner.get_repo_tree(max_depth=3))

    print("\n=== Discovered Files ===")
    files = scanner.scan()
    for f in files[:20]:
        marker = "TEST" if f.is_test else "SRC "
        print(f"  [{marker}] {f.rel_path} ({f.size} bytes)")
