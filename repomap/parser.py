"""
AST parser - extracts symbols from source files.
"""
import ast
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Symbol:
    """Represents a code symbol (function, class, etc.)."""
    name: str
    kind: str  # "function", "class", "method", "import", "constant"
    line: int
    end_line: int
    file: str
    signature: str = ""
    docstring: str = ""
    decorators: List[str] = field(default_factory=list)
    children: List["Symbol"] = field(default_factory=list)
    complexity: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "end_line": self.end_line,
            "file": self.file,
            "signature": self.signature,
            "docstring": self.docstring[:100] if self.docstring else "",
            "decorators": self.decorators,
            "complexity": self.complexity,
        }


class PythonSymbolParser:
    """Parses Python files to extract symbols using AST."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _get_docstring(self, node: ast.AST) -> str:
        """Extract docstring from AST node."""
        doc = ast.get_docstring(node)
        return doc or ""

    def _get_decorators(self, node: ast.AST) -> List[str]:
        """Extract decorators from AST node."""
        decorators = []
        for d in getattr(node, 'decorator_list', []):
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                # Handle @module.decorator
                try:
                    if hasattr(ast, 'unparse'):
                        decorators.append(ast.unparse(d))
                    else:
                        decorators.append(f"{d.value.id}.{d.attr}")
                except Exception:
                    decorators.append(d.attr)
            else:
                try:
                    decorators.append(ast.unparse(d))
                except Exception:
                    decorators.append(str(d))
        return decorators

    def _count_queries(self, node: ast.AST) -> int:
        """Rough complexity measure - count branches/loops."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                count += 1
        return max(1, count)

    def _parse_function(self, node: ast.FunctionDef, file: str) -> Symbol:
        """Parse a function/method definition."""
        try:
            sig = ast.get_source_segment(self._get_source(file), node) or ""
            if not sig:
                args = [a.arg for a in node.args.args]
                defaults = node.args.defaults
                n_defaults = len(defaults)
                n_args = len(args)
                params = []
                for i, arg in enumerate(args):
                    default_idx = i - n_args + n_defaults
                    if default_idx >= 0:
                        params.append(f"{arg}=...")
                    else:
                        params.append(arg)
                sig = f"def {node.name}({', '.join(params)})"
        except Exception:
            sig = f"def {node.name}(...)"

        return Symbol(
            name=node.name,
            kind="method" if isinstance(node, ast.FunctionDef) else "function",
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            file=file,
            signature=sig,
            docstring=self._get_docstring(node),
            decorators=self._get_decorators(node),
            complexity=self._count_queries(node),
        )

    def _parse_class(self, node: ast.ClassDef, file: str) -> Symbol:
        """Parse a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else "...")

        sig = f"class {node.name}"
        if bases:
            sig += f"({', '.join(bases)})"

        children = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._parse_function(item, file)
                method.kind = "method"
                children.append(method)

        return Symbol(
            name=node.name,
            kind="class",
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            file=file,
            signature=sig,
            docstring=self._get_docstring(node),
            decorators=self._get_decorators(node),
            children=children,
            complexity=self._count_queries(node),
        )

    def _get_source(self, file: str) -> str:
        """Get source code for a file."""
        try:
            with open(file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Extract source code for a node."""
        try:
            if hasattr(ast, 'get_source_segment'):
                return ast.get_source_segment(source, node) or ""
            return ""
        except Exception:
            return ""

    def parse_file(self, file_path: str) -> List[Symbol]:
        """Parse a Python file and extract all top-level symbols."""
        symbols = []
        rel_path = self._get_rel_path(file_path)

        try:
            source = self._get_source(file_path)
            if not source:
                return symbols

            tree = ast.parse(source, filename=file_path)
        except (SyntaxError, ValueError) as e:
            print(f"Warning: Failed to parse {rel_path}: {e}")
            return symbols

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(self._parse_class(node, rel_path))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._parse_function(node, rel_path))

        return symbols

    def _get_rel_path(self, file_path: str) -> str:
        """Get path relative to root."""
        try:
            return str(Path(file_path).relative_to(self.root))
        except ValueError:
            return file_path


class SymbolParser:
    """Unified symbol parser supporting multiple languages."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.python_parser = PythonSymbolParser(root)

    def parse(self, file_path: str) -> List[Symbol]:
        """Parse a file based on its extension."""
        ext = Path(file_path).suffix.lower()

        if ext in {'.py', '.pyx', '.pyi'}:
            return self.python_parser.parse_file(file_path)

        # For other languages, return empty for MVP
        return []


def symbols_to_markdown(symbols: List[Symbol], file_name: Optional[str] = None) -> str:
    """Convert symbols to markdown format."""
    lines = []

    if file_name:
        lines.append(f"### {file_name}")

    current_file = None
    for sym in sorted(symbols, key=lambda s: (s.file, s.line)):
        if sym.file != current_file:
            current_file = sym.file
            if file_name is None:
                lines.append(f"\n### {current_file}")

        indent = "    " if file_name else "        "
        lines.append(f"{indent}{sym.signature} [L{sym.line}]")

        if sym.docstring:
            first_line = sym.docstring.split('\n')[0][:60]
            if first_line:
                lines.append(f"{indent}    # {first_line}")

        if sym.children:
            for child in sorted(sym.children, key=lambda s: s.line):
                lines.append(f"{indent}    {child.signature} [L{child.line}]")
                if child.docstring:
                    first_line = child.docstring.split('\n')[0][:60]
                    if first_line:
                        lines.append(f"{indent}        # {first_line}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from config import RepoMapConfig
    from scanner import FileScanner

    config = RepoMapConfig()
    scanner = FileScanner(".", config)
    parser = SymbolParser(".")

    files = scanner.scan()
    py_files = [f for f in files if f.extension == ".py"][:10]

    print("=== Symbol Extraction ===\n")
    for f in py_files:
        symbols = parser.parse(f.path)
        if symbols:
            print(f"\n{f.rel_path}:")
            print(symbols_to_markdown(symbols))
