"""
Code Chunker - breaks source code into retrievable chunks

This module handles:
- Language-aware code parsing
- Semantic chunking (functions, classes, methods)
- Line-based chunking with overlap
- Metadata extraction (symbols, docstrings, imports)
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Iterator
from dataclasses import dataclass, field

from .base import Chunk, ChunkType


@dataclass
class ChunkerConfig:
    """Configuration for code chunking."""
    max_chunk_lines: int = 100
    min_chunk_lines: int = 3
    overlap_lines: int = 5
    include_docstrings: bool = True
    include_comments: bool = False
    chunk_by_symbol: bool = True  # Split by function/class
    max_chunks_per_file: int = 50


class CodeChunker:
    """
    Chunks source code into retrieval-friendly pieces.

    Supports:
    - Python AST-based chunking (functions, classes, methods)
    - Generic line-based chunking for other languages
    - Configurable chunk size and overlap
    """

    def __init__(self, root: str, config: Optional[ChunkerConfig] = None):
        self.root = Path(root).resolve()
        self.config = config or ChunkerConfig()

    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Chunk a source file into pieces.

        Args:
            file_path: Path to source file (absolute or relative to root)

        Returns:
            List of Chunks
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root / path

        if not path.exists():
            return []

        ext = path.suffix.lower()
        rel_path = str(path.relative_to(self.root))

        if ext in {'.py', '.pyx', '.pyi'}:
            return self._chunk_python(path, rel_path)
        else:
            return self._chunk_generic(path, rel_path)

    def _chunk_python(self, file_path: Path, rel_path: str) -> List[Chunk]:
        """Chunk Python file using AST."""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            content_lines = source.split('\n')
        except (SyntaxError, ValueError):
            return self._chunk_generic(file_path, rel_path)

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree) or ""

        # Process top-level definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunk = self._extract_class_chunk(node, rel_path, content_lines)
                if chunk:
                    chunks.append(chunk)

                    # Also add method chunks
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_chunk = self._extract_function_chunk(
                                item, rel_path, content_lines,
                                class_name=node.name
                            )
                            if method_chunk:
                                chunks.append(method_chunk)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function_chunk(node, rel_path, content_lines)
                if chunk:
                    chunks.append(chunk)

        # Limit chunks per file
        if len(chunks) > self.config.max_chunks_per_file:
            # Keep most important chunks (classes first, then by size)
            chunks = sorted(chunks, key=lambda c: (
                0 if c.chunk_type == ChunkType.CLASS else 1,
                -(c.end_line - c.start_line)
            ))[:self.config.max_chunks_per_file]

        return chunks

    def _extract_class_chunk(
        self,
        node: ast.ClassDef,
        rel_path: str,
        content_lines: List[str],
    ) -> Optional[Chunk]:
        """Extract a class as a chunk."""
        start = node.lineno - 1
        end = node.end_lineno or start + 1

        lines = content_lines[start:end]
        content = '\n'.join(lines)

        # Skip if too small
        if len(lines) < self.config.min_chunk_lines:
            return None

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract symbols
        symbols = [node.name]
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                symbols.append(item.name)

        # Get decorators
        decorators = []
        for d in node.decorator_list:
            try:
                decorators.append(ast.unparse(d) if hasattr(ast, 'unparse') else d.attr if hasattr(d, 'attr') else str(d))
            except Exception:
                pass

        chunk_id = f"{rel_path}:{node.lineno}-{node.end_lineno}"

        return Chunk(
            id=chunk_id,
            content=content,
            file_path=rel_path,
            chunk_type=ChunkType.CLASS,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            symbols=symbols,
            docstring=docstring,
            language="python",
            metadata={
                "decorators": decorators,
                "bases": [ast.unparse(b) for b in node.bases] if hasattr(ast, 'unparse') else [],
            },
        )

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef,
        rel_path: str,
        content_lines: List[str],
        class_name: Optional[str] = None,
    ) -> Optional[Chunk]:
        """Extract a function/method as a chunk."""
        start = node.lineno - 1
        end = node.end_lineno or start + 1

        lines = content_lines[start:end]
        content = '\n'.join(lines)

        # Skip if too small
        if len(lines) < self.config.min_chunk_lines:
            return None

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Build signature
        try:
            args = [a.arg for a in node.args.args]
            sig = f"def {node.name}({', '.join(args)})"
        except Exception:
            sig = f"def {node.name}(...)"

        chunk_type = ChunkType.METHOD if class_name else ChunkType.FUNCTION

        # Get decorators
        decorators = []
        for d in node.decorator_list:
            try:
                decorators.append(ast.unparse(d) if hasattr(ast, 'unparse') else str(d))
            except Exception:
                pass

        chunk_id = f"{rel_path}:{node.lineno}-{node.end_lineno}"

        symbols = [node.name]
        if class_name:
            symbols.append(f"{class_name}.{node.name}")

        return Chunk(
            id=chunk_id,
            content=content,
            file_path=rel_path,
            chunk_type=chunk_type,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            symbols=symbols,
            docstring=docstring,
            language="python",
            metadata={
                "signature": sig,
                "decorators": decorators,
                "class_name": class_name,
            },
        )

    def _chunk_generic(self, file_path: Path, rel_path: str) -> List[Chunk]:
        """Generic line-based chunking for non-Python files."""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            return []

        # Detect language
        ext = file_path.suffix.lower()
        lang = self._detect_language(ext)

        total_lines = len(lines)
        chunk_size = self.config.max_chunk_lines
        overlap = self.config.overlap_lines

        # Calculate chunk positions
        start = 0
        chunk_idx = 0

        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            content = ''.join(lines[start:end])

            # Skip if too small
            if end - start < self.config.min_chunk_lines:
                break

            chunk_id = f"{rel_path}:{start + 1}-{end}"

            # Try to extract symbols from first line
            symbols = self._extract_symbols_from_line(lines[start]) if lines else []

            chunk = Chunk(
                id=chunk_id,
                content=content,
                file_path=rel_path,
                chunk_type=ChunkType.BLOCK,
                start_line=start + 1,
                end_line=end,
                symbols=symbols,
                language=lang,
                metadata={"chunk_index": chunk_idx},
            )

            chunks.append(chunk)

            # Move window with overlap
            start = end - overlap
            chunk_idx += 1

            if start >= total_lines:
                break

        return chunks

    def _detect_language(self, ext: str) -> str:
        """Detect programming language from extension."""
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.vue': 'vue',
            '.svelte': 'svelte',
        }
        return lang_map.get(ext.lower(), 'unknown')

    def _extract_symbols_from_line(self, line: str) -> List[str]:
        """Extract function/class names from a line."""
        symbols = []

        # Match function definitions
        func_match = re.search(r'(def|function|func|public|private|protected)\s+(\w+)', line)
        if func_match:
            symbols.append(func_match.group(2))

        # Match class definitions
        class_match = re.search(r'class\s+(\w+)', line)
        if class_match:
            symbols.append(class_match.group(1))

        return symbols

    def chunk_directory(self, directory: str = ".") -> List[Chunk]:
        """Chunk all files in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.root / dir_path

        all_chunks = []

        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)

        return all_chunks

    def chunk_string(
        self,
        content: str,
        file_path: str = "snippet",
        chunk_id_prefix: str = "",
    ) -> List[Chunk]:
        """
        Chunk arbitrary code string.

        Useful for testing or chunking generated code.
        """
        lines = content.split('\n')
        total_lines = len(lines)
        chunk_size = self.config.max_chunk_lines
        overlap = self.config.overlap_lines

        chunks = []
        start = 0
        chunk_idx = 0

        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            snippet = '\n'.join(lines[start:end])

            if end - start < self.config.min_chunk_lines:
                break

            chunk = Chunk(
                id=f"{chunk_id_prefix}chunk_{chunk_idx}",
                content=snippet,
                file_path=file_path,
                chunk_type=ChunkType.BLOCK,
                start_line=start + 1,
                end_line=end,
                symbols=self._extract_symbols_from_line(lines[start]) if lines else [],
            )
            chunks.append(chunk)

            start = end - overlap
            chunk_idx += 1

            if start >= total_lines:
                break

        return chunks


def quick_chunk(content: str, file_path: str = "snippet") -> List[Chunk]:
    """Quick chunking with defaults."""
    chunker = CodeChunker(".")
    return chunker.chunk_string(content, file_path)