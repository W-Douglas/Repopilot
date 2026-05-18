"""
RepoMap - Repository structure understanding via static analysis.
Inspired by Aider's RepoMap - provides LLM with low-cost global repo awareness.

This module provides:
1. File scanning and filtering
2. AST-based symbol extraction (class/function/method)
3. Structured repo map generation
4. File importance ranking based on symbol count and type

NO LLM is used in this module - it is purely static code analysis.
"""

from .scanner import FileScanner
from .parser import SymbolParser, Symbol
from .builder import RepoMapBuilder
from .config import RepoMapConfig

__all__ = ["FileScanner", "SymbolParser", "Symbol", "RepoMapBuilder", "RepoMapConfig"]