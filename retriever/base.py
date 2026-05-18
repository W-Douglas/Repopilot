"""
Base Retriever Interface - defines contract for all retrievers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


class ChunkType(Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    BLOCK = "block"  # Generic code block
    COMMENT = "comment"
    DOCSTRING = "docstring"


@dataclass
class Chunk:
    """Represents a code chunk for retrieval."""
    id: str
    content: str
    file_path: str
    chunk_type: ChunkType
    start_line: int
    end_line: int
    symbols: List[str] = field(default_factory=list)  # Function/class names
    docstring: str = ""
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbols": self.symbols,
            "docstring": self.docstring,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            content=data["content"],
            file_path=data["file_path"],
            chunk_type=ChunkType(data["chunk_type"]),
            start_line=data["start_line"],
            end_line=data["end_line"],
            symbols=data.get("symbols", []),
            docstring=data.get("docstring", ""),
            language=data.get("language", "python"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk: Chunk
    score: float
    rank: int = 0
    query: str = ""
    method: str = "unknown"  # "bm25", "embedding", "rerank"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "query": self.query,
            "method": self.method,
        }


@dataclass
class RetrievalQuery:
    """Query for retrieval."""
    text: str
    file_filter: Optional[List[str]] = None  # Limit to specific files
    chunk_type_filter: Optional[List[ChunkType]] = None
    top_k: int = 10
    min_score: float = 0.0

    @classmethod
    def from_text(cls, text: str, top_k: int = 10) -> "RetrievalQuery":
        return cls(text=text, top_k=top_k)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    Implementations must provide:
    - index(): Build or load index
    - retrieve(): Query the index
    - add_chunks(): Add new chunks to index
    """

    name: str = "base"
    description: str = ""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._chunks: List[Chunk] = []
        self._indexed = False

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """Build index from chunks."""
        pass

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve relevant chunks for query."""
        pass

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        self._chunks.extend(chunks)

    def clear(self) -> None:
        """Clear all chunks and index."""
        self._chunks.clear()
        self._indexed = False

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "chunks": [c.to_dict() for c in self._chunks],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        path = Path(path)
        if not path.exists():
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self._chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        self.index(self._chunks)

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def is_indexed(self) -> bool:
        """Whether index is built."""
        return self._indexed