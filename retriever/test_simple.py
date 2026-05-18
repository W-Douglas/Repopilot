"""
Simple test for retriever module - no model downloads required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import (
    BM25Retriever,
    CodeChunker,
    Chunk,
    ChunkType,
    RetrievalQuery,
)


def test_bm25_retriever():
    """Test BM25 retriever."""
    print("\n=== Testing BM25 Retriever ===")

    chunks = [
        Chunk(
            id="1",
            content="def calculate_sum(a, b): return a + b",
            file_path="math.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=1,
            symbols=["calculate_sum"],
        ),
        Chunk(
            id="2",
            content="def calculate_product(a, b): return a * b",
            file_path="math.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=3,
            end_line=3,
            symbols=["calculate_product"],
        ),
        Chunk(
            id="3",
            content="class MathHelper: def double(self, x): return x * 2",
            file_path="helper.py",
            chunk_type=ChunkType.CLASS,
            start_line=1,
            end_line=3,
            symbols=["MathHelper", "double"],
        ),
    ]

    retriever = BM25Retriever(".")
    retriever.index(chunks)

    results = retriever.retrieve(RetrievalQuery.from_text("calculate function", top_k=3))

    print(f"Query: 'calculate function'")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - [{r.rank}] {r.chunk.file_path}:{r.chunk.start_line} (score: {r.score:.3f})")
        print(f"    {r.chunk.content[:50]}...")

    print(f"\nStats: {retriever.get_stats()}")
    assert len(results) == 3, "Should have 3 results"
    print("BM25 test passed!")


def test_code_chunker():
    """Test code chunking."""
    print("\n=== Testing Code Chunker ===")

    code = '''
class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b

def standalone_function(x):
    """A standalone function."""
    return x * 2
'''

    chunker = CodeChunker(".")
    chunks = chunker.chunk_string(code, "test.py", "calc")

    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n[{chunk.id}] {chunk.chunk_type.value} at line {chunk.start_line}-{chunk.end_line}")
        print(f"Symbols: {chunk.symbols}")
        if chunk.docstring:
            print(f"Doc: {chunk.docstring}")

    assert len(chunks) >= 2, "Should have at least 2 chunks (class + function)"
    print("\nChunker test passed!")


def test_chunk_types():
    """Test all chunk types."""
    print("\n=== Testing Chunk Types ===")
    for ct in ChunkType:
        print(f"  - {ct.value}")

    # Test serialization
    chunk = Chunk(
        id="test1",
        content="def foo(): pass",
        file_path="test.py",
        chunk_type=ChunkType.FUNCTION,
        start_line=1,
        end_line=1,
        symbols=["foo"],
    )
    data = chunk.to_dict()
    restored = Chunk.from_dict(data)
    assert restored.id == chunk.id
    assert restored.content == chunk.content
    print("Chunk serialization test passed!")


def test_query_filtering():
    """Test query filtering."""
    print("\n=== Testing Query Filtering ===")

    chunks = [
        Chunk(id="1", content="def login(): pass", file_path="auth.py",
              chunk_type=ChunkType.FUNCTION, start_line=1, end_line=1),
        Chunk(id="2", content="def logout(): pass", file_path="auth.py",
              chunk_type=ChunkType.FUNCTION, start_line=5, end_line=5),
        Chunk(id="3", content="class User", file_path="user.py",
              chunk_type=ChunkType.CLASS, start_line=1, end_line=3),
    ]

    retriever = BM25Retriever(".")
    retriever.index(chunks)

    # Test file filter
    query = RetrievalQuery(
        text="def login",
        file_filter=["auth.py"],
        top_k=10
    )
    results = retriever.retrieve(query)
    print(f"File filter test: {len(results)} results from auth.py")
    assert all(r.chunk.file_path == "auth.py" for r in results)

    # Test chunk type filter
    query = RetrievalQuery(
        text="class",
        chunk_type_filter=[ChunkType.CLASS],
        top_k=10
    )
    results = retriever.retrieve(query)
    print(f"Type filter test: {len(results)} class results")
    assert all(r.chunk.chunk_type == ChunkType.CLASS for r in results)

    print("Query filtering test passed!")


if __name__ == "__main__":
    print("Retriever Module Test Suite")
    print("=" * 50)

    test_chunk_types()
    test_bm25_retriever()
    test_code_chunker()
    test_query_filtering()

    print("\n" + "=" * 50)
    print("All tests passed!")