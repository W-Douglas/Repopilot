"""
Test and example usage for the Retriever module.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import (
    HybridRetriever,
    BM25Retriever,
    EmbeddingRetriever,
    CodeChunker,
    Chunk,
    ChunkType,
    RetrievalQuery,
    RetrievalStats,
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
        content_preview = chunk.content[:80].replace('\n', ' ')
        print(f"Content: {content_preview}...")


def test_hybrid_retriever():
    """Test hybrid retriever."""
    print("\n=== Testing Hybrid Retriever ===")

    chunks = [
        Chunk(
            id="1",
            content="def authenticate_user(username, password): check credentials and return user object",
            file_path="auth.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=5,
            symbols=["authenticate_user"],
            docstring="Authenticate a user with username and password",
        ),
        Chunk(
            id="2",
            content="class UserSession: manages user sessions and tokens",
            file_path="session.py",
            chunk_type=ChunkType.CLASS,
            start_line=1,
            end_line=10,
            symbols=["UserSession"],
            docstring="Manages user sessions and authentication tokens",
        ),
        Chunk(
            id="3",
            content="def parse_json(data): parse JSON string and return dict",
            file_path="parser.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=5,
            symbols=["parse_json"],
        ),
        Chunk(
            id="4",
            content="def validate_token(token): verify JWT token is valid",
            file_path="auth.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=20,
            end_line=25,
            symbols=["validate_token"],
        ),
        Chunk(
            id="5",
            content="def create_user(name, email, password): create new user account",
            file_path="user.py",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=10,
            symbols=["create_user"],
        ),
    ]

    # Initialize hybrid retriever
    retriever = HybridRetriever(".")
    retriever.index(chunks)

    # Test query
    query = "authentication and user validation"
    print(f"\nQuery: '{query}'")

    # BM25 only
    bm25_results = retriever.retrieve_bm25_only(RetrievalQuery.from_text(query, top_k=5))
    print(f"\nBM25 results ({len(bm25_results)}):")
    for r in bm25_results:
        print(f"  - {r.chunk.file_path}:{r.chunk.symbols[0] if r.chunk.symbols else '?'} (score: {r.score:.3f})")

    # Embedding only
    try:
        emb_results = retriever.retrieve_embedding_only(RetrievalQuery.from_text(query, top_k=5))
        print(f"\nEmbedding results ({len(emb_results)}):")
        for r in emb_results:
            print(f"  - {r.chunk.file_path}:{r.chunk.symbols[0] if r.chunk.symbols else '?'} (score: {r.score:.3f})")
    except Exception as e:
        print(f"\nEmbedding not available: {e}")

    # Hybrid retrieval
    results, stats = retriever.retrieve(
        RetrievalQuery.from_text(query, top_k=5),
        return_stats=True
    )
    print(f"\nHybrid results ({len(results)}):")
    for r in results:
        print(f"  - [{r.rank}] {r.chunk.file_path}:{r.chunk.symbols[0] if r.chunk.symbols else '?'}")
        print(f"    Score: {r.score:.3f} | Method: {r.method}")
        print(f"    Doc: {r.chunk.docstring[:50] if r.chunk.docstring else 'N/A'}...")

    print(f"\nRetrieval stats:")
    print(f"  BM25 time: {stats.bm25_time:.3f}s, results: {stats.bm25_results}")
    print(f"  Embedding time: {stats.embedding_time:.3f}s, results: {stats.embedding_results}")
    print(f"  Fusion time: {stats.fusion_time:.3f}s, results: {stats.fused_results}")
    print(f"  Rerank time: {stats.rerank_time:.3f}s, results: {stats.final_results}")
    print(f"  Total time: {stats.total_time:.4f}s")


def test_chunk_types():
    """Test all chunk types."""
    print("\n=== Testing Chunk Types ===")

    for ct in ChunkType:
        print(f"  - {ct.value}")


if __name__ == "__main__":
    print("Retriever Module Test Suite")
    print("=" * 50)

    test_chunk_types()
    test_bm25_retriever()
    test_code_chunker()
    test_hybrid_retriever()

    print("\n" + "=" * 50)
    print("All tests completed!")