"""
Retriever Module CLI Entry Point

Usage:
    python -m retriever.main --query "your query" --top-k 10
    python -m retriever.main --index ./src --query "find auth" --method hybrid
"""

import argparse
import json
import sys
from pathlib import Path

from retriever import (
    HybridRetriever,
    BM25Retriever,
    CodeChunker,
    Chunk,
    RetrievalQuery,
    ChunkType,
)


def build_index(root: str, output_path: str = None) -> tuple:
    """Build index from source files."""
    print(f"Building index for: {root}")

    chunker = CodeChunker(root)
    chunks = chunker.chunk_directory(root)

    print(f"Generated {len(chunks)} chunks")

    if not chunks:
        print("No chunks generated!")
        return [], None

    # Save chunks if output specified
    if output_path:
        chunks_data = [c.to_dict() for c in chunks]
        with open(output_path, 'w') as f:
            json.dump({"chunks": chunks_data}, f, indent=2)
        print(f"Saved chunks to: {output_path}")

    # Build retriever
    retriever = HybridRetriever(root)
    retriever.index(chunks)

    return chunks, retriever


def search(retriever, query: str, top_k: int, method: str):
    """Perform search."""
    retrieval_query = RetrievalQuery.from_text(query, top_k)

    if method == "bm25":
        results = retriever.retrieve_bm25_only(retrieval_query)
    elif method == "embedding":
        results = retriever.retrieve_embedding_only(retrieval_query)
    else:
        results, stats = retriever.retrieve(retrieval_query, return_stats=True)
        print(f"\nRetrieval stats: {stats.to_dict()}")

    print(f"\nFound {len(results)} results for: '{query}'")
    print("-" * 60)

    for r in results:
        print(f"[{r.rank}] {r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line}")
        print(f"    Type: {r.chunk.chunk_type.value} | Score: {r.score:.3f}")
        print(f"    Symbols: {', '.join(r.chunk.symbols)}")
        if r.chunk.docstring:
            print(f"    Doc: {r.chunk.docstring[:80]}...")
        content_preview = r.chunk.content[:100].replace('\n', ' ')
        print(f"    Code: {content_preview}...")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Code RAG Retriever CLI")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--index", help="Index directory or save path")
    parser.add_argument("--chunks", help="Load chunks from JSON file")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--method", choices=["bm25", "embedding", "hybrid"],
                        default="hybrid", help="Retrieval method")
    parser.add_argument("--stats", action="store_true", help="Show retrieval stats")

    args = parser.parse_args()

    # Build or load index
    if args.chunks:
        # Load chunks from file
        with open(args.chunks, 'r') as f:
            data = json.load(f)
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        print(f"Loaded {len(chunks)} chunks from: {args.chunks}")

        retriever = HybridRetriever(args.root)
        retriever.index(chunks)
    elif args.index:
        # Index directory
        chunks, retriever = build_index(args.index)
    else:
        # Try to find existing chunks in root
        chunker = CodeChunker(args.root)
        chunks = chunker.chunk_directory(args.root)
        print(f"Chunked {len(chunks)} from: {args.root}")

        retriever = HybridRetriever(args.root)
        retriever.index(chunks)

    # Perform search
    if args.query:
        results = search(retriever, args.query, args.top_k, args.method)

        if args.stats:
            print("\nRetriever stats:")
            print(json.dumps(retriever.get_stats(), indent=2))
    else:
        # Interactive mode
        print("\nEnter queries (Ctrl+C to exit):")
        while True:
            try:
                query = input("\nQuery> ").strip()
                if query:
                    search(retriever, query, args.top_k, args.method)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()