"""
Hybrid Retriever - combines BM25 + Embedding + Rerank

This module provides a unified retrieval pipeline:
1. BM25: Fast keyword matching
2. Embedding: Semantic similarity search
3. RRF Fusion: Combine results from multiple methods
4. Cross-encoder: Final reranking for precision

Strategy:
- Run BM25 and embedding retrieval in parallel
- Fuse results using Reciprocal Rank Fusion
- Apply cross-encoder reranking for top candidates
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import time

from .base import BaseRetriever, Chunk, RetrievalQuery, RetrievalResult, ChunkType
from .bm25 import BM25Retriever, BM25Config
from .embedding import EmbeddingRetriever, EmbeddingConfig
from .rerank import (
    Reranker,
    CrossEncoderReranker,
    ReciprocalRankReranker,
    SimpleReranker,
    create_reranker,
)
from .chunker import CodeChunker, ChunkerConfig


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # BM25 settings
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_faiss: bool = True

    # Reranking settings
    reranker_type: str = "cross_encoder"  # "cross_encoder", "simple", "rrf"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20  # Candidates to rerank

    # Fusion settings
    fusion_k: int = 60  # RRF k parameter
    bm25_weight: float = 0.4
    embedding_weight: float = 0.6

    # Retrieval settings
    initial_top_k: int = 50  # Results from each retriever before fusion
    final_top_k: int = 10


@dataclass
class RetrievalStats:
    """Statistics for a retrieval operation."""
    bm25_time: float = 0.0
    embedding_time: float = 0.0
    fusion_time: float = 0.0
    rerank_time: float = 0.0
    total_time: float = 0.0
    bm25_results: int = 0
    embedding_results: int = 0
    fused_results: int = 0
    final_results: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bm25_time": round(self.bm25_time, 3),
            "embedding_time": round(self.embedding_time, 3),
            "fusion_time": round(self.fusion_time, 3),
            "rerank_time": round(self.rerank_time, 3),
            "total_time": round(self.total_time, 4),
            "bm25_results": self.bm25_results,
            "embedding_results": self.embedding_results,
            "fused_results": self.fused_results,
            "final_results": self.final_results,
        }


class HybridRetriever:
    """
    Hybrid retrieval combining multiple methods.

    Pipeline:
    1. Build BM25 and embedding indexes
    2. Parallel retrieval from both
    3. RRF fusion of results
    4. Cross-encoder reranking
    5. Return top-k results with scores
    """

    name = "hybrid"
    description = "BM25 + Embedding + Rerank hybrid retrieval"

    def __init__(
        self,
        root: str,
        config: Optional[HybridConfig] = None,
    ):
        self.root = Path(root).resolve()
        self.config = config or HybridConfig()

        # Initialize sub-retrievers
        bm25_cfg = BM25Config(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self.bm25_retriever = BM25Retriever(root, bm25_cfg)

        embedding_cfg = EmbeddingConfig(
            model_name=self.config.embedding_model,
            normalize=True,
        )
        self.embedding_retriever = EmbeddingRetriever(
            root,
            embedding_cfg,
            use_faiss=self.config.use_faiss,
        )

        # Initialize reranker
        rerank_cfg = None
        if self.config.reranker_type == "cross_encoder":
            from .rerank import RerankConfig
            rerank_cfg = RerankConfig(model_name=self.config.reranker_model)
        self.reranker = create_reranker(self.config.reranker_type, rerank_cfg)

        # RRF fusion reranker
        self.rrf_fuser = ReciprocalRankReranker()

        # Chunk storage
        self._chunks: List[Chunk] = []
        self._indexed = False

        # Stats
        self.stats = RetrievalStats()

    def index(self, chunks: List[Chunk]) -> None:
        """Build all indexes from chunks."""
        self._chunks = chunks

        if not chunks:
            self._indexed = False
            return

        # Build BM25 index
        self.bm25_retriever.index(chunks)

        # Build embedding index
        self.embedding_retriever.index(chunks)

        self._indexed = True

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to index."""
        self._chunks.extend(chunks)
        self.bm25_retriever.add_chunks(chunks)
        self.embedding_retriever.add_chunks(chunks)
        # Re-index
        self.index(self._chunks)

    def retrieve(
        self,
        query: RetrievalQuery,
        return_stats: bool = False,
    ) -> Tuple[List[RetrievalResult], Optional[RetrievalStats]]:
        """
        Perform hybrid retrieval.

        Args:
            query: Retrieval query
            return_stats: Whether to return retrieval statistics

        Returns:
            Tuple of (results, stats) if return_stats=True, else just results
        """
        start_time = time.time()
        self.stats = RetrievalStats()

        if not self._indexed:
            if return_stats:
                return [], None
            return []

        # Adjust top_k for initial retrieval
        query_bm25 = RetrievalQuery(
            text=query.text,
            file_filter=query.file_filter,
            chunk_type_filter=query.chunk_type_filter,
            top_k=self.config.initial_top_k,
            min_score=query.min_score,
        )

        query_emb = RetrievalQuery(
            text=query.text,
            file_filter=query.file_filter,
            chunk_type_filter=query.chunk_type_filter,
            top_k=self.config.initial_top_k,
            min_score=query.min_score,
        )

        # Run BM25 retrieval
        bm25_start = time.time()
        bm25_results = self.bm25_retriever.retrieve(query_bm25)
        self.stats.bm25_time = time.time() - bm25_start
        self.stats.bm25_results = len(bm25_results)

        # Run embedding retrieval
        emb_start = time.time()
        embedding_results = self.embedding_retriever.retrieve(query_emb)
        self.stats.embedding_time = time.time() - emb_start
        self.stats.embedding_results = len(embedding_results)

        # Fuse results using RRF
        fusion_start = time.time()
        fused_results = self.rrf_fuser.fuse_results(
            [bm25_results, embedding_results],
            weights=[self.config.bm25_weight, self.config.embedding_weight],
        )
        self.stats.fusion_time = time.time() - fusion_start
        self.stats.fused_results = len(fused_results)

        # Take candidates for reranking
        candidates = fused_results[:self.config.rerank_top_k]

        # Apply reranking
        rerank_start = time.time()
        if candidates and self.config.reranker_type == "cross_encoder":
            final_results = self.reranker.rerank(
                candidates,
                query.text,
                top_k=query.top_k,
            )
        else:
            # Skip cross-encoder, use fused results
            final_results = candidates[:query.top_k]
            for i, result in enumerate(final_results):
                result.rank = i + 1

        self.stats.rerank_time = time.time() - rerank_start

        # Update query and method on results
        for result in final_results:
            result.query = query.text
            result.method = "hybrid"

        self.stats.total_time = time.time() - start_time
        self.stats.final_results = len(final_results)

        if return_stats:
            return final_results, self.stats
        return final_results

    def retrieve_bm25_only(
        self,
        query: RetrievalQuery,
    ) -> List[RetrievalResult]:
        """Retrieve using BM25 only (fast keyword search)."""
        return self.bm25_retriever.retrieve(query)

    def retrieve_embedding_only(
        self,
        query: RetrievalQuery,
    ) -> List[RetrievalResult]:
        """Retrieve using embedding only (semantic search)."""
        return self.embedding_retriever.retrieve(query)

    def clear(self) -> None:
        """Clear all indexes."""
        self._chunks.clear()
        self.bm25_retriever.clear()
        self.embedding_retriever.clear()
        self._indexed = False

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def is_indexed(self) -> bool:
        """Whether index is built."""
        return self._indexed

    def save_index(self, path: str) -> None:
        """Save indexes to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save chunks metadata
        chunks_data = [c.to_dict() for c in self._chunks]
        import json
        with open(path, 'w') as f:
            json.dump({"chunks": chunks_data}, f)

    def load_index(self, path: str) -> None:
        """Load indexes from disk."""
        path = Path(path)
        if not path.exists():
            return

        import json
        with open(path, 'r') as f:
            data = json.load(f)

        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        self.index(chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        return {
            "hybrid_config": {
                "bm25_weight": self.config.bm25_weight,
                "embedding_weight": self.config.embedding_weight,
                "reranker": self.config.reranker_type,
            },
            "bm25_stats": self.bm25_retriever.get_stats(),
            "embedding_stats": self.embedding_retriever.get_stats(),
            "retrieval_stats": self.stats.to_dict(),
        }


def quick_retrieve(
    chunks: List[Chunk],
    query: str,
    top_k: int = 10,
    method: str = "hybrid",
) -> List[RetrievalResult]:
    """
    Quick retrieval with defaults.

    Args:
        chunks: List of code chunks
        query: Query string
        top_k: Number of results
        method: "hybrid", "bm25", or "embedding"

    Returns:
        Retrieval results
    """
    config = HybridConfig()
    retriever = HybridRetriever(".", config)
    retriever.index(chunks)

    retrieval_query = RetrievalQuery.from_text(query, top_k)

    if method == "bm25":
        return retriever.retrieve_bm25_only(retrieval_query)
    elif method == "embedding":
        return retriever.retrieve_embedding_only(retrieval_query)
    else:
        return retriever.retrieve(retrieval_query)