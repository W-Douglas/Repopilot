"""
Retriever Module - Code RAG with BM25 + Embedding + Rerank

This module provides retrieval capabilities for code context:
- BM25 keyword search
- Embedding-based semantic search
- Cross-encoder reranking
- Hybrid retrieval combining all methods
"""

from .base import (
    BaseRetriever,
    RetrievalResult,
    RetrievalQuery,
    ChunkType,
    Chunk,
)
from .bm25 import BM25Retriever, BM25Config
from .embedding import EmbeddingRetriever, EmbeddingConfig
from .rerank import Reranker, CrossEncoderReranker, RerankConfig, create_reranker
from .chunker import CodeChunker, ChunkerConfig
from .hybrid import HybridRetriever, HybridConfig, RetrievalStats

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "RetrievalQuery",
    "ChunkType",
    "Chunk",
    "BM25Retriever",
    "BM25Config",
    "EmbeddingRetriever",
    "EmbeddingConfig",
    "Reranker",
    "CrossEncoderReranker",
    "RerankConfig",
    "create_reranker",
    "CodeChunker",
    "ChunkerConfig",
    "HybridRetriever",
    "HybridConfig",
    "RetrievalStats",
]