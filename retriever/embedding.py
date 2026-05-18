"""
Embedding-based Semantic Retrieval

This module provides embedding-based retrieval using sentence transformers
or other embedding models. It supports:
- Multiple embedding backends (sentence-transformers, OpenAI, etc.)
- Caching of embeddings
- Efficient similarity search via FAISS or simple numpy
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

from .base import BaseRetriever, Chunk, RetrievalQuery, RetrievalResult, ChunkType


@dataclass
class EmbeddingConfig:
    """Configuration for embedding retrieval."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda"
    batch_size: int = 32
    cache_dir: Optional[str] = None
    embedding_dim: int = 384  # for all-MiniLM-L6-v2
    normalize: bool = True  # Normalize embeddings for cosine similarity


class EmbeddingRetriever(BaseRetriever):
    """
    Embedding-based semantic retriever.

    Supports:
    - Sentence transformers (default)
    - OpenAI embeddings (optional)
    - FAISS index for efficient search
    - Embedding caching
    """

    name = "embedding"
    description = "Semantic embedding-based retrieval"

    def __init__(
        self,
        root: str,
        config: Optional[EmbeddingConfig] = None,
        use_faiss: bool = True,
    ):
        super().__init__(root)
        self.config = config or EmbeddingConfig()
        self.use_faiss = use_faiss and _FAISS_AVAILABLE

        # Embeddings storage
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

        # Embedding model (lazy loaded)
        self._model = None
        self._model_loaded = False

        # Cache path
        self._cache_dir = Path(root) / ".retriever_cache" / "embeddings"
        if config and config.cache_dir:
            self._cache_dir = Path(config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Lazy load embedding model."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=str(self._cache_dir.parent / "models"),
            )
            self._model_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts."""
        text_hash = hashlib.md5("||".join(texts).encode()).hexdigest()
        return text_hash

    def _get_cached_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Check cache for embeddings."""
        cache_key = self._get_cache_key(texts)
        cache_file = self._cache_dir / f"{cache_key}.npy"

        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                return None
        return None

    def _cache_embeddings(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Cache embeddings to disk."""
        cache_key = self._get_cache_key(texts)
        cache_file = self._cache_dir / f"{cache_key}.npy"

        try:
            np.save(cache_file, embeddings)
        except Exception:
            pass  # Ignore cache write errors

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()

        # Check cache
        cached = self._get_cached_embeddings(texts)
        if cached is not None:
            return cached

        # Encode with model
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize,
        )

        # Cache results
        self._cache_embeddings(texts, embeddings)

        return embeddings

    def index(self, chunks: List[Chunk]) -> None:
        """Build embedding index from chunks."""
        self._chunks = chunks

        if not chunks:
            self._embeddings = None
            self._indexed = True
            return

        # Prepare texts for encoding
        texts = [self._prepare_text(chunk) for chunk in chunks]

        # Encode all chunks
        embeddings = self._encode(texts)
        self._embeddings = embeddings

        # Build FAISS index if enabled
        if self.use_faiss and _FAISS_AVAILABLE:
            import faiss

            dim = embeddings.shape[1]
            if self.config.normalize:
                # Use inner product for normalized vectors (cosine similarity)
                self._faiss_index = faiss.IndexFlatIP(dim)
            else:
                self._faiss_index = faiss.IndexFlatL2(dim)

            # Normalize for cosine similarity if not already
            if not self.config.normalize:
                faiss.normalize_L2(embeddings)

            self._faiss_index.add(embeddings.astype(np.float32))

        self._indexed = True

    def _prepare_text(self, chunk: Chunk) -> str:
        """Prepare text for embedding."""
        parts = []

        # Add file path context
        parts.append(f"file: {chunk.file_path}")

        # Add symbols
        if chunk.symbols:
            parts.append(f"symbols: {', '.join(chunk.symbols)}")

        # Add docstring
        if chunk.docstring:
            parts.append(chunk.docstring)

        # Add content
        parts.append(chunk.content)

        return " | ".join(parts)

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve chunks using embedding similarity.

        Args:
            query: RetrievalQuery with text, filters, and top_k

        Returns:
            List of RetrievalResult sorted by score
        """
        if not self._indexed or self._embeddings is None:
            return []

        # Encode query
        query_embedding = self._encode([query.text])[0]

        # Search
        if self.use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_search(query_embedding, query.top_k)
        else:
            scores, indices = self._numpy_search(query_embedding, query.top_k)

        # Build results with filtering
        results = []
        for i, (score, doc_idx) in enumerate(zip(scores, indices)):
            chunk = self._chunks[doc_idx]

            # Apply filters
            if query.file_filter:
                if not any(chunk.file_path.startswith(f) for f in query.file_filter):
                    continue

            if query.chunk_type_filter:
                if chunk.chunk_type not in query.chunk_type_filter:
                    continue

            if score < query.min_score:
                continue

            result = RetrievalResult(
                chunk=chunk,
                score=float(score),
                rank=len(results) + 1,
                query=query.text,
                method="embedding",
            )
            results.append(result)

            if len(results) >= query.top_k:
                break

        return results

    def _faiss_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        import faiss

        query = query_embedding.reshape(1, -1).astype(np.float32)
        if self.config.normalize:
            faiss.normalize_L2(query)

        scores, indices = self._faiss_index.search(query, k)
        return scores[0], indices[0]

    def _numpy_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback numpy-based search."""
        embeddings = self._embeddings

        # Compute similarities
        similarities = np.dot(embeddings, query_embedding)

        # Get top k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        return similarities[top_indices], top_indices

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        return self._encode(texts)

    def get_embedding(self, chunk: Chunk) -> Optional[np.ndarray]:
        """Get embedding for a specific chunk."""
        if not self._indexed:
            return None

        try:
            idx = self._chunks.index(chunk)
            return self._embeddings[idx]
        except ValueError:
            return None

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "doc_count": len(self._chunks),
            "embedding_dim": self.config.embedding_dim,
            "model": self.config.model_name,
            "use_faiss": self.use_faiss and self._faiss_index is not None,
            "indexed": self._indexed,
        }


# Try to import FAISS
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    # Will use numpy fallback


# Convenience function for quick semantic search
def quick_semantic_search(
    chunks: List[Chunk],
    query: str,
    top_k: int = 10,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[RetrievalResult]:
    """
    Quick semantic search without persistent index.
    """
    config = EmbeddingConfig(model_name=model_name)
    retriever = EmbeddingRetriever(".", config, use_faiss=False)
    retriever.index(chunks)
    return retriever.retrieve(RetrievalQuery.from_text(query, top_k))