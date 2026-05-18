"""
Reranking Module - Cross-encoder based result refinement

Reranking improves initial retrieval results by:
1. Using a cross-encoder model to score (query, document) pairs
2. Reordering results based on more accurate relevance scores
3. Combining signals from both BM25 and embedding retrieval

Cross-encoder advantages:
- Joint encoding of query and document (better interaction)
- More accurate relevance scoring
- Handles synonyms and paraphrases better
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .base import Chunk, RetrievalResult


@dataclass
class RerankConfig:
    """Configuration for reranking."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 16
    normalize_scores: bool = True


class Reranker(ABC):
    """Abstract base class for rerankers."""

    name: str = "base"

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._model = None
        self._model_loaded = False

    @abstractmethod
    def load(self):
        """Load reranking model."""
        pass

    @abstractmethod
    def score(self, queries: List[str], documents: List[str]) -> List[float]:
        """
        Score (query, document) pairs.

        Args:
            queries: List of query strings
            documents: List of document strings

        Returns:
            List of relevance scores
        """
        pass

    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results.

        Args:
            results: Initial retrieval results
            query: Original query string
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Load model if needed
        if not self._model_loaded:
            self.load()

        # Prepare documents for reranking
        documents = [r.chunk.content for r in results]
        queries = [query] * len(documents)

        # Score pairs
        scores = self.score(queries, documents)

        # Attach scores to results
        for result, score in zip(results, scores):
            result.score = score

        # Sort by rerank score
        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked[:top_k]):
            result.rank = i + 1
            result.method = f"{result.method}+rerank"

        return reranked[:top_k]


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranker.

    Uses a pre-trained cross-encoder model for relevance scoring.
    Models:
    - ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - ms-marco-Multi-MiniLM-L-6-v2 (better for diverse queries)
    - cross-encoder/qna (good for question answering)
    """

    name = "cross_encoder"

    def __init__(self, config: Optional[RerankConfig] = None):
        super().__init__(config)

    def load(self):
        """Load cross-encoder model."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.config.model_name,
                max_length=512,
            )
            self._model_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def score(self, queries: List[str], documents: List[str]) -> List[float]:
        """Score query-document pairs using cross-encoder."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create input pairs
        pairs = list(zip(queries, documents))

        # Get scores
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Convert to list
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        else:
            scores = list(scores)

        # Normalize scores if configured
        if self.config.normalize_scores and scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]

        return scores


class SimpleReranker(Reranker):
    """
    Simple rule-based reranker for when cross-encoder is unavailable.

    Uses heuristics:
    - Exact match bonus
    - Symbol match bonus
    - File path relevance
    """

    name = "simple"

    def __init__(self):
        super().__init__(RerankConfig())

    def load(self):
        """No-op for simple reranker."""
        self._model_loaded = True

    def score(self, queries: List[str], documents: List[str]) -> List[float]:
        """Score using simple heuristics."""
        scores = []

        for query, doc in zip(queries, documents):
            score = 0.0
            query_lower = query.lower()
            doc_lower = doc.lower()

            # Exact match bonus
            if query_lower in doc_lower:
                score += 1.0

            # Symbol matching (function/class names)
            query_words = set(query_lower.split())
            doc_words = set(re.findall(r'\w+', doc_lower))
            overlap = len(query_words & doc_words)
            score += overlap * 0.5

            # First line match bonus
            first_line = doc.split('\n')[0] if doc else ""
            if query_lower in first_line.lower():
                score += 0.5

            scores.append(score)

        return scores


import re


class ReciprocalRankReranker(Reranker):
    """
    Reciprocal Rank Fusion (RRF) for combining multiple retrieval methods.

    RRF formula:
    score(doc) = sum over methods: 1 / (k + rank_in_method)

    Where k is a constant (usually 60).
    """

    name = "rrf"
    k: int = 60

    def __init__(self):
        super().__init__(RerankConfig())

    def load(self):
        """No-op for RRF reranker."""
        self._model_loaded = True

    def score(self, queries: List[str], documents: List[str]) -> List[float]:
        """RRF doesn't use cross-encoder scores."""
        return [1.0] * len(documents)

    def fuse_results(
        self,
        result_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Fuse multiple retrieval result lists using RRF.

        Args:
            result_lists: List of result lists from different retrievers
            weights: Optional weights for each retrieval method

        Returns:
            Fused and ranked results
        """
        if not result_lists:
            return []

        # Default weights (equal)
        if weights is None:
            weights = [1.0] * len(result_lists)
        weights = weights[:len(result_lists)]

        # Collect all unique chunks
        chunk_scores: dict = {}

        for retriever_idx, results in enumerate(result_lists):
            weight = weights[retriever_idx]

            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk.id

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "result": result,
                        "score": 0.0,
                    }

                # RRF formula with weight
                rrf_score = weight / (self.k + rank)
                chunk_scores[chunk_id]["score"] += rrf_score

        # Sort by fused score
        fused = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # Build final results
        final_results = []
        for rank, item in enumerate(fused):
            result = item["result"]
            result.score = item["score"]
            result.rank = rank + 1
            result.method = "rrf_fusion"
            final_results.append(result)

        return final_results


# Try to import sentence-transformers for CrossEncoder
try:
    from sentence_transformers import CrossEncoder as _CT
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


def create_reranker(
    reranker_type: str = "cross_encoder",
    config: Optional[RerankConfig] = None,
) -> Reranker:
    """
    Factory function to create rerankers.

    Args:
        reranker_type: "cross_encoder", "simple", or "rrf"
        config: Optional configuration

    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        if not _CROSS_ENCODER_AVAILABLE:
            print("Warning: cross-encoder not available, using simple reranker")
            return SimpleReranker()
        return CrossEncoderReranker(config)
    elif reranker_type == "simple":
        return SimpleReranker()
    elif reranker_type == "rrf":
        return ReciprocalRankReranker()
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")