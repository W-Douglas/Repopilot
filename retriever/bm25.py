"""
BM25 Keyword Search Retriever

BM25 (Best Matching 25) is a classic probabilistic ranking function
used for information retrieval. It handles term frequency saturation
and document length normalization well.

Based on the original BM25 formula:
- tf(t, d) / (tf(t, d) + k1 * (1 - b + b * |d|/avgdl))

Where:
- tf = term frequency in document
- k1 = term frequency saturation parameter (default 1.5)
- b = document length normalization (default 0.75)
- |d| = document length
- avgdl = average document length
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseRetriever, Chunk, RetrievalQuery, RetrievalResult, ChunkType


@dataclass
class BM25Config:
    """Configuration for BM25 ranking."""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75   # Document length normalization
    min_term_length: int = 2
    stopwords: Optional[List[str]] = None


class BM25Retriever(BaseRetriever):
    """
    BM25-based keyword search retriever.

    Supports:
    - Configurable BM25 parameters
    - Custom tokenization
    - IDF weighting
    - File/type filtering
    """

    name = "bm25"
    description = "BM25 keyword search with configurable ranking"

    def __init__(self, root: str, config: Optional[BM25Config] = None):
        super().__init__(root)
        self.config = config or BM25Config()

        # Index data structures
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: Counter = Counter()  # term -> doc frequency
        self._term_doc_ids: dict = defaultdict(set)  # term -> set of doc ids
        self._doc_term_freqs: List[Counter] = []  # doc_id -> {term: freq}
        self._tokenized_docs: List[List[str]] = []

        # Default Python stopwords
        stopwords = self.config.stopwords or [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            'def', 'class', 'return', 'if', 'else', 'elif', 'while', 'for',
            'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'raise',
        ]
        self._stopwords = set(stopwords)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Lowercase
        text = text.lower()

        # Split on non-alphanumeric (keep underscores in identifiers)
        tokens = re.findall(r'[a-z0-9_]+', text)

        # Filter short tokens and stopwords
        tokens = [
            t for t in tokens
            if len(t) >= self.config.min_term_length
            and t not in self._stopwords
        ]

        return tokens

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query string."""
        return self._tokenize(query)

    def _calculate_idf(self, term: str, n_docs: int) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.

        IDF = log((N - n + 0.5) / (n + 0.5) + 1)

        Smooth variant to handle unseen terms.
        """
        df = self._doc_freqs.get(term, 0)
        # Smoothed IDF formula
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        return max(idf, 0)  # Ensure non-negative

    def _score_bm25(
        self,
        query_terms: List[str],
        doc_idx: int,
        doc_len: int,
        n_docs: int,
    ) -> float:
        """
        Calculate BM25 score for a document.

        BM25 formula:
        sum over t in query:
            IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
        """
        score = 0.0
        term_freqs = self._doc_term_freqs[doc_idx]

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._calculate_idf(term, n_docs)

            # BM25 scoring
            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * (
                1 - self.config.b + self.config.b * doc_len / max(self._avg_doc_length, 1)
            )

            score += idf * (numerator / denominator)

        return score

    def index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._doc_lengths.clear()
        self._doc_term_freqs.clear()
        self._tokenized_docs.clear()
        self._doc_freqs.clear()
        self._term_doc_ids.clear()

        total_len = 0

        for chunk in chunks:
            # Tokenize content
            tokens = self._tokenize(chunk.content)
            self._tokenized_docs.append(tokens)

            # Calculate document length
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            total_len += doc_len

            # Count term frequencies
            term_freq = Counter(tokens)
            self._doc_term_freqs.append(term_freq)

            # Update document frequencies
            for term in set(tokens):
                self._doc_freqs[term] += 1
                self._term_doc_ids[term].add(len(self._doc_term_freqs) - 1)

        # Calculate average document length
        self._avg_doc_length = total_len / max(len(chunks), 1)

        self._indexed = True

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve chunks using BM25 keyword search.

        Args:
            query: RetrievalQuery with text, filters, and top_k

        Returns:
            List of RetrievalResult sorted by score
        """
        if not self._indexed:
            return []

        query_terms = self._tokenize_query(query.text)
        if not query_terms:
            return []

        n_docs = len(self._chunks)
        scores: List[Tuple[int, float]] = []

        for doc_idx in range(n_docs):
            # Apply file filter
            if query.file_filter:
                chunk = self._chunks[doc_idx]
                if not any(chunk.file_path.startswith(f) for f in query.file_filter):
                    continue

            # Apply chunk type filter
            if query.chunk_type_filter:
                chunk = self._chunks[doc_idx]
                if chunk.chunk_type not in query.chunk_type_filter:
                    continue

            # Calculate BM25 score
            score = self._score_bm25(
                query_terms,
                doc_idx,
                self._doc_lengths[doc_idx],
                n_docs,
            )

            # Apply minimum score threshold
            if score >= query.min_score:
                scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Take top_k
        results = []
        for rank, (doc_idx, score) in enumerate(scores[:query.top_k]):
            chunk = self._chunks[doc_idx]
            result = RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                query=query.text,
                method="bm25",
            )
            results.append(result)

        return results

    def search_terms(self, term: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents containing a term.

        Useful for autocomplete and debugging.
        """
        term = term.lower()
        if term not in self._term_doc_ids:
            return []

        doc_ids = self._term_doc_ids[term]
        n_docs = len(self._chunks)

        results = []
        for doc_idx in doc_ids:
            score = self._score_bm25(
                [term],
                doc_idx,
                self._doc_lengths[doc_idx],
                n_docs,
            )
            chunk = self._chunks[doc_idx]
            results.append((chunk.file_path, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term."""
        return self._doc_freqs.get(term.lower(), 0)

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "doc_count": len(self._chunks),
            "vocab_size": len(self._doc_freqs),
            "avg_doc_length": self._avg_doc_length,
            "indexed": self._indexed,
        }


# Convenience function for quick search
def quick_search(
    chunks: List[Chunk],
    query: str,
    top_k: int = 10,
    config: Optional[BM25Config] = None,
) -> List[RetrievalResult]:
    """
    Quick BM25 search without persistent index.

    Useful for one-off searches.
    """
    retriever = BM25Retriever(".", config)
    retriever.index(chunks)
    return retriever.retrieve(RetrievalQuery.from_text(query, top_k))