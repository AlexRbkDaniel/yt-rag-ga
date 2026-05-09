from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

LOG = logging.getLogger(__name__)

_DEFAULT_TOP_K = 4
_DEFAULT_SCORE_THRESHOLD = 0.0


class Retriever:
    """Runs similarity search against a FAISS vector store."""

    def __init__(
        self,
        store: FAISS,
        top_k: int = _DEFAULT_TOP_K,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    ):
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if score_threshold < 0.0:
            raise ValueError(
                f"score_threshold must be non-negative, got {score_threshold}"
            )

        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

        LOG.debug(
            "Retriever configured: top_k=%d, score_threshold=%.2f",
            top_k,
            score_threshold,
        )

    def retrieve(self, query: str) -> list[str]:
        """Returns the top-k most relevant chunks for the given query."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        LOG.debug("Retrieving top-%d chunks for query: %s", self._top_k, query)

        results = self._store.similarity_search_with_relevance_scores(
            query, k=self._top_k
        )

        chunks = [
            doc.page_content for doc, score in results if score >= self._score_threshold
        ]

        LOG.debug("Retrieved %d chunks (after score filtering)", len(chunks))
        return chunks
