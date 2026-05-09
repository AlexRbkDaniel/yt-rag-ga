import pytest
from unittest.mock import MagicMock

from src.rag.retrieval.retriever import Retriever


def _make_store(results: list[tuple[str, float]]) -> MagicMock:
    """Build a mock FAISS store that returns the given (text, score) pairs."""
    store = MagicMock()
    store.similarity_search_with_relevance_scores.return_value = [
        (_make_doc(text), score) for text, score in results
    ]
    return store


def _make_doc(text: str) -> MagicMock:
    doc = MagicMock()
    doc.page_content = text
    return doc


class TestRetrieverInit:
    def test_default_config(self):
        store = MagicMock()
        retriever = Retriever(store)
        assert retriever._top_k == 4
        assert retriever._score_threshold == 0.0

    def test_custom_config(self):
        store = MagicMock()
        retriever = Retriever(store, top_k=2, score_threshold=0.5)
        assert retriever._top_k == 2
        assert retriever._score_threshold == 0.5

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            Retriever(MagicMock(), top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            Retriever(MagicMock(), top_k=-1)

    def test_score_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="score_threshold must be non-negative"):
            Retriever(MagicMock(), score_threshold=-0.1)

    def test_score_threshold_boundary_zero_valid(self):
        retriever = Retriever(MagicMock(), score_threshold=0.0)
        assert retriever._score_threshold == 0.0

    def test_score_threshold_boundary_one_valid(self):
        retriever = Retriever(MagicMock(), score_threshold=1.0)
        assert retriever._score_threshold == 1.0

    def test_score_threshold_above_one_valid(self):
        retriever = Retriever(MagicMock(), score_threshold=1.5)
        assert retriever._score_threshold == 1.5


class TestRetrieverRetrieve:
    def test_empty_query_raises(self):
        retriever = Retriever(MagicMock())
        with pytest.raises(ValueError, match="query cannot be empty"):
            retriever.retrieve("")

    def test_whitespace_query_raises(self):
        retriever = Retriever(MagicMock())
        with pytest.raises(ValueError, match="query cannot be empty"):
            retriever.retrieve("   ")

    def test_returns_chunk_texts(self):
        store = _make_store([("chunk one", 0.9), ("chunk two", 0.8)])
        retriever = Retriever(store)
        result = retriever.retrieve("some question")
        assert result == ["chunk one", "chunk two"]

    def test_passes_top_k_to_store(self):
        store = _make_store([])
        retriever = Retriever(store, top_k=3)
        retriever.retrieve("question")
        store.similarity_search_with_relevance_scores.assert_called_once_with(
            "question", k=3
        )

    def test_chunks_below_threshold_are_filtered(self):
        store = _make_store([("relevant", 0.8), ("irrelevant", 0.3)])
        retriever = Retriever(store, score_threshold=0.5)
        result = retriever.retrieve("question")
        assert result == ["relevant"]

    def test_all_chunks_below_threshold_returns_empty(self):
        store = _make_store([("low", 0.2), ("also low", 0.1)])
        retriever = Retriever(store, score_threshold=0.5)
        result = retriever.retrieve("question")
        assert result == []

    def test_all_chunks_above_threshold_returned(self):
        store = _make_store([("a", 0.9), ("b", 0.8), ("c", 0.7)])
        retriever = Retriever(store, score_threshold=0.5)
        result = retriever.retrieve("question")
        assert len(result) == 3

    def test_zero_threshold_returns_all_chunks(self):
        store = _make_store([("a", 0.1), ("b", 0.05)])
        retriever = Retriever(store, score_threshold=0.0)
        result = retriever.retrieve("question")
        assert result == ["a", "b"]

    def test_empty_store_results_returns_empty(self):
        store = _make_store([])
        retriever = Retriever(store)
        result = retriever.retrieve("question")
        assert result == []
