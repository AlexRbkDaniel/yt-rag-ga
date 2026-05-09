import pytest
from unittest.mock import patch, MagicMock

from src.rag.processing.embedder import Embedder


class TestEmbedderInit:
    def test_default_config(self):
        embedder = Embedder()
        assert embedder._model == "voyage-3-lite"
        assert embedder._batch_size == 32

    def test_custom_config(self):
        embedder = Embedder(model="voyage-3-lite", batch_size=16)
        assert embedder._model == "voyage-3-lite"
        assert embedder._batch_size == 16

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model cannot be empty"):
            Embedder(model="")

    def test_whitespace_model_raises(self):
        with pytest.raises(ValueError, match="model cannot be empty"):
            Embedder(model="   ")

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            Embedder(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            Embedder(batch_size=-1)

    def test_embeddings_not_cached_at_init(self):
        embedder = Embedder()
        assert embedder._embeddings is None


class TestEmbedderGetEmbeddings:
    def test_missing_api_key_raises(self):
        embedder = Embedder()
        with patch("src.rag.processing.embedder.os.getenv", return_value=None):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY is not set"):
                embedder._get_embeddings()

    def test_empty_api_key_raises(self):
        embedder = Embedder()
        with patch("src.rag.processing.embedder.os.getenv", return_value=""):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY is not set"):
                embedder._get_embeddings()

    def test_returns_voyage_embeddings_instance(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
        ):
            result = embedder._get_embeddings()
        assert result is mock_emb

    def test_embeddings_cached_after_first_call(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ) as mock_cls,
        ):
            first = embedder._get_embeddings()
            second = embedder._get_embeddings()

        assert first is second
        mock_cls.assert_called_once()

    def test_cached_instance_set_after_call(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
        ):
            embedder._get_embeddings()

        assert embedder._embeddings is mock_emb


class TestEmbedderEmbed:
    def test_empty_chunks_raises(self):
        embedder = Embedder()
        with pytest.raises(ValueError, match="chunks cannot be empty"):
            embedder.embed([])

    def test_returns_faiss_store(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
            patch(
                "src.rag.processing.embedder.FAISS.from_texts", return_value=mock_store
            ) as mock_faiss,
        ):
            result = embedder.embed(["hello world"])

        assert result is mock_store
        mock_faiss.assert_called_once_with(["hello world"], mock_emb)

    def test_all_chunks_passed_to_faiss(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        mock_store = MagicMock()
        chunks = ["chunk one", "chunk two", "chunk three"]
        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
            patch(
                "src.rag.processing.embedder.FAISS.from_texts", return_value=mock_store
            ) as mock_faiss,
        ):
            embedder.embed(chunks)

        mock_faiss.assert_called_once_with(chunks, mock_emb)

    def test_retries_on_transient_failure_then_succeeds(self):
        embedder = Embedder()
        mock_emb = MagicMock()
        mock_store = MagicMock()
        call_count = {"n": 0}

        def flaky_from_texts(chunks, emb):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient")
            return mock_store

        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
            patch(
                "src.rag.processing.embedder.FAISS.from_texts",
                side_effect=flaky_from_texts,
            ),
            patch("src.rag.processing.embedder.time.sleep"),
        ):
            result = embedder.embed(["chunk"], max_retries=3, retry_delay=0.0)

        assert result is mock_store
        assert call_count["n"] == 3

    def test_all_retries_exhausted_raises_runtime_error(self):
        embedder = Embedder()
        mock_emb = MagicMock()

        with (
            patch("src.rag.processing.embedder.os.getenv", return_value="test-key"),
            patch(
                "src.rag.processing.embedder.VoyageAIEmbeddings", return_value=mock_emb
            ),
            patch(
                "src.rag.processing.embedder.FAISS.from_texts",
                side_effect=ConnectionError("always fails"),
            ),
            patch("src.rag.processing.embedder.time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="Embedding failed after 3 attempts"):
                embedder.embed(["chunk"], max_retries=3, retry_delay=0.0)
