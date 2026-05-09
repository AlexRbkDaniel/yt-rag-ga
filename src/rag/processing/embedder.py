from __future__ import annotations

import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr

LOG = logging.getLogger(__name__)

_DEFAULT_MODEL = "voyage-3-lite"
_DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Converts text chunks into vector embeddings and stores them in a FAISS index."""

    def __init__(
        self, model: str = _DEFAULT_MODEL, batch_size: int = _DEFAULT_BATCH_SIZE
    ):
        if not model.strip():
            raise ValueError("model cannot be empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self._model = model
        self._batch_size = batch_size
        self._embeddings: VoyageAIEmbeddings | None = None

        LOG.debug("Embedder configured: model=%s, batch_size=%d", model, batch_size)

    def _get_embeddings(self) -> VoyageAIEmbeddings:
        if self._embeddings is not None:
            return self._embeddings

        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is not set. Add it to your .env file.")

        embeddings = VoyageAIEmbeddings(
            model=self._model,
            batch_size=self._batch_size,
            api_key=SecretStr(api_key),
        )
        self._embeddings = embeddings
        LOG.debug("VoyageAIEmbeddings initialised: model=%s", self._model)
        return embeddings

    def embed(self, chunks: list[str]) -> FAISS:
        """Embeds the given text chunks and returns a searchable FAISS vector store."""
        if not chunks:
            raise ValueError("chunks cannot be empty — nothing to embed")

        LOG.info("Embedding %d chunks with model=%s", len(chunks), self._model)
        embeddings = self._get_embeddings()
        store = FAISS.from_texts(chunks, embeddings)
        LOG.info("FAISS index built: %d vectors", store.index.ntotal)
        return store
