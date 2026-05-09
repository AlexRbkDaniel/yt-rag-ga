from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from src.rag.models.transcript import TranscriptSegment

LOG = logging.getLogger(__name__)


class TranscriptChunker:
    """Splits transcript segments into overlapping text chunks for embedding and retrieval."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialises the chunker with the given chunk size and overlap.

        Args:
            chunk_size: maximum number of characters per chunk.
            chunk_overlap: number of characters overlapping between consecutive chunks.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # Initialise the splitter once — reused across all chunk() calls
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        LOG.debug("TranscriptChunker initialised with chunk_size=%d, chunk_overlap=%d", chunk_size, chunk_overlap)

    def chunk(self, segments: list[TranscriptSegment]) -> list[str]:
        """Joins transcript segments into plain text and splits into overlapping chunks.

        Args:
            segments: list of TranscriptSegment DTOs from YtLoader.

        Returns:
            List of text chunks ready for embedding.
        """
        if not segments:
            LOG.warning("chunk() called with empty segments list — returning empty result")
            return []

        # Join plain text from all segments — start times are not included in chunks
        plain_text: str = " ".join(segment.text for segment in segments)
        LOG.debug("Chunking transcript of %d characters from %d segments", len(plain_text), len(segments))

        chunks: list[str] = self._splitter.split_text(plain_text)

        if not chunks:
            LOG.warning("Chunking produced no output for transcript of %d characters", len(plain_text))
            return []

        LOG.debug("Produced %d chunks from transcript", len(chunks))
        return chunks