import pytest

from src.rag.models.transcript import TranscriptSegment
from src.rag.processing.chunker import TranscriptChunker


class TestTranscriptChunkerInit:
    def test_default_config(self):
        chunker = TranscriptChunker()
        assert chunker._chunk_size == 1000
        assert chunker._chunk_overlap == 100

    def test_custom_config(self):
        chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
        assert chunker._chunk_size == 500
        assert chunker._chunk_overlap == 50

    def test_splitter_initialised_at_construction(self):
        chunker = TranscriptChunker()
        assert chunker._splitter is not None

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            TranscriptChunker(chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            TranscriptChunker(chunk_size=-1)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TranscriptChunker(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(
            ValueError, match="chunk_overlap .* must be less than chunk_size"
        ):
            TranscriptChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(
            ValueError, match="chunk_overlap .* must be less than chunk_size"
        ):
            TranscriptChunker(chunk_size=100, chunk_overlap=200)

    def test_splitter_reused_across_calls(self):
        chunker = TranscriptChunker()
        splitter_before = chunker._splitter
        segments = [TranscriptSegment(text="Hello world", start=0.0)]
        chunker.chunk(segments)
        assert chunker._splitter is splitter_before


class TestTranscriptChunkerChunk:
    def test_empty_segments_returns_empty_list(self):
        chunker = TranscriptChunker()
        assert chunker.chunk([]) == []

    def test_single_short_segment_returns_one_chunk(self):
        chunker = TranscriptChunker(chunk_size=1000, chunk_overlap=100)
        segments = [TranscriptSegment(text="Hello world", start=0.0)]
        chunks = chunker.chunk(segments)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        chunker = TranscriptChunker(chunk_size=50, chunk_overlap=10)
        segments = [TranscriptSegment(text="word " * 100, start=0.0)]
        chunks = chunker.chunk(segments)
        assert len(chunks) > 1

    def test_chunk_size_not_exceeded(self):
        chunker = TranscriptChunker(chunk_size=100, chunk_overlap=0)
        segments = [TranscriptSegment(text="word " * 200, start=0.0)]
        chunks = chunker.chunk(segments)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_text_extracted_from_multiple_segments(self):
        chunker = TranscriptChunker()
        segments = [
            TranscriptSegment(text="Hello world", start=0.0),
            TranscriptSegment(text="goodbye world", start=5.0),
        ]
        chunks = chunker.chunk(segments)
        full = " ".join(chunks)
        assert "Hello world" in full
        assert "goodbye world" in full

    def test_start_times_not_included_in_chunks(self):
        chunker = TranscriptChunker()
        segments = [TranscriptSegment(text="Hello world", start=123.45)]
        chunks = chunker.chunk(segments)
        full = " ".join(chunks)
        assert "123.45" not in full
        assert "Start:" not in full

    def test_multiple_segments_joined_with_space(self):
        chunker = TranscriptChunker(chunk_size=1000, chunk_overlap=0)
        segments = [
            TranscriptSegment(text="first", start=0.0),
            TranscriptSegment(text="second", start=1.0),
        ]
        chunks = chunker.chunk(segments)
        assert chunks[0] == "first second"
