from unittest.mock import MagicMock, patch

from src.rag.models.transcript import TranscriptSegment
from src.rag.models.video import VideoData, VideoMetadata
from src.rag.pipeline import summarize_video, answer_question


def _make_video_data(segments: list[TranscriptSegment]) -> VideoData:
    metadata = VideoMetadata(
        video_id="abc123", video_url="https://www.youtube.com/watch?v=abc123"
    )
    return VideoData(metadata=metadata, transcript=segments)


class TestSummarizeVideo:
    def test_empty_url_returns_message(self):
        result = summarize_video("")
        assert "valid YouTube URL" in result

    def test_whitespace_url_returns_message(self):
        result = summarize_video("   ")
        assert "valid YouTube URL" in result

    def test_empty_transcript_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            return_value=_make_video_data([]),
        ):
            result = summarize_video("https://www.youtube.com/watch?v=abc123")
        assert "No transcript available" in result

    def test_returns_summary_string(self):
        segments = [TranscriptSegment(text="Python is great.", start=0.0)]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Python is a great language."
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.create_summary_chain", return_value=mock_chain),
        ):
            result = summarize_video(
                "https://www.youtube.com/watch?v=abc123", llm_loader=mock_loader
            )

        assert result == "Python is a great language."

    def test_transcript_segments_joined_with_space(self):
        segments = [
            TranscriptSegment(text="Hello world.", start=0.0),
            TranscriptSegment(text="Goodbye world.", start=5.0),
        ]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "A summary."
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.create_summary_chain", return_value=mock_chain),
        ):
            summarize_video(
                "https://www.youtube.com/watch?v=abc123", llm_loader=mock_loader
            )

        payload = mock_chain.invoke.call_args[0][0]
        assert payload["transcript"] == "Hello world. Goodbye world."

    def test_chain_not_called_when_no_transcript(self):
        mock_chain = MagicMock()
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data([]),
            ),
            patch("src.rag.pipeline.create_summary_chain", return_value=mock_chain),
        ):
            summarize_video(
                "https://www.youtube.com/watch?v=abc123", llm_loader=mock_loader
            )

        mock_chain.invoke.assert_not_called()

    def test_default_llm_loader_used_when_not_provided(self):
        segments = [TranscriptSegment(text="Some content.", start=0.0)]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Summary."
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.create_summary_chain", return_value=mock_chain),
            patch("src.rag.pipeline._default_llm_loader", mock_loader),
        ):
            summarize_video("https://www.youtube.com/watch?v=abc123")

        mock_loader.load.assert_called_once()

    def test_value_error_from_loader_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            side_effect=ValueError("No transcript"),
        ):
            result = summarize_video("https://www.youtube.com/watch?v=abc123")
        assert "Could not load video" in result

    def test_unexpected_error_from_loader_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            side_effect=RuntimeError("Network error"),
        ):
            result = summarize_video("https://www.youtube.com/watch?v=abc123")
        assert "unexpected error" in result.lower()


class TestAnswerQuestion:
    def test_empty_url_returns_message(self):
        result = answer_question("", "What is this about?")
        assert "valid YouTube URL" in result

    def test_whitespace_url_returns_message(self):
        result = answer_question("   ", "What is this about?")
        assert "valid YouTube URL" in result

    def test_empty_question_returns_message(self):
        result = answer_question("https://www.youtube.com/watch?v=abc123", "")
        assert "valid question" in result

    def test_whitespace_question_returns_message(self):
        result = answer_question("https://www.youtube.com/watch?v=abc123", "   ")
        assert "valid question" in result

    def test_empty_transcript_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            return_value=_make_video_data([]),
        ):
            result = answer_question(
                "https://www.youtube.com/watch?v=abc123", "What is this about?"
            )
        assert "No transcript available" in result

    def test_returns_answer_string(self):
        segments = [TranscriptSegment(text="Python is great.", start=0.0)]
        mock_store = MagicMock()
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.TranscriptChunker") as mock_chunker_cls,
            patch("src.rag.pipeline.Embedder") as mock_embedder_cls,
            patch("src.rag.pipeline.Retriever"),
            patch("src.rag.pipeline.create_qa_chain"),
            patch("src.rag.pipeline.generate_answer", return_value="Python is great."),
        ):
            mock_chunker_cls.return_value.chunk.return_value = ["Python is great."]
            mock_embedder_cls.return_value.embed.return_value = mock_store

            result = answer_question(
                "https://www.youtube.com/watch?v=abc123",
                "What is Python?",
                llm_loader=mock_loader,
            )

        assert result == "Python is great."

    def test_chunker_called_with_transcript(self):
        segments = [TranscriptSegment(text="Some content.", start=0.0)]
        mock_store = MagicMock()
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.TranscriptChunker") as mock_chunker_cls,
            patch("src.rag.pipeline.Embedder") as mock_embedder_cls,
            patch("src.rag.pipeline.Retriever"),
            patch("src.rag.pipeline.create_qa_chain"),
            patch("src.rag.pipeline.generate_answer", return_value="answer"),
        ):
            mock_chunker_cls.return_value.chunk.return_value = ["Some content."]
            mock_embedder_cls.return_value.embed.return_value = mock_store

            answer_question(
                "https://www.youtube.com/watch?v=abc123",
                "question?",
                llm_loader=mock_loader,
            )

        mock_chunker_cls.return_value.chunk.assert_called_once_with(segments)

    def test_embedder_called_with_chunks(self):
        segments = [TranscriptSegment(text="Some content.", start=0.0)]
        mock_store = MagicMock()
        mock_loader = MagicMock()
        chunks = ["Some content."]

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.TranscriptChunker") as mock_chunker_cls,
            patch("src.rag.pipeline.Embedder") as mock_embedder_cls,
            patch("src.rag.pipeline.Retriever"),
            patch("src.rag.pipeline.create_qa_chain"),
            patch("src.rag.pipeline.generate_answer", return_value="answer"),
        ):
            mock_chunker_cls.return_value.chunk.return_value = chunks
            mock_embedder_cls.return_value.embed.return_value = mock_store

            answer_question(
                "https://www.youtube.com/watch?v=abc123",
                "question?",
                llm_loader=mock_loader,
            )

        mock_embedder_cls.return_value.embed.assert_called_once_with(chunks)

    def test_default_llm_loader_used_when_not_provided(self):
        segments = [TranscriptSegment(text="Some content.", start=0.0)]
        mock_store = MagicMock()
        mock_loader = MagicMock()

        with (
            patch(
                "src.rag.pipeline.YtLoader.load_video_data",
                return_value=_make_video_data(segments),
            ),
            patch("src.rag.pipeline.TranscriptChunker") as mock_chunker_cls,
            patch("src.rag.pipeline.Embedder") as mock_embedder_cls,
            patch("src.rag.pipeline.Retriever"),
            patch("src.rag.pipeline.create_qa_chain"),
            patch("src.rag.pipeline.generate_answer", return_value="answer"),
            patch("src.rag.pipeline._default_llm_loader", mock_loader),
        ):
            mock_chunker_cls.return_value.chunk.return_value = ["Some content."]
            mock_embedder_cls.return_value.embed.return_value = mock_store

            answer_question("https://www.youtube.com/watch?v=abc123", "question?")

        mock_loader.load.assert_called_once()

    def test_value_error_from_loader_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            side_effect=ValueError("No transcript"),
        ):
            result = answer_question(
                "https://www.youtube.com/watch?v=abc123", "What is this?"
            )
        assert "Could not load video" in result

    def test_unexpected_error_from_loader_returns_message(self):
        with patch(
            "src.rag.pipeline.YtLoader.load_video_data",
            side_effect=RuntimeError("Network error"),
        ):
            result = answer_question(
                "https://www.youtube.com/watch?v=abc123", "What is this?"
            )
        assert "unexpected error" in result.lower()
