from unittest.mock import MagicMock, patch

from src.rag.models.transcript import TranscriptSegment
from src.rag.models.video import VideoData, VideoMetadata
from src.ui.app import (
    _format_metadata,
    _format_timestamp,
    _format_transcript,
    on_ask,
    on_load,
    on_summarize,
)


def _make_metadata(**kwargs) -> VideoMetadata:
    defaults = {"video_id": "abc123", "video_url": "https://youtube.com/watch?v=abc123"}
    defaults.update(kwargs)
    return VideoMetadata(**defaults)


def _make_video_data(segments=None, **meta_kwargs) -> VideoData:
    meta = _make_metadata(**meta_kwargs)
    return VideoData(metadata=meta, transcript=segments or [])


class TestFormatTimestamp:
    def test_zero_seconds(self):
        assert _format_timestamp(0.0) == "[00:00]"

    def test_one_minute(self):
        assert _format_timestamp(60.0) == "[01:00]"

    def test_mixed(self):
        assert _format_timestamp(125.0) == "[02:05]"

    def test_fractional_seconds_truncated(self):
        assert _format_timestamp(90.9) == "[01:30]"


class TestFormatTranscript:
    def test_single_segment(self):
        segments = [TranscriptSegment(text="Hello world", start=0.0)]
        result = _format_transcript(segments)
        assert "[00:00] Hello world" in result

    def test_multiple_segments_separated_by_newlines(self):
        segments = [
            TranscriptSegment(text="First", start=0.0),
            TranscriptSegment(text="Second", start=60.0),
        ]
        result = _format_transcript(segments)
        assert "[00:00] First" in result
        assert "[01:00] Second" in result

    def test_empty_segments_returns_empty_string(self):
        result = _format_transcript([])
        assert result == ""


class TestFormatMetadata:
    def test_title_included(self):
        meta = _make_metadata(title="My Video")
        result = _format_metadata(meta)
        assert "My Video" in result

    def test_author_without_url(self):
        meta = _make_metadata(author="John Doe")
        result = _format_metadata(meta)
        assert "John Doe" in result

    def test_author_with_url_as_link(self):
        meta = _make_metadata(author="John Doe", author_url="https://youtube.com/@john")
        result = _format_metadata(meta)
        assert "[John Doe](https://youtube.com/@john)" in result

    def test_none_optional_fields_show_dash(self):
        meta = _make_metadata()
        result = _format_metadata(meta)
        assert "—" in result

    def test_api_v3_notice_always_present(self):
        meta = _make_metadata()
        result = _format_metadata(meta)
        assert "YouTube Data API v3" in result


class TestOnLoad:
    def test_empty_url_returns_hidden_state(self):
        outputs = on_load("")
        video_details_update = outputs[0]
        assert video_details_update["visible"] is False

    def test_whitespace_url_returns_hidden_state(self):
        outputs = on_load("   ")
        video_details_update = outputs[0]
        assert video_details_update["visible"] is False

    def test_load_error_shows_error_message(self):
        with patch(
            "src.ui.app.YtLoader.load_video_data",
            side_effect=ValueError("No transcript"),
        ):
            outputs = on_load("https://youtube.com/watch?v=abc123")
        error_update = outputs[-1]
        assert error_update["visible"] is True
        assert "Error" in error_update["value"]

    def test_unexpected_error_shows_generic_error(self):
        with patch(
            "src.ui.app.YtLoader.load_video_data", side_effect=RuntimeError("network")
        ):
            outputs = on_load("https://youtube.com/watch?v=abc123")
        error_update = outputs[-1]
        assert error_update["visible"] is True

    def test_successful_load_hides_error(self):
        segments = [TranscriptSegment(text="Hello", start=0.0)]
        video_data = _make_video_data(segments=segments)
        with (
            patch("src.ui.app.YtLoader.load_video_data", return_value=video_data),
            patch("src.ui.app.TranscriptChunker") as mock_chunker,
            patch("src.ui.app.Embedder") as mock_embedder,
            patch("src.ui.app.Retriever"),
        ):
            mock_chunker.return_value.chunk.return_value = ["Hello"]
            mock_embedder.return_value.embed.return_value = MagicMock()
            outputs = on_load("https://youtube.com/watch?v=abc123")
        error_update = outputs[-1]
        assert error_update["visible"] is False

    def test_returns_12_outputs(self):
        outputs = on_load("")
        assert len(outputs) == 12


class TestOnSummarize:
    def test_none_video_data_returns_message(self):
        result = on_summarize(None)
        assert "No transcript" in result["value"]

    def test_empty_transcript_returns_message(self):
        video_data = _make_video_data()
        result = on_summarize(video_data)
        assert "No transcript" in result["value"]

    def test_returns_summary_on_success(self):
        segments = [TranscriptSegment(text="Python is great.", start=0.0)]
        video_data = _make_video_data(segments=segments)
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "A summary."

        with (
            patch("src.ui.app.create_summary_chain", return_value=mock_chain),
            patch("src.ui.app._llm_loader"),
        ):
            result = on_summarize(video_data)

        assert result["value"] == "A summary."
        assert result["visible"] is True

    def test_chain_failure_returns_friendly_message(self):
        segments = [TranscriptSegment(text="Python is great.", start=0.0)]
        video_data = _make_video_data(segments=segments)
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("API down")

        with (
            patch("src.ui.app.create_summary_chain", return_value=mock_chain),
            patch("src.ui.app._llm_loader"),
        ):
            result = on_summarize(video_data)

        assert "error" in result["value"].lower()


class TestOnAsk:
    def test_empty_question_returns_unchanged_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        result_chatbot, result_history, cleared = on_ask("", None, history)
        assert result_chatbot == history
        assert cleared == ""

    def test_whitespace_question_returns_unchanged_history(self):
        chatbot, history, cleared = on_ask("   ", None, [])
        assert chatbot == []
        assert cleared == ""

    def test_none_retriever_returns_error_message(self):
        chatbot, history, cleared = on_ask("What is this?", None, [])
        assert any(
            "not available" in m["content"].lower()
            for m in chatbot
            if m["role"] == "assistant"
        )
        assert cleared == ""

    def test_successful_answer_appended_to_history(self):
        mock_chain = MagicMock()
        mock_retriever = MagicMock()

        with (
            patch("src.ui.app.create_qa_chain", return_value=mock_chain),
            patch("src.ui.app.generate_answer", return_value="Python is great."),
            patch("src.ui.app._llm_loader"),
        ):
            chatbot, history, cleared = on_ask("What is Python?", mock_retriever, [])

        assert any(m["content"] == "What is Python?" for m in chatbot)
        assert any(m["content"] == "Python is great." for m in chatbot)
        assert cleared == ""

    def test_chain_failure_appends_friendly_message(self):
        mock_retriever = MagicMock()

        with (
            patch("src.ui.app.create_qa_chain"),
            patch("src.ui.app.generate_answer", side_effect=RuntimeError("API down")),
            patch("src.ui.app._llm_loader"),
        ):
            chatbot, history, cleared = on_ask("What is Python?", mock_retriever, [])

        assert any(
            "error" in m["content"].lower() for m in chatbot if m["role"] == "assistant"
        )

    def test_history_preserved_across_asks(self):
        prior = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        mock_retriever = MagicMock()

        with (
            patch("src.ui.app.create_qa_chain"),
            patch("src.ui.app.generate_answer", return_value="new answer"),
            patch("src.ui.app._llm_loader"),
        ):
            chatbot, history, _ = on_ask("second question", mock_retriever, prior)

        assert len(chatbot) == 4
        assert chatbot[0]["content"] == "first"
