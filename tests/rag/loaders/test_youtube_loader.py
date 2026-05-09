import pytest
import requests as req
from unittest.mock import patch, MagicMock

from src.rag.loaders.youtube_loader import YtLoader
from src.rag.models.transcript import TranscriptSegment


class TestExtractVideoId:

    def test_standard_url(self):
        assert YtLoader._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert YtLoader._extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        assert YtLoader._extract_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_http_url(self):
        assert YtLoader._extract_video_id("http://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_leading_trailing_whitespace(self):
        assert YtLoader._extract_video_id("  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid or unsupported YouTube URL"):
            YtLoader._extract_video_id("https://www.google.com")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid or unsupported YouTube URL"):
            YtLoader._extract_video_id("")

    def test_random_string_raises(self):
        with pytest.raises(ValueError, match="Invalid or unsupported YouTube URL"):
            YtLoader._extract_video_id("not-a-url")


class TestFetchVideoMetadata:

    def _mock_response(self, data: dict) -> MagicMock:
        response = MagicMock()
        response.json.return_value = data
        return response

    def test_successful_fetch_populates_all_fields(self):
        data = {
            "title": "Test Video",
            "author_name": "Test Author",
            "author_url": "https://youtube.com/channel/test",
            "thumbnail_url": "https://i.ytimg.com/vi/test/hqdefault.jpg",
        }
        with patch("src.rag.loaders.youtube_loader.requests.get", return_value=self._mock_response(data)):
            metadata = YtLoader._fetch_video_metadata("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert metadata.title == "Test Video"
        assert metadata.author == "Test Author"
        assert metadata.author_url == "https://youtube.com/channel/test"
        assert metadata.thumbnail_url == "https://i.ytimg.com/vi/test/hqdefault.jpg"
        assert metadata.video_id == "dQw4w9WgXcQ"

    def test_network_failure_returns_partial_metadata(self):
        with patch("src.rag.loaders.youtube_loader.requests.get", side_effect=req.RequestException("timeout")):
            metadata = YtLoader._fetch_video_metadata("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert metadata.video_id == "dQw4w9WgXcQ"
        assert metadata.title is None
        assert metadata.author is None

    def test_invalid_json_returns_partial_metadata(self):
        response = MagicMock()
        response.json.side_effect = ValueError("invalid json")
        with patch("src.rag.loaders.youtube_loader.requests.get", return_value=response):
            metadata = YtLoader._fetch_video_metadata("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert metadata.video_id == "dQw4w9WgXcQ"
        assert metadata.title is None

    def test_missing_fields_default_to_none(self):
        with patch("src.rag.loaders.youtube_loader.requests.get", return_value=self._mock_response({})):
            metadata = YtLoader._fetch_video_metadata("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert metadata.title is None
        assert metadata.author is None
        assert metadata.author_url is None
        assert metadata.thumbnail_url is None


class TestProcessTranscript:

    def _make_snippet(self, text: str, start: float) -> MagicMock:
        snippet = MagicMock()
        snippet.text = text
        snippet.start = start
        return snippet

    def test_valid_snippets_return_segments(self):
        snippets = [self._make_snippet("Hello world", 0.0), self._make_snippet("Goodbye", 5.0)]
        segments = YtLoader._process_transcript(snippets)

        assert len(segments) == 2
        assert segments[0] == TranscriptSegment(text="Hello world", start=0.0)
        assert segments[1] == TranscriptSegment(text="Goodbye", start=5.0)

    def test_malformed_snippet_is_skipped(self):
        good = self._make_snippet("Hello", 0.0)
        bad = MagicMock(spec=[])  # no attributes — raises AttributeError on access
        segments = YtLoader._process_transcript([good, bad])

        assert len(segments) == 1
        assert segments[0].text == "Hello"

    def test_all_malformed_snippets_raises(self):
        bad = MagicMock(spec=[])
        with pytest.raises(ValueError, match="Transcript is empty"):
            YtLoader._process_transcript([bad])

    def test_empty_fetched_raises(self):
        with pytest.raises(ValueError, match="Transcript is empty"):
            YtLoader._process_transcript([])


class TestLoadVideoData:

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="video_url cannot be empty"):
            YtLoader.load_video_data("")

    def test_whitespace_url_raises(self):
        with pytest.raises(ValueError, match="video_url cannot be empty"):
            YtLoader.load_video_data("   ")

    def test_url_is_stripped_before_storing(self):
        video_id = "dQw4w9WgXcQ"
        url = f"  https://www.youtube.com/watch?v={video_id}  "

        mock_metadata = MagicMock()
        mock_fetched = [MagicMock()]
        mock_fetched[0].text = "Hello"
        mock_fetched[0].start = 0.0

        with patch.object(YtLoader, "_fetch_video_metadata", return_value=mock_metadata) as mock_meta, \
             patch.object(YtLoader, "_extract_transcript", return_value=mock_fetched):
            YtLoader.load_video_data(url)
            # Stripped URL is passed to metadata fetcher
            mock_meta.assert_called_once_with(video_id, url.strip())