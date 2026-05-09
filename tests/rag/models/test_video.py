from src.rag.models.transcript import TranscriptSegment
from src.rag.models.video import VideoData, VideoMetadata


class TestVideoMetadata:
    def test_required_fields(self):
        meta = VideoMetadata(
            video_id="abc123", video_url="https://youtube.com/watch?v=abc123"
        )
        assert meta.video_id == "abc123"
        assert meta.video_url == "https://youtube.com/watch?v=abc123"

    def test_optional_fields_default_to_none(self):
        meta = VideoMetadata(
            video_id="abc123", video_url="https://youtube.com/watch?v=abc123"
        )
        assert meta.title is None
        assert meta.author is None
        assert meta.author_url is None
        assert meta.thumbnail_url is None
        assert meta.likes is None
        assert meta.views is None
        assert meta.description is None
        assert meta.published_at is None
        assert meta.duration is None

    def test_tags_default_to_empty_list(self):
        meta = VideoMetadata(
            video_id="abc123", video_url="https://youtube.com/watch?v=abc123"
        )
        assert meta.tags == []

    def test_tags_not_shared_between_instances(self):
        a = VideoMetadata(video_id="a", video_url="u")
        b = VideoMetadata(video_id="b", video_url="u")
        a.tags.append("python")
        assert b.tags == []

    def test_all_fields_set(self):
        meta = VideoMetadata(
            video_id="abc123",
            video_url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            author="Test Author",
            author_url="https://youtube.com/@testauthor",
            thumbnail_url="https://img.youtube.com/vi/abc123/0.jpg",
            likes=1000,
            views=50000,
            description="A test video.",
            published_at="2024-01-01",
            duration="PT10M",
            tags=["python", "ai"],
        )
        assert meta.title == "Test Video"
        assert meta.likes == 1000
        assert meta.views == 50000
        assert meta.tags == ["python", "ai"]

    def test_likes_zero_is_none_by_default(self):
        meta = VideoMetadata(video_id="x", video_url="u")
        assert meta.likes is None

    def test_views_zero_is_none_by_default(self):
        meta = VideoMetadata(video_id="x", video_url="u")
        assert meta.views is None


class TestVideoData:
    def test_requires_metadata(self):
        meta = VideoMetadata(
            video_id="abc123", video_url="https://youtube.com/watch?v=abc123"
        )
        data = VideoData(metadata=meta)
        assert data.metadata is meta

    def test_transcript_defaults_to_empty_list(self):
        meta = VideoMetadata(
            video_id="abc123", video_url="https://youtube.com/watch?v=abc123"
        )
        data = VideoData(metadata=meta)
        assert data.transcript == []

    def test_transcript_not_shared_between_instances(self):
        meta = VideoMetadata(video_id="abc123", video_url="u")
        a = VideoData(metadata=meta)
        b = VideoData(metadata=meta)
        a.transcript.append(TranscriptSegment(text="hi", start=0.0))
        assert b.transcript == []

    def test_transcript_set_explicitly(self):
        meta = VideoMetadata(video_id="abc123", video_url="u")
        segments = [TranscriptSegment(text="Hello", start=0.0)]
        data = VideoData(metadata=meta, transcript=segments)
        assert data.transcript == segments
