from dataclasses import dataclass, field

from src.rag.models.transcript import TranscriptSegment


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video.

    Fields marked as oEmbed are populated via the existing YouTube oEmbed API call.
    Fields marked as API v3 require the YouTube Data API v3 (see docs/youtube-data-api-setup.md).
    """

    video_id: str
    video_url: str

    # TODO: populated from oEmbed API call — already integrated
    title: str | None = None
    author: str | None = None
    author_url: str | None = None
    thumbnail_url: str | None = None

    # TODO: populate via YouTube Data API v3 (see docs/youtube-data-api-setup.md)
    likes: int | None = None
    views: int | None = None
    description: str | None = None
    published_at: str | None = None
    duration: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class VideoData:
    """Combines video metadata with its processed transcript segments."""

    metadata: VideoMetadata
    transcript: list[TranscriptSegment] = field(default_factory=list)
