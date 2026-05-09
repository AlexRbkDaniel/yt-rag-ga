from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING, Any

import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable, NoTranscriptFound

if TYPE_CHECKING:
    from youtube_transcript_api._transcripts import TranscriptList, FetchedTranscript

from src.rag.models.transcript import TranscriptSegment
from src.rag.models.video import VideoData, VideoMetadata

LOG = logging.getLogger(__name__)

_OEMBED_URL = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"


class YtLoader:
    """Loads and processes English transcripts from YouTube videos."""

    _pattern: re.Pattern = re.compile(
        r'(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    )

    @staticmethod
    def _extract_video_id(video_url: str) -> str:
        """Extracts the 11-character video ID from a YouTube URL."""
        # Strip whitespace to handle URLs pasted with leading/trailing spaces
        match: re.Match | None = YtLoader._pattern.search(video_url.strip())
        if not match:
            LOG.error("Invalid or unsupported YouTube URL: '%s'", video_url)
            raise ValueError(f"Invalid or unsupported YouTube URL: '{video_url}'")
        video_id: str = str(match.group(1))
        LOG.debug("Extracted video_id: %s", video_id)
        return video_id

    @staticmethod
    def _fetch_video_metadata(video_id: str, video_url: str) -> VideoMetadata:
        """Fetches video title, author, author_url and thumbnail_url via the YouTube oEmbed API.

        No API key required. Likes, views, description, published_at, duration and tags
        require the YouTube Data API v3 and default to 0 / None until that integration
        is added (see docs/youtube-data-api-setup.md).
        """
        LOG.debug("Fetching video metadata for video_id: %s", video_id)
        try:
            response = requests.get(_OEMBED_URL.format(video_id=video_id), timeout=10)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            metadata = VideoMetadata(
                video_id=video_id,
                video_url=video_url,
                title=data.get("title"),
                author=data.get("author_name"),
                author_url=data.get("author_url"),
                thumbnail_url=data.get("thumbnail_url"),
            )
            LOG.debug("Fetched metadata for video_id %s: title='%s', author='%s'", video_id, metadata.title, metadata.author)
            return metadata
        except (requests.RequestException, ValueError) as e:
            # Non-fatal — return partial metadata rather than failing the whole load
            LOG.warning("Failed to fetch video metadata for video_id %s: %s", video_id, e)
            return VideoMetadata(video_id=video_id, video_url=video_url)

    @staticmethod
    def _list_transcripts(video_id: str) -> TranscriptList:
        """Fetches the list of available transcripts for a video, raising ValueError on API failures."""
        try:
            return YouTubeTranscriptApi().list(video_id)
        except VideoUnavailable:
            LOG.error("Video is unavailable for video_id: %s", video_id)
            raise ValueError(f"Video is unavailable for video_id: '{video_id}'")
        except TranscriptsDisabled:
            LOG.error("Transcripts are disabled for video_id: %s", video_id)
            raise ValueError(f"Transcripts are disabled for video_id: '{video_id}'")
        except NoTranscriptFound:
            LOG.error("No transcript found for video_id: %s", video_id)
            raise ValueError(f"No transcript found for video_id: '{video_id}'")
        except Exception as e:
            LOG.error("Unexpected error while listing transcripts for video_id %s: %s", video_id, e)
            raise ValueError(f"Unexpected error while listing transcripts for video_id: '{video_id}'") from e

    @staticmethod
    def _extract_transcript(video_id: str) -> FetchedTranscript:
        """Selects and fetches the best available English transcript, preferring manual over auto-generated."""
        LOG.debug("Extracting transcript for video_id: %s", video_id)
        transcripts: TranscriptList = YtLoader._list_transcripts(video_id)
        fetched: FetchedTranscript | None = None
        for inner_transcript in transcripts:
            # Only consider English transcripts
            if inner_transcript.language_code == 'en':
                if inner_transcript.is_generated:
                    # Use auto-generated only if no transcript has been set yet
                    if not fetched:
                        try:
                            fetched = inner_transcript.fetch()
                            LOG.debug("Using auto-generated English transcript for video_id: %s", video_id)
                        except Exception as e:
                            LOG.warning("Failed to fetch auto-generated transcript for video_id %s: %s", video_id, e)
                else:
                    # Manual transcript takes priority — fall back to auto-generated if it fails
                    try:
                        fetched = inner_transcript.fetch()
                        LOG.debug("Using manual English transcript for video_id: %s", video_id)
                        break
                    except Exception as e:
                        LOG.warning("Failed to fetch manual transcript, keeping auto-generated fallback for video_id %s: %s", video_id, e)
        if not fetched:
            LOG.error("No English transcript found for video_id: %s", video_id)
            raise ValueError(f"No English transcript found for video_id: '{video_id}'")
        return fetched

    @staticmethod
    def _process_transcript(fetched: FetchedTranscript) -> list[TranscriptSegment]:
        """Converts a FetchedTranscript into a list of TranscriptSegment DTOs."""
        LOG.debug("Processing transcript snippets")
        segments: list[TranscriptSegment] = []
        for snippet in fetched:
            try:
                # Extract plain text and start time into the DTO — no formatting applied here
                segments.append(TranscriptSegment(text=snippet.text, start=snippet.start))
            except AttributeError:
                # Malformed snippet — warn and skip
                LOG.warning("Skipping malformed snippet with missing text or start fields")
        if not segments:
            LOG.error("Transcript is empty — no valid snippets were processed")
            raise ValueError("Transcript is empty — no valid snippets were processed")
        LOG.debug("Processed %d transcript segments", len(segments))
        return segments

    @staticmethod
    def load_video_data(video_url: str) -> VideoData:
        """Loads a YouTube video's metadata and transcript concurrently, returning a fully populated VideoData DTO."""
        if not video_url.strip():
            raise ValueError("video_url cannot be empty")
        # Strip once here so both metadata and video_url field are consistent
        video_url = video_url.strip()
        LOG.info("Loading video data for video_url: %s", video_url)
        video_id: str = YtLoader._extract_video_id(video_url)

        # Fetch metadata and transcript concurrently — they hit independent APIs
        with ThreadPoolExecutor(max_workers=2) as executor:
            metadata_future: Future = executor.submit(YtLoader._fetch_video_metadata, video_id, video_url)
            transcript_future: Future = executor.submit(YtLoader._extract_transcript, video_id)
            metadata: VideoMetadata = metadata_future.result()
            fetched: FetchedTranscript = transcript_future.result()

        segments: list[TranscriptSegment] = YtLoader._process_transcript(fetched)
        result = VideoData(metadata=metadata, transcript=segments)
        LOG.info("Successfully loaded video '%s' with %d transcript segments", metadata.title, len(segments))
        return result
