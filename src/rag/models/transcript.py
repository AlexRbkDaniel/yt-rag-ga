from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    """Represents a single transcript snippet with its text and timestamp."""

    text: str
    start: float
