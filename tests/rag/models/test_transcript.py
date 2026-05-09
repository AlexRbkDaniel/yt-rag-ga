import pytest

from src.rag.models.transcript import TranscriptSegment


class TestTranscriptSegment:
    def test_basic_instantiation(self):
        seg = TranscriptSegment(text="Hello world", start=0.0)
        assert seg.text == "Hello world"
        assert seg.start == 0.0

    def test_fractional_start_time(self):
        seg = TranscriptSegment(text="mid segment", start=12.345)
        assert seg.start == 12.345

    def test_negative_start_time_allowed(self):
        seg = TranscriptSegment(text="edge case", start=-1.0)
        assert seg.start == -1.0

    def test_empty_text_allowed(self):
        seg = TranscriptSegment(text="", start=0.0)
        assert seg.text == ""

    def test_immutable_text(self):
        seg = TranscriptSegment(text="Hello", start=0.0)
        with pytest.raises(Exception):
            seg.text = "Changed"

    def test_immutable_start(self):
        seg = TranscriptSegment(text="Hello", start=0.0)
        with pytest.raises(Exception):
            seg.start = 5.0

    def test_equality(self):
        a = TranscriptSegment(text="same", start=1.0)
        b = TranscriptSegment(text="same", start=1.0)
        assert a == b

    def test_inequality_different_text(self):
        a = TranscriptSegment(text="one", start=1.0)
        b = TranscriptSegment(text="two", start=1.0)
        assert a != b

    def test_inequality_different_start(self):
        a = TranscriptSegment(text="same", start=1.0)
        b = TranscriptSegment(text="same", start=2.0)
        assert a != b

    def test_hashable(self):
        seg = TranscriptSegment(text="Hello", start=0.0)
        s = {seg}
        assert seg in s
