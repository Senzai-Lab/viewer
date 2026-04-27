from viewer.core import (
    NS_PER_S,
    LoadedChunk,
    Selection,
    StreamFrame,
    StreamView,
    TimeRange,
    ViewRequest,
    ns_to_seconds,
    seconds_to_ns,
)


def test_seconds_ns_roundtrip():
    assert seconds_to_ns(1.5) == 1_500_000_000
    assert ns_to_seconds(NS_PER_S) == 1.0
    assert ns_to_seconds(seconds_to_ns(0.25)) == 0.25


def test_time_range_duration():
    tr = TimeRange(100, 250)
    assert tr.duration_ns == 150


def test_stream_view_default_selection():
    view = StreamView("s")
    assert view.stream_id == "s"
    assert view.selection == Selection()


def test_view_request_defaults():
    req = ViewRequest(time=TimeRange(0, 10), width_px=800)
    assert req.streams == ()
    assert req.cursor_ns == 0
    assert req.direction == 1
    assert req.playing is False
    assert req.jumped is False


def test_loaded_chunk_fields():
    chunk = LoadedChunk(key="k", time=TimeRange(0, 1), data=42, nbytes=8)
    assert chunk.key == "k"
    assert chunk.data == 42
    assert chunk.nbytes == 8


def test_stream_frame_ready():
    sv = StreamView("a")
    assert StreamFrame(sv, coverage=1.0).ready is True
    assert StreamFrame(sv, coverage=0.5).ready is False
