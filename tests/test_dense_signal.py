import numpy as np
import pytest

from viewer.core import NS_PER_S, Selection, StreamView, TimeRange, ViewRequest
from viewer.dense_signal import DenseSignalAdapter


def make_adapter(samples: int = 1000, channels: int = 3, rate: float = 1000.0):
    data = np.arange(channels * samples, dtype=np.float32).reshape(channels, samples)
    return DenseSignalAdapter("sig", data, rate, target_chunk_bytes=4096)


def make_request(start_ns: int, stop_ns: int, width_px: int = 200, channels=None):
    streams = (StreamView("sig", Selection(channels=channels)),)
    return ViewRequest(time=TimeRange(start_ns, stop_ns), width_px=width_px, streams=streams)


def test_time_range_spans_full_data():
    adapter = make_adapter(samples=1000, rate=1000.0)
    assert adapter.time_range.start_ns == 0
    assert adapter.time_range.stop_ns == NS_PER_S  # 1000 samples / 1000 Hz = 1s


def test_invalid_data_shape_raises():
    with pytest.raises(ValueError):
        DenseSignalAdapter("x", np.zeros((2, 2, 2)), 100.0)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        DenseSignalAdapter("x", np.zeros((1, 0)), 100.0)


def test_visible_keys_contiguous():
    adapter = make_adapter()
    req = make_request(0, NS_PER_S // 2)
    keys = adapter.visible_keys(req, req.streams[0])
    assert len(keys) >= 1
    indices = [int(k) for k in keys]
    assert indices == sorted(indices)
    assert indices == list(range(indices[0], indices[-1] + 1))


def test_neighbor_key_walks_indices():
    adapter = make_adapter()
    k0 = adapter._key(2)
    assert adapter.neighbor_key(k0, 1) == adapter._key(3)
    assert adapter.neighbor_key(k0, -1) == adapter._key(1)


def test_neighbor_key_returns_none_at_edges():
    adapter = make_adapter()
    first = adapter._key(0)
    last = adapter._key(adapter._chunk_count - 1)
    assert adapter.neighbor_key(first, -1) is None
    assert adapter.neighbor_key(last, 1) is None


def test_fetch_returns_chunk_with_nbytes():
    adapter = make_adapter()
    chunk = adapter.fetch(adapter._key(0))
    assert chunk.key == adapter._key(0)
    assert chunk.nbytes > 0
    assert "values" in chunk.data
    assert "time_ns" in chunk.data
    assert chunk.data["values"].shape[0] == adapter.channel_count


def test_build_frame_returns_payload_within_window():
    adapter = make_adapter(samples=500, rate=1000.0)  # 0.5s of data
    req = make_request(0, NS_PER_S // 4, width_px=50)  # request 0..0.25s
    chunks = [adapter.fetch(k) for k in adapter.visible_keys(req, req.streams[0])]
    payload = adapter.build_frame(chunks, req, req.streams[0])
    assert payload is not None
    assert payload["time_ns"].size <= 100  # downsampled near width_px
    assert (payload["time_ns"] >= req.time.start_ns).all()
    assert (payload["time_ns"] < req.time.stop_ns).all()


def test_build_frame_applies_channel_selection():
    adapter = make_adapter(channels=4)
    req = make_request(0, NS_PER_S // 4, channels=(0, 2))
    chunks = [adapter.fetch(k) for k in adapter.visible_keys(req, req.streams[0])]
    payload = adapter.build_frame(chunks, req, req.streams[0])
    assert payload["values"].shape[0] == 2
    assert tuple(payload["channel_indices"]) == (0, 2)


def test_build_frame_no_chunks_returns_none():
    adapter = make_adapter()
    req = make_request(0, NS_PER_S // 4)
    assert adapter.build_frame([], req, req.streams[0]) is None
