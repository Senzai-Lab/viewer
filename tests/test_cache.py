import threading
import time

import pytest

from viewer.cache import ChunkManager
from viewer.core import LoadedChunk, StreamView, TimeRange, ViewRequest


class StubAdapter:
    """Minimal in-memory adapter for testing the cache in isolation."""

    def __init__(self, stream_id: str, n_chunks: int = 8, nbytes_per_chunk: int = 1024):
        self.stream_id = stream_id
        self.n_chunks = n_chunks
        self.nbytes_per_chunk = nbytes_per_chunk
        self.time_range = TimeRange(0, n_chunks * 1000)
        self.fetch_calls = 0
        self._lock = threading.Lock()

    def _key(self, i: int) -> str:
        return f"{i:04d}"

    def visible_keys(self, request, stream):
        # interpret request.time as chunk index range
        start = max(0, request.time.start_ns // 1000)
        stop = min(self.n_chunks, max(start + 1, request.time.stop_ns // 1000))
        return [self._key(i) for i in range(start, stop)]

    def neighbor_key(self, key, direction):
        i = int(key) + (1 if direction >= 0 else -1)
        if not 0 <= i < self.n_chunks:
            return None
        return self._key(i)

    def fetch(self, key):
        with self._lock:
            self.fetch_calls += 1
        return LoadedChunk(
            key=key,
            time=TimeRange(int(key) * 1000, (int(key) + 1) * 1000),
            data={"key": key},
            nbytes=self.nbytes_per_chunk,
        )

    def build_frame(self, chunks, request, stream):
        return [c.data["key"] for c in chunks]


def drain(manager: ChunkManager, request: ViewRequest, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frames = manager.update(request)
        if all(f.ready for f in frames.values()):
            return frames
        time.sleep(0.01)
    pytest.fail("cache did not converge to coverage=1.0 within timeout")


def make_request(start: int, stop: int, *stream_ids: str) -> ViewRequest:
    return ViewRequest(
        time=TimeRange(start, stop),
        width_px=100,
        streams=tuple(StreamView(sid) for sid in stream_ids),
    )


def test_rejects_empty_adapters():
    with pytest.raises(ValueError):
        ChunkManager([])


def test_rejects_duplicate_stream_id():
    with pytest.raises(ValueError):
        ChunkManager([StubAdapter("a"), StubAdapter("a")])


def test_rejects_invalid_budget():
    with pytest.raises(ValueError):
        ChunkManager([StubAdapter("a")], memory_budget_bytes=0)


def test_update_eventually_ready():
    adapter = StubAdapter("a", n_chunks=4)
    mgr = ChunkManager([adapter], max_workers=2)
    try:
        request = make_request(0, 3000, "a")  # chunks 0,1,2
        frames = drain(mgr, request)
        assert frames["a"].coverage == 1.0
        assert frames["a"].data == ["0000", "0001", "0002"]
    finally:
        mgr.close()


def test_multiple_streams_round_robin():
    a = StubAdapter("a", n_chunks=4)
    b = StubAdapter("b", n_chunks=4)
    mgr = ChunkManager([a, b], max_workers=2)
    try:
        request = make_request(0, 2000, "a", "b")
        frames = drain(mgr, request)
        assert frames["a"].coverage == 1.0
        assert frames["b"].coverage == 1.0
        assert a.fetch_calls >= 2
        assert b.fetch_calls >= 2
    finally:
        mgr.close()


def test_loaded_bytes_tracks_chunks():
    adapter = StubAdapter("a", n_chunks=4, nbytes_per_chunk=1024)
    mgr = ChunkManager([adapter])
    try:
        request = make_request(0, 3000, "a")
        drain(mgr, request)
        assert mgr.loaded_bytes == 3 * 1024
    finally:
        mgr.close()


def test_eviction_respects_budget_and_keeps_working_set():
    adapter = StubAdapter("a", n_chunks=10, nbytes_per_chunk=1024)
    # budget allows only ~3 chunks, but visible window has 4 chunks (sticky)
    mgr = ChunkManager([adapter], memory_budget_bytes=3 * 1024)
    try:
        # First load chunks 0..3
        request1 = make_request(0, 4000, "a")
        drain(mgr, request1)
        # Working set is sticky -> all 4 stay despite budget
        assert mgr.loaded_bytes == 4 * 1024
        # Move window to chunks 4..7. Old non-working chunks should be evicted.
        request2 = make_request(4000, 8000, "a")
        drain(mgr, request2)
        # New working set is 4 chunks, also sticky over budget
        assert mgr.loaded_bytes == 4 * 1024
    finally:
        mgr.close()


def test_close_is_idempotent():
    mgr = ChunkManager([StubAdapter("a")])
    mgr.close()
    mgr.close()


def test_clear_flushes_loaded_state():
    adapter = StubAdapter("a", n_chunks=6, nbytes_per_chunk=1024)
    mgr = ChunkManager([adapter])
    try:
        request = make_request(0, 3000, "a")
        drain(mgr, request)
        assert mgr.loaded_bytes == 3 * 1024
        assert mgr.loaded_chunk_count == 3

        mgr.clear()

        assert mgr.loaded_bytes == 0
        assert mgr.loaded_chunk_count == 0
        assert mgr.pending_chunk_count == 0
        assert mgr.working_chunk_count == 0
    finally:
        mgr.close()


def test_stream_debug_stats_report_per_stream_details():
    a = StubAdapter("a", n_chunks=4, nbytes_per_chunk=1024)
    b = StubAdapter("b", n_chunks=4, nbytes_per_chunk=2048)
    mgr = ChunkManager([a, b], max_workers=2)
    try:
        request = make_request(0, 2000, "a", "b")
        drain(mgr, request)

        stats = mgr.stream_debug_stats()

        assert stats["a"].loaded_chunk_count == 2
        assert stats["a"].loaded_bytes == 2 * 1024
        assert stats["a"].loaded_keys == ("0000", "0001")
        assert stats["b"].loaded_chunk_count == 2
        assert stats["b"].loaded_bytes == 2 * 2048
        assert stats["b"].loaded_keys == ("0000", "0001")
    finally:
        mgr.close()
