from __future__ import annotations

from typing import TYPE_CHECKING

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

if TYPE_CHECKING:
    from .stream import Stream

ChunkKey = tuple[str, int]


class Cache:
    """Threaded cache for current, previous, and next stream chunks."""

    def __init__(
        self,
        streams=(),
        *,
        workers: int = 4,
        max_nbytes: int = 2 * 1024**3,
    ):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.cache: OrderedDict[ChunkKey, dict] = OrderedDict()
        self.pending: dict[ChunkKey, Future] = {}
        self.streams: dict[str, Stream] = {}
        self.wanted: dict[str, set[int]] = {}
        self.max_nbytes = max_nbytes

        for stream in streams:
            self.add(stream)

    def add(self, stream: Stream):
        if stream.name in self.streams:
            raise ValueError(f"Duplicate stream name: {stream.name}")

        new_required = self.required_nbytes + stream.chunk_nbytes * 3

        if new_required > self.max_nbytes:
            raise ValueError(
                f"Can't add {stream.name}: required {new_required} > budget {self.max_nbytes}"
            )

        self.streams[stream.name] = stream
        self.wanted[stream.name] = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __contains__(self, key) -> bool:
        return self._key(*key) in self.cache

    def __getitem__(self, key) -> dict:
        return self.cache[self._key(*key)]

    def __len__(self) -> int:
        return len(self.cache)

    def keys(self):
        return self.cache.keys()

    @property
    def required_nbytes(self) -> int:
        return sum(s.chunk_nbytes * 3 for s in self.streams.values())

    @property
    def nbytes(self) -> int:
        return sum(payload["nbytes"] for payload in self.cache.values())

    def cached_indices(self, stream: Stream) -> list[int]:
        stream_name = stream.name
        return sorted(idx for name, idx in self.cache if name == stream_name)

    def pending_indices(self, stream: Stream) -> list[int]:
        stream_name = stream.name
        return sorted(idx for name, idx in self.pending if name == stream_name)

    def wanted_indices(self, stream: Stream) -> list[int]:
        return sorted(self.wanted.get(stream.name, set()))

    def cached_nbytes(self, stream: Stream) -> int:
        stream_name = stream.name
        return sum(
            payload["nbytes"]
            for (name, _), payload in self.cache.items()
            if name == stream_name
        )

    def _is_wanted(self, key: ChunkKey) -> bool:
        stream_name, chunk_idx = key
        return chunk_idx in self.wanted.get(stream_name, set())

    def request(self, stream: Stream, t: float):
        stream = self.streams[stream.name]
        indices = self._chunk_indices(stream, t)
        wanted = set(indices)
        center = stream.chunk_at(t)
        stream_name = stream.name

        self.wanted[stream_name] = wanted

        # Drop cached chunks from this stream that are no longer wanted.
        for key in list(self.cache):
            name, idx = key
            if name == stream_name and idx not in wanted:
                self.cache.pop(key)

        # Cancel pending chunks from this stream that are no longer wanted.
        for key, future in list(self.pending.items()):
            name, idx = key
            if name == stream_name and idx not in wanted:
                if future.cancel():
                    self.pending.pop(key)

        # Submit missing wanted chunks. Current chunk is submitted first.
        for idx in sorted(indices, key=lambda i: abs(i - center)):
            key = (stream_name, idx)
            if key not in self.cache and key not in self.pending:
                self.pending[key] = self.pool.submit(stream.read, idx)

    def chunks(self, stream: Stream, t: float) -> list[dict]:
        """Return cached chunks around cursor time in drawing order."""
        chunks = []
        stream = self.streams[stream.name]
        stream_name = stream.name

        for idx in self._chunk_indices(stream, t):
            key = (stream_name, idx)
            if key in self.cache:
                chunks.append(self.cache[key])
        return chunks

    def _chunk_indices(self, stream: Stream, t: float) -> list[int]:
        center = stream.chunk_at(t)

        start = max(0, center - 1)
        stop = min(stream.n_chunks, center + 2)

        return list(range(start, stop))

    def poll(self):
        done_keys = [key for key, future in self.pending.items() if future.done()]

        for key in done_keys:
            future = self.pending.pop(key)

            if not self._is_wanted(key):
                continue

            try:
                payload = future.result()
            except Exception as e:
                print(f"Chunk load failed {key}: {e}")
                continue

            self.cache[key] = payload

    def drop(self, stream: Stream | None = None):
        stream_name = None
        if stream is not None:
            stream_name = self.streams[stream.name].name

        for key in list(self.cache):
            if stream_name is None or key[0] == stream_name:
                self.cache.pop(key)
        for key, future in list(self.pending.items()):
            if stream_name is None or key[0] == stream_name:
                future.cancel()
                self.pending.pop(key)

        if stream_name is None:
            for wanted in self.wanted.values():
                wanted.clear()
        else:
            self.wanted.get(stream_name, set()).clear()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()
        self.cache.clear()
        for wanted in self.wanted.values():
            wanted.clear()

    def _key(self, stream_or_name, chunk_idx: int) -> ChunkKey:
        if isinstance(stream_or_name, str):
            return (stream_or_name, chunk_idx)
        return (stream_or_name.name, chunk_idx)
