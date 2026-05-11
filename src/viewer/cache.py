from __future__ import annotations
from typing import TYPE_CHECKING

from concurrent.futures import ThreadPoolExecutor, Future
from collections import OrderedDict

if TYPE_CHECKING:
    from .stream import Stream

class ChunkCache:
    def __init__(self, max_workers: int = 4, chunks_per_stream: int = 5, max_budget: int = 2 * 1024**3):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.cache: OrderedDict[tuple, dict] = OrderedDict()
        self.pending: dict[tuple, Future] = {}
        self.streams: dict[str, Stream] = {}
        self.chunks_per_stream = chunks_per_stream
        self.budget = 0
        self.max_budget = max_budget

    def add(self, stream: Stream):
        self.streams[stream.name] = stream
        self.budget = min(self.budget + stream.chunk_nbytes * self.chunks_per_stream,
                          self.max_budget)
    
    @property
    def used(self) -> int:
        return sum(self.streams[k[0]].chunk_nbytes for k in self.cache)

    def get_range(self, stream_name: str, t0: float, t1: float) -> list[dict]:
        """Return cached chunks covering the given time interval."""
        # TODO: limit chunks when range is too large
        stream = self.streams[stream_name]
        start = stream.chunk_at(t0)
        stop = stream.chunk_at(t1)

        chunks = []
        for idx in range(start, stop + 1):
            key = (stream_name, idx)
            if key in self.cache:
                self.cache.move_to_end(key)
                chunks.append(self.cache[key])
        return chunks

    def prefetch(self, stream_name: str, t: float):
        """Submit current chunk + immediate neighbors for loading."""
        stream = self.streams[stream_name]
        center = stream.chunk_at(t)

        for idx in range(max(center - 1, 0), min(center + 2, stream.n_chunks)):
            key = (stream_name, idx)
            if key not in self.cache and key not in self.pending:
                self.pending[key] = self.pool.submit(stream.load_chunk, idx)

    def poll(self):
        done_keys = [k for k, f in self.pending.items() if f.done()]
        for key in done_keys:
            future = self.pending.pop(key)
            try:
                self._admit(key, future.result())
            except Exception as e:
                print(f"Chunk load failed {key}: {e}")

    def reset(self):
        for f in self.pending.values():
            f.cancel()
        self.pending.clear()
        self.cache.clear()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()
        self.cache.clear()

    def _admit(self, key: tuple, payload: dict):
        # key = (name, chunk_idx)
        size = self.streams[key[0]].chunk_nbytes
        while self.used + size > self.budget and self.cache:
            self.cache.popitem(last=False)
        self.cache[key] = payload
        self.cache.move_to_end(key)