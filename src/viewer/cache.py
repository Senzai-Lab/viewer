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

        self.stream_counts: dict[str, int] = {}
        self.used_bytes = 0
        self.chunks_per_stream = chunks_per_stream
        self.max_budget = max_budget

    def add(self, stream: Stream):
        self.streams[stream.name] = stream
    
    @property
    def used(self) -> int:
        return self.used_bytes
    
    def _evict(self, key):
        payload = self.cache.pop(key)
        self.used_bytes -= payload['n_bytes']
        self.stream_counts[key[0]] -= 1
    
    def _admit(self, key: tuple, payload: dict):
        stream_name = key[0]
        size = payload['n_bytes']

        # per-stream cap
        while self.stream_counts.get(stream_name, 0) >= self.chunks_per_stream:
            for k in self.cache: # oldest-first iteration
                if k[0] == stream_name:
                    self._evict(k)
                    break
        
        # global hard cap
        while self.used_bytes + size > self.max_budget and self.cache:
            self._evict(next(iter(self.cache)))

        self.cache[key] = payload
        self.cache.move_to_end(key)
        self.used_bytes += size
        self.stream_counts[stream_name] = self.stream_counts.get(stream_name, 0) + 1

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
        self.stream_counts.clear()
        self.used_bytes = 0

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()
        self.cache.clear()
        self.stream_counts.clear()
        self.used_bytes = 0
