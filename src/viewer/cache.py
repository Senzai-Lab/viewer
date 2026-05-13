from __future__ import annotations
from typing import TYPE_CHECKING

from concurrent.futures import ThreadPoolExecutor, Future
from collections import OrderedDict

if TYPE_CHECKING:
    from .stream import Stream

ChunkKey = tuple[str, int]

class ChunkCache:
    """
    For each stream at given time t, request the current chunk and its
    immediate neighbor chunks.
    """
    def __init__(self, max_workers: int = 4, max_budget: int = 2 * 1024**3):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.cache: OrderedDict[ChunkKey, dict] = OrderedDict()
        self.pending: dict[ChunkKey, Future] = {}
        self.streams: dict[str, Stream] = {}
        self.desired: dict[str, set[int]] = {}
        self.max_budget = max_budget

    def add(self, stream: Stream):
        if stream.name in self.streams:
            raise ValueError(f"Duplicate stream name: {stream.name}")

        new_required = self.required_budget + stream.chunk_nbytes * 3

        if new_required > self.max_budget:
            raise ValueError(f"Can't add {stream.name}: required {new_required} > budget {self.max_budget}")

        self.streams[stream.name] = stream
        self.desired[stream.name] = set()
    
    @property
    def required_budget(self) -> int:
        return sum(s.chunk_nbytes * 3 for s in self.streams.values())
    
    @property
    def used_bytes(self) -> int:
        return sum(payload.get("n_bytes", 0) for payload in self.cache.values())

    def _is_desired(self, key: ChunkKey) -> bool:
        stream_name, chunk_idx = key
        return chunk_idx in self.desired.get(stream_name, set())

    def request(self, stream_name: str, t: float):
        indices = self.chunk_indices(stream_name, t)
        wanted = set(indices)
        center = self.streams[stream_name].chunk_at(t)

        self.desired[stream_name] = wanted

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
                self.pending[key] = self.pool.submit(
                    self.streams[stream_name].load_chunk,
                    idx,
                )
    
    def get_chunks(self, stream_name: str, t: float) -> list[dict]:
        """Return cached chunks around cursor time in drawing order."""
        chunks = []
        
        for idx in self.chunk_indices(stream_name, t):
            key = (stream_name, idx)
            if key in self.cache:
                chunks.append(self.cache[key])
        return chunks

    def chunk_indices(self, stream_name: str, t: float) -> list[int]:
        stream = self.streams[stream_name]
        center = stream.chunk_at(t)

        start = max(0, center - 1)
        stop = min(stream.n_chunks, center + 2)

        return list(range(start, stop))

    def poll(self):
        done_keys = [key for key, future in self.pending.items() if future.done()]

        for key in done_keys:
            future = self.pending.pop(key)
            
            if not self._is_desired(key):
                continue
            
            try:
                payload = future.result()
            except Exception as e:
                print(f"Chunk load failed {key}: {e}")
                continue

            self.cache[key] = payload

    def reset(self):
        for f in self.pending.values():
            f.cancel()
        self.pending.clear()
        self.cache.clear()
        for wanted in self.desired.values():
            wanted.clear()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()
        self.cache.clear()
        for wanted in self.desired.values():
            wanted.clear()
