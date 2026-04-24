from __future__ import annotations

from collections.abc import Hashable
from itertools import count
from queue import Empty, PriorityQueue
import threading
from typing import Any, Protocol, TypeVar

from .utils import CachedStreamState, ChunkConfig, ChunkKey, LoadedChunk, TimeRange, ViewRequest, split_chunk_key


HashableT = TypeVar("HashableT", bound=Hashable)


def stable_unique(values: list[HashableT]) -> list[HashableT]:
    seen: set[HashableT] = set()
    ordered: list[HashableT] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


class StreamAdapter(Protocol):
    stream_id: str
    t_min: float
    t_max: float
    chunk_config: ChunkConfig

    def visible_keys(self, request: ViewRequest) -> list[ChunkKey]: ...
    def neighbor_key(self, key: ChunkKey, direction: int) -> ChunkKey | None: ...
    def fetch(self, key: ChunkKey) -> LoadedChunk: ...
    def build_view(
        self,
        chunks: list[LoadedChunk],
        *,
        view: TimeRange,
        width_px: int | None = None,
    ) -> Any: ...


class ChunkManager:
    """
    Background-loading cache that keeps only the current working set.

    Each ``poll()`` asks every adapter for visible keys, derives optional
    prefetch keys, drops anything outside that working set, and queues missing
    keys. Call ``adapter.build_view(...)`` with the ready subset for one stream
    to turn it into renderable data.
    """

    def __init__(
        self,
        adapters: list[StreamAdapter],
        *,
        max_workers: int = 2,
        prefetch_distance: int = 1,
    ):
        if not adapters:
            raise ValueError("adapters must not be empty")

        adapters_by_id: dict[str, StreamAdapter] = {}
        for adapter in adapters:
            if adapter.stream_id in adapters_by_id:
                raise ValueError(f"duplicate stream_id: {adapter.stream_id}")
            adapters_by_id[adapter.stream_id] = adapter

        self._adapters = adapters_by_id
        self.stream_ids = list(adapters_by_id)

        self._lock = threading.Lock()
        self._loaded: dict[ChunkKey, LoadedChunk] = {}
        self._pending: set[ChunkKey] = set()
        self._working_set: set[ChunkKey] = set()
        self._queue: PriorityQueue[tuple[int, int, int, int, ChunkKey]] = PriorityQueue()
        self._sequence = count()
        self._generation = 0
        self._closed = threading.Event()
        self._prefetch_distance = max(0, int(prefetch_distance))

        worker_count = max(1, int(max_workers))
        self._workers = [
            threading.Thread(target=self._worker, name=f"stream-cache-{index}", daemon=True)
            for index in range(worker_count)
        ]
        for worker in self._workers:
            worker.start()

    def _prefetch_keys(
        self,
        adapter: StreamAdapter,
        request: ViewRequest,
        visible_keys: list[ChunkKey],
    ) -> list[ChunkKey]:
        if self._prefetch_distance <= 0 or request.jumped or not request.playing or not visible_keys:
            return []

        anchor_key = visible_keys[-1] if request.direction >= 0 else visible_keys[0]
        prefetch_keys: list[ChunkKey] = []
        visible_set = set(visible_keys)
        next_key = anchor_key

        for _ in range(self._prefetch_distance):
            next_key = adapter.neighbor_key(next_key, request.direction)
            if next_key is None or next_key in visible_set or next_key in prefetch_keys:
                break
            prefetch_keys.append(next_key)

        return prefetch_keys

    def close(self) -> None:
        self._closed.set()
        for worker in self._workers:
            worker.join(timeout=0.2)

    def poll(self, request: ViewRequest) -> dict[str, CachedStreamState]:
        visible_by_stream: dict[str, list[ChunkKey]] = {}
        visible_keys: list[ChunkKey] = []
        prefetch_keys: list[ChunkKey] = []

        for stream_id in self.stream_ids:
            adapter = self._adapters[stream_id]
            stream_visible_keys = stable_unique(adapter.visible_keys(request))
            stream_prefetch_keys = self._prefetch_keys(adapter, request, stream_visible_keys)

            for key in [*stream_visible_keys, *stream_prefetch_keys]:
                key_stream_id, _ = split_chunk_key(key)
                if key_stream_id != stream_id:
                    raise ValueError(
                        f"stream {stream_id!r} returned key {key!r} for {key_stream_id!r}"
                    )

            visible_by_stream[stream_id] = stream_visible_keys
            visible_keys.extend(stream_visible_keys)
            prefetch_keys.extend(stream_prefetch_keys)

        visible_keys = stable_unique(visible_keys)
        visible_set = set(visible_keys)
        prefetch_keys = [key for key in stable_unique(prefetch_keys) if key not in visible_set]
        working_set = set([*visible_keys, *prefetch_keys])

        with self._lock:
            self._generation += 1
            generation = self._generation
            self._working_set = working_set

            stale_keys = [key for key in self._loaded if key not in working_set]
            for key in stale_keys:
                self._loaded.pop(key, None)

            for band, keys in enumerate((visible_keys, prefetch_keys)):
                for order, key in enumerate(keys):
                    if key in self._loaded or key in self._pending:
                        continue
                    self._pending.add(key)
                    self._queue.put((-generation, band, order, next(self._sequence), key))

            states = {
                stream_id: CachedStreamState(
                    visible_keys=stream_visible_keys,
                    ready_chunks=[self._loaded[key] for key in stream_visible_keys if key in self._loaded],
                )
                for stream_id, stream_visible_keys in visible_by_stream.items()
            }

        return states

    def _worker(self) -> None:
        while not self._closed.is_set():
            try:
                _, _, _, _, key = self._queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                with self._lock:
                    if key in self._loaded or key not in self._working_set:
                        continue
                    stream_id, _ = split_chunk_key(key)
                    adapter = self._adapters[stream_id]

                loaded_chunk = adapter.fetch(key)
                if loaded_chunk.key != key:
                    raise ValueError(f"stream {stream_id!r} fetched {loaded_chunk.key!r} for {key!r}")
                if loaded_chunk.stream_id != stream_id:
                    raise ValueError(
                        f"stream {stream_id!r} fetched a chunk for {loaded_chunk.stream_id!r}"
                    )

                with self._lock:
                    if key in self._working_set:
                        self._loaded[key] = loaded_chunk

            finally:
                with self._lock:
                    self._pending.discard(key)
                self._queue.task_done()