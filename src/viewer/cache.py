from dataclasses import dataclass
from collections import OrderedDict
from itertools import count
from queue import Empty, PriorityQueue
import threading
from typing import Any, Protocol

from .core import (
    ChunkKey,
    LoadedChunk,
    StreamFrame,
    StreamView,
    TimeRange,
    ViewRequest,
)


class StreamAdapter(Protocol):
    stream_id: str
    time_range: TimeRange

    def visible_keys(self, request: ViewRequest, stream: StreamView) -> list[ChunkKey]: ...
    def neighbor_key(self, key: ChunkKey, direction: int) -> ChunkKey | None: ...
    def fetch(self, key: ChunkKey) -> LoadedChunk: ...
    def build_frame(
        self,
        chunks: list[LoadedChunk],
        request: ViewRequest,
        stream: StreamView,
    ) -> Any: ...


@dataclass(frozen=True)
class StreamCacheStats:
    loaded_chunk_count: int = 0
    loaded_bytes: int = 0
    pending_chunk_count: int = 0
    working_chunk_count: int = 0
    loaded_keys: tuple[ChunkKey, ...] = ()


class ChunkManager:
    """
    Threaded chunk cache shared by all adapters.

    State:
      - ``_loaded``: global LRU keyed by ``(stream_id, chunk_key)``.
      - ``_pending``: per-stream set of in-flight keys.
      - ``_working``: per-stream set of keys the current view depends on
        (visible + prefetch). Working chunks are sticky and never evicted.
      - ``_total_bytes``: rolling sum of ``LoadedChunk.nbytes`` in ``_loaded``.

    Eviction is byte-budget LRU: out-of-view chunks linger so back-and-forth
    scrubbing reuses them. They are dropped only when the budget is exceeded.
    """

    def __init__(
        self,
        adapters: list[StreamAdapter],
        *,
        max_workers: int = 2,
        prefetch_distance: int = 1,
        memory_budget_bytes: int | None = None,
    ):
        if not adapters:
            raise ValueError("adapters must not be empty")
        if memory_budget_bytes is not None and memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be > 0 or None")

        self._adapters: dict[str, StreamAdapter] = {}
        for adapter in adapters:
            if adapter.stream_id in self._adapters:
                raise ValueError(f"duplicate stream_id: {adapter.stream_id}")
            self._adapters[adapter.stream_id] = adapter

        self.stream_ids = list(self._adapters)
        self._stream_index = {sid: i for i, sid in enumerate(self.stream_ids)}
        self._prefetch_distance = max(0, int(prefetch_distance))
        self.memory_budget_bytes = memory_budget_bytes

        self._lock = threading.Lock()
        self._loaded: OrderedDict[tuple[str, ChunkKey], LoadedChunk] = OrderedDict()
        self._total_bytes = 0
        self._pending: dict[str, set[ChunkKey]] = {sid: set() for sid in self.stream_ids}
        self._working: dict[str, set[ChunkKey]] = {sid: set() for sid in self.stream_ids}
        self._queue: PriorityQueue[tuple[int, int, int, int, int, str, ChunkKey]] = PriorityQueue()
        self._sequence = count()
        self._generation = 0

        self._closed = threading.Event()
        worker_count = max(1, int(max_workers))
        self._workers = [
            threading.Thread(target=self._worker, name=f"chunk-cache-{i}", daemon=True)
            for i in range(worker_count)
        ]
        for worker in self._workers:
            worker.start()

    def close(self) -> None:
        self._closed.set()
        for worker in self._workers:
            worker.join(timeout=0.2)

    # -- introspection ------------------------------------------------------

    @property
    def loaded_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def loaded_chunk_count(self) -> int:
        with self._lock:
            return len(self._loaded)

    @property
    def pending_chunk_count(self) -> int:
        with self._lock:
            return sum(len(keys) for keys in self._pending.values())

    @property
    def working_chunk_count(self) -> int:
        with self._lock:
            return sum(len(keys) for keys in self._working.values())

    @property
    def prefetch_distance(self) -> int:
        return self._prefetch_distance

    def stream_debug_stats(self) -> dict[str, StreamCacheStats]:
        with self._lock:
            loaded_chunk_count = {sid: 0 for sid in self.stream_ids}
            loaded_bytes = {sid: 0 for sid in self.stream_ids}
            loaded_keys = {sid: [] for sid in self.stream_ids}

            for (stream_id, key), chunk in self._loaded.items():
                loaded_chunk_count[stream_id] += 1
                loaded_bytes[stream_id] += chunk.nbytes
                loaded_keys[stream_id].append(key)

            return {
                stream_id: StreamCacheStats(
                    loaded_chunk_count=loaded_chunk_count[stream_id],
                    loaded_bytes=loaded_bytes[stream_id],
                    pending_chunk_count=len(self._pending[stream_id]),
                    working_chunk_count=len(self._working[stream_id]),
                    loaded_keys=tuple(loaded_keys[stream_id]),
                )
                for stream_id in self.stream_ids
            }

    # -- public API ---------------------------------------------------------

    def clear(self) -> None:
        """Best-effort flush of loaded and queued chunks.

        In-flight fetches may still complete, but their results are dropped unless a
        subsequent update marks the same key as working again.
        """

        with self._lock:
            self._loaded.clear()
            self._total_bytes = 0
            for stream_id in self.stream_ids:
                self._pending[stream_id].clear()
                self._working[stream_id].clear()
            self._queue = PriorityQueue()
            self._generation += 1

    def update(self, request: ViewRequest) -> dict[str, StreamFrame]:
        streams = request.streams or tuple(StreamView(sid) for sid in self.stream_ids)

        plan: dict[str, tuple[StreamView, list[ChunkKey], list[ChunkKey]]] = {}
        for stream in streams:
            adapter = self._adapters[stream.stream_id]
            visible = list(dict.fromkeys(adapter.visible_keys(request, stream)))
            prefetch = self._prefetch_keys(adapter, request, visible)
            plan[stream.stream_id] = (stream, visible, prefetch)

        with self._lock:
            self._generation += 1
            generation = self._generation

            for stream_id, (_, visible, prefetch) in plan.items():
                self._working[stream_id] = set(visible) | set(prefetch)

            # Round-robin enqueue: order-0 across all streams, then order-1, ...
            for band, key_field in enumerate((1, 2)):  # 1=visible, 2=prefetch in plan tuple
                max_len = max((len(p[key_field]) for p in plan.values()), default=0)
                for order in range(max_len):
                    for stream_id, entry in plan.items():
                        keys = entry[key_field]
                        if order >= len(keys):
                            continue
                        key = keys[order]
                        composite = (stream_id, key)
                        if composite in self._loaded or key in self._pending[stream_id]:
                            continue
                        self._pending[stream_id].add(key)
                        self._queue.put((
                            -generation,
                            band,
                            order,
                            self._stream_index[stream_id],
                            next(self._sequence),
                            stream_id,
                            key,
                        ))

            # Collect ready visible chunks and refresh LRU recency.
            ready_chunks: dict[str, list[LoadedChunk]] = {}
            for stream_id, (_, visible, _) in plan.items():
                chunks: list[LoadedChunk] = []
                for key in visible:
                    composite = (stream_id, key)
                    chunk = self._loaded.get(composite)
                    if chunk is None:
                        continue
                    self._loaded.move_to_end(composite)
                    chunks.append(chunk)
                ready_chunks[stream_id] = chunks

        stream_frames: dict[str, StreamFrame] = {}
        for stream_id, (stream, visible, _) in plan.items():
            chunks = ready_chunks[stream_id]
            coverage = 1.0 if not visible else len(chunks) / len(visible)
            data = (
                self._adapters[stream_id].build_frame(chunks, request, stream)
                if chunks
                else None
            )
            stream_frames[stream_id] = StreamFrame(stream=stream, coverage=coverage, data=data)

        return stream_frames

    # -- internals ----------------------------------------------------------

    def _prefetch_keys(
        self,
        adapter: StreamAdapter,
        request: ViewRequest,
        visible: list[ChunkKey],
    ) -> list[ChunkKey]:
        if (
            self._prefetch_distance <= 0
            or request.jumped
            or not request.playing
            or not visible
        ):
            return []

        anchor = visible[-1] if request.direction >= 0 else visible[0]
        out: list[ChunkKey] = []
        seen = set(visible)
        cursor = anchor
        for _ in range(self._prefetch_distance):
            cursor = adapter.neighbor_key(cursor, request.direction)
            if cursor is None or cursor in seen:
                break
            seen.add(cursor)
            out.append(cursor)
        return out

    def _evict_to_budget(self) -> None:
        """Evict oldest non-working chunks until under the byte budget."""
        budget = self.memory_budget_bytes
        if budget is None or self._total_bytes <= budget:
            return

        for composite in list(self._loaded):
            if self._total_bytes <= budget:
                return
            stream_id, key = composite
            if key in self._working[stream_id]:
                continue
            chunk = self._loaded.pop(composite)
            self._total_bytes -= chunk.nbytes

    def _worker(self) -> None:
        while not self._closed.is_set():
            try:
                _, _, _, _, _, stream_id, key = self._queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                with self._lock:
                    composite = (stream_id, key)
                    if composite in self._loaded or key not in self._working[stream_id]:
                        continue
                    adapter = self._adapters[stream_id]

                chunk = adapter.fetch(key)
                if chunk.key != key:
                    raise ValueError(
                        f"adapter {stream_id!r} fetched {chunk.key!r} for {key!r}"
                    )

                with self._lock:
                    if key not in self._working[stream_id]:
                        continue
                    self._loaded[composite] = chunk
                    self._total_bytes += chunk.nbytes
                    self._evict_to_budget()
            finally:
                with self._lock:
                    self._pending[stream_id].discard(key)
                self._queue.task_done()
