from __future__ import annotations

import numpy as np

from .utils import ChunkConfig, ChunkKey, LoadedChunk, TimeRange, ViewRequest, split_chunk_key


class DenseSignalAdapter:
    """
    Adapter for regularly sampled arrays stored as ``(channels, samples)``.

    Chunk boundaries are derived from ``target_chunk_bytes``, and materialized
    data contains ``times`` and ``values`` arrays for the requested view. The
    input array is kept as a view and each loaded chunk copies values into a
    compact in-memory buffer on demand.
    """

    def __init__(
        self,
        stream_id: str,
        data: np.ndarray,
        sample_rate_hz: float,
        *,
        start_time: float = 0.0,
        target_chunk_bytes: int = 8 * 1024 * 1024,
    ):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError("data must have shape (channels, samples) or (samples,)")
        if data.shape[1] == 0:
            raise ValueError("data must contain at least one sample")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("data dtype must be numeric")

        self.stream_id = str(stream_id)
        self.data = data
        self.sample_rate_hz = float(sample_rate_hz)
        self.start_time = float(start_time)
        self.chunk_config = ChunkConfig(target_bytes=int(target_chunk_bytes))

        self.channel_count, self.sample_count = self.data.shape
        self.t_min = self.start_time
        self.t_max = self.start_time + max(self.sample_count - 1, 0) / self.sample_rate_hz

        per_sample_bytes = self.channel_count * self.data.dtype.itemsize + np.dtype(np.float64).itemsize
        self.chunk_samples = max(1, self.chunk_config.target_bytes // per_sample_bytes)
        self.chunk_key_width = max(5, len(str(self.chunk_count() - 1)))

    def chunk_count(self) -> int:
        return max(1, (self.sample_count + self.chunk_samples - 1) // self.chunk_samples)

    def clamp_chunk_index(self, chunk_index: int) -> int:
        return int(np.clip(chunk_index, 0, self.chunk_count() - 1))

    def require_chunk_index(self, chunk_index: int) -> int:
        if not 0 <= int(chunk_index) < self.chunk_count():
            raise ValueError(f"chunk index {chunk_index} is out of range")
        return int(chunk_index)

    def time_to_sample_index(self, timestamp: float) -> int:
        sample = int(round((float(timestamp) - self.start_time) * self.sample_rate_hz))
        return int(np.clip(sample, 0, self.sample_count - 1))

    def time_to_chunk_index(self, timestamp: float) -> int:
        sample = self.time_to_sample_index(timestamp)
        return self.clamp_chunk_index(sample // self.chunk_samples)

    def key_for_index(self, chunk_index: int) -> ChunkKey:
        chunk_index = self.require_chunk_index(chunk_index)
        return f"{self.stream_id}/{chunk_index:0{self.chunk_key_width}d}"

    def chunk_index_for_key(self, key: ChunkKey) -> int:
        stream_id, chunk_suffix = split_chunk_key(key)
        if stream_id != self.stream_id:
            raise ValueError(f"expected key for {self.stream_id!r}, got {key!r}")
        try:
            chunk_index = int(chunk_suffix)
        except ValueError as error:
            raise ValueError(f"invalid chunk key {key!r}") from error
        return self.require_chunk_index(chunk_index)

    def chunk_time_bounds(self, chunk_index: int) -> tuple[float, float, int, int]:
        chunk_index = self.require_chunk_index(chunk_index)
        start = chunk_index * self.chunk_samples
        stop = min(self.sample_count, start + self.chunk_samples)
        t0 = self.start_time + start / self.sample_rate_hz
        t1 = self.start_time + (stop - 1) / self.sample_rate_hz
        return float(t0), float(t1), start, stop

    def keys_for_range(self, view: TimeRange) -> list[ChunkKey]:
        start_index = self.time_to_chunk_index(view.start)
        end_index = self.time_to_chunk_index(np.nextafter(view.end, -np.inf))
        return [self.key_for_index(index) for index in range(start_index, end_index + 1)]

    def visible_keys(self, request: ViewRequest) -> list[ChunkKey]:
        visible_keys = self.keys_for_range(request.view)
        if not visible_keys:
            visible_keys = [self.key_for_index(self.time_to_chunk_index(request.cursor_t))]
        return visible_keys

    def neighbor_key(self, key: ChunkKey, direction: int) -> ChunkKey | None:
        chunk_index = self.chunk_index_for_key(key)
        next_index = chunk_index + (1 if direction >= 0 else -1)
        if not 0 <= next_index < self.chunk_count():
            return None
        return self.key_for_index(next_index)

    def fetch(self, key: ChunkKey) -> LoadedChunk:
        chunk_index = self.chunk_index_for_key(key)
        t0, t1, start, stop = self.chunk_time_bounds(chunk_index)

        times = self.start_time + np.arange(start, stop, dtype=np.float64) / self.sample_rate_hz
        values = np.ascontiguousarray(self.data[:, start:stop])
        data = {
            "times": times,
            "values": values,
        }
        return LoadedChunk(
            key=key,
            stream_id=self.stream_id,
            t0=t0,
            t1=t1,
            nbytes=int(data["times"].nbytes + values.nbytes),
            data=data,
        )

    def build_view(
        self,
        chunks: list[LoadedChunk],
        *,
        view: TimeRange,
        width_px: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        if not chunks:
            return None
        if width_px is not None and width_px <= 0:
            raise ValueError("width_px must be > 0 when provided")

        ordered_chunks = sorted(chunks, key=lambda chunk: self.chunk_index_for_key(chunk.key))
        times = np.concatenate([chunk.data["times"] for chunk in ordered_chunks])
        values = np.concatenate([chunk.data["values"] for chunk in ordered_chunks], axis=1)
        mask = (times >= view.start) & (times <= view.end)

        return {
            "times": times[mask],
            "values": values[:, mask],
        }