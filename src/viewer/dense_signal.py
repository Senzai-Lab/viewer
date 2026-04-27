"""Adapter for regularly sampled in-memory dense arrays."""

from __future__ import annotations

import numpy as np

from .core import (
    NS_PER_S,
    ChunkKey,
    LoadedChunk,
    StreamView,
    TimeRange,
    ViewRequest,
)


class DenseSignalAdapter:
    """
    Dense ``(channels, samples)`` signal adapter.

    The source array is held as a view; ``fetch()`` materializes a compact
    contiguous slice for the renderer. Chunk size is derived from the target
    byte budget, and ``build_frame()`` does width-based stride downsampling.
    """

    def __init__(
        self,
        stream_id: str,
        data: np.ndarray,
        sample_rate_hz: float,
        *,
        start_time_ns: int = 0,
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
        if target_chunk_bytes <= 0:
            raise ValueError("target_chunk_bytes must be > 0")

        self.stream_id = str(stream_id)
        self.data = data
        self.sample_rate_hz = float(sample_rate_hz)
        self.start_time_ns = int(start_time_ns)
        self.sample_period_ns = NS_PER_S / self.sample_rate_hz

        self.channel_count, self.sample_count = self.data.shape
        per_sample_bytes = (
            self.channel_count * self.data.dtype.itemsize + np.dtype(np.int64).itemsize
        )
        self.chunk_samples = max(1, int(target_chunk_bytes) // per_sample_bytes)
        self._chunk_count = max(
            1, (self.sample_count + self.chunk_samples - 1) // self.chunk_samples
        )
        self._key_width = max(5, len(str(self._chunk_count - 1)))

        self.time_range = TimeRange(
            self.start_time_ns,
            self._sample_to_ns(self.sample_count),
        )

    # -- index <-> time helpers ---------------------------------------------

    def _sample_to_ns(self, sample_index: int) -> int:
        return self.start_time_ns + int(round(int(sample_index) * self.sample_period_ns))

    def _ns_to_sample(self, timestamp_ns: int) -> int:
        sample = int(round((int(timestamp_ns) - self.start_time_ns) / self.sample_period_ns))
        return max(0, min(self.sample_count - 1, sample))

    def _ns_to_chunk(self, timestamp_ns: int) -> int:
        return self._ns_to_sample(timestamp_ns) // self.chunk_samples

    def _key(self, chunk_index: int) -> ChunkKey:
        return f"{self.stream_id}/{chunk_index:0{self._key_width}d}"

    def _index_for_key(self, key: ChunkKey) -> int:
        prefix, _, suffix = key.rpartition("/")
        if prefix != self.stream_id:
            raise ValueError(f"key {key!r} does not belong to {self.stream_id!r}")
        index = int(suffix)
        if not 0 <= index < self._chunk_count:
            raise ValueError(f"chunk index {index} out of range")
        return index

    # -- StreamAdapter protocol ---------------------------------------------

    def visible_keys(self, request: ViewRequest, stream: StreamView) -> list[ChunkKey]:
        view = request.time
        start_index = self._ns_to_chunk(view.start_ns)
        end_anchor = view.start_ns if view.stop_ns <= view.start_ns else view.stop_ns - 1
        end_index = self._ns_to_chunk(end_anchor)
        return [self._key(i) for i in range(start_index, end_index + 1)]

    def neighbor_key(self, key: ChunkKey, direction: int) -> ChunkKey | None:
        next_index = self._index_for_key(key) + (1 if direction >= 0 else -1)
        if not 0 <= next_index < self._chunk_count:
            return None
        return self._key(next_index)

    def fetch(self, key: ChunkKey) -> LoadedChunk:
        chunk_index = self._index_for_key(key)
        start = chunk_index * self.chunk_samples
        stop = min(self.sample_count, start + self.chunk_samples)

        sample_indices = np.arange(start, stop, dtype=np.float64)
        time_ns = self.start_time_ns + np.rint(
            sample_indices * self.sample_period_ns
        ).astype(np.int64)
        values = np.ascontiguousarray(self.data[:, start:stop])

        return LoadedChunk(
            key=key,
            time=TimeRange(self._sample_to_ns(start), self._sample_to_ns(stop)),
            data={"time_ns": time_ns, "values": values},
            nbytes=int(time_ns.nbytes + values.nbytes),
        )

    def build_frame(
        self,
        chunks: list[LoadedChunk],
        request: ViewRequest,
        stream: StreamView,
    ) -> dict[str, np.ndarray] | None:
        if not chunks:
            return None

        ordered = sorted(chunks, key=lambda c: c.time.start_ns)
        time_ns = np.concatenate([c.data["time_ns"] for c in ordered])
        values = np.concatenate([c.data["values"] for c in ordered], axis=1)

        mask = (time_ns >= request.time.start_ns) & (time_ns < request.time.stop_ns)
        if not np.any(mask):
            return None

        time_ns = time_ns[mask]
        values = values[:, mask]

        if stream.selection.channels is None:
            channel_indices = np.arange(self.channel_count, dtype=np.int64)
        else:
            channel_indices = np.asarray(stream.selection.channels, dtype=np.int64)
        values = values[channel_indices]

        time_ns, values = _stride_downsample(time_ns, values, request.width_px)
        return {
            "time_ns": time_ns,
            "values": values,
            "channel_indices": channel_indices,
        }


def _stride_downsample(
    time_ns: np.ndarray,
    values: np.ndarray,
    width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    if width_px <= 0 or time_ns.size <= width_px:
        return time_ns, values
    stride = max(1, (time_ns.size + width_px - 1) // width_px)
    indices = np.arange(0, time_ns.size, stride, dtype=np.int64)
    if indices[-1] != time_ns.size - 1:
        indices = np.append(indices, time_ns.size - 1)
    return time_ns[indices], values[:, indices]
