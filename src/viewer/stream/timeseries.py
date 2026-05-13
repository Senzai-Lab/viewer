from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zarr import Array


class TimeSeries:
    """Continuous signal: (n_samples, n_channels) values array and timestamps."""
    kind = "timeseries"

    def __init__(
            self,
            name: str,
            values: Array,
            ts: Array,
            fs: float,
            *,
            chunk_samples: int | None = None,
    ):
        self.name = name
        self.t_min = float(ts[0])
        self.t_max = float(ts[-1])
        self.values = values
        self.ts = ts

        self.fs = fs
        self.n_samples = values.shape[0]
        self.n_channels = values.shape[1] if values.ndim > 1 else 1
        self.chunk_samples = chunk_samples if chunk_samples is not None else values.chunks[0]
        self.n_chunks = -(-self.n_samples // self.chunk_samples)

        self.chunk_nbytes = self.chunk_samples * (
            self.n_channels * values.dtype.itemsize + ts.dtype.itemsize
        )

        chunk_start_idx = np.arange(0, self.n_samples, self.chunk_samples)
        self.chunk_times = np.asarray(ts[chunk_start_idx])

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min

    def view(self, chunks, t0: float, t1: float, width_px: float):
        stride = max(1, int(self.fs * (t1 - t0) / max(width_px, 1)))

        for chunk in chunks:
            chunk_t0 = chunk["t_start"]
            data = chunk["data"]
            n = data.shape[0]

            i0 = max(0, int(np.floor((t0 - chunk_t0) * self.fs)) - 1)
            i1 = min(n, int(np.ceil((t1 - chunk_t0) * self.fs)) + 1)

            if i0 >= i1:
                continue

            if stride > 1:
                offset = (stride - ((chunk["sample_start"] + i0) % stride)) % stride
                i0 += offset
                if i0 >= i1:
                    continue

            yield (
                data[i0:i1:stride],
                chunk_t0 + i0 / self.fs,
                stride / self.fs,
            )

    def chunk_at(self, t: float) -> int:
        idx = np.searchsorted(self.chunk_times, t, side="right")
        return max(int(idx) - 1, 0)

    def load_chunk(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)

        data = np.asarray(self.values[start:stop])
        if data.ndim == 1:
            data = data[:, np.newaxis]

        return {
            "t_start": float(self.ts[start]),
            "sample_start": start,
            "sample_stop": stop,
            "n_bytes": self.chunk_nbytes,
            "data": data,
        }
