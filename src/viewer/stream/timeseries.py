from __future__ import annotations

import math
from typing import Any

import numpy as np


class TimeSeries:
    """Continuous signal: (n_samples, n_channels) values array and timestamps."""

    def __init__(
            self,
            name: str,
            values: Any,
            ts: Any,
            fs: float,
            *,
            chunk_samples: int,
    ):
        self.name = name
        self.values = values
        self.ts = ts

        self.fs = float(fs)
        self.t_min = float(ts[0])
        self.t_max = float(ts[-1])
        self.duration = self.t_max - self.t_min

        self.n_samples = values.shape[0]
        self.n_channels = values.shape[1] if values.ndim > 1 else 1
        self.chunk_samples = int(chunk_samples)
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self.n_chunks = -(-self.n_samples // self.chunk_samples)

        self.chunk_nbytes = self.chunk_samples * self.n_channels * values.dtype.itemsize

    def iter_visible(self, chunks, t0: float, t1: float, width_px: float):
        stride = max(1, math.floor(self.fs * (t1 - t0) / max(width_px, 1)))

        for chunk in chunks:
            chunk_t0 = chunk["t_start"]
            data = chunk["data"]
            n = data.shape[0]

            i0 = max(0, math.floor((t0 - chunk_t0) * self.fs) - 1)
            i1 = min(n, math.ceil((t1 - chunk_t0) * self.fs) + 1)

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
        sample_idx = math.floor((t - self.t_min) * self.fs)
        chunk_idx = sample_idx // self.chunk_samples
        return max(0, min(chunk_idx, self.n_chunks - 1))

    def load_chunk(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)

        data = np.asarray(self.values[start:stop])
        if data.ndim == 1:
            data = data[:, np.newaxis]

        return {
            "t_start": self.t_min + start / self.fs,
            "sample_start": start,
            "sample_stop": stop,
            "n_bytes": data.nbytes,
            "data": data,
        }
