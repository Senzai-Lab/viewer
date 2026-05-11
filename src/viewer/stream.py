from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    from zarr import Array

class Stream(ABC):
    name: str
    ts: Array
    n_chunks: int
    chunk_nbytes: int

    @abstractmethod
    def chunk_at(self, t: float) -> int: ...

    @abstractmethod
    def load_chunk(self, chunk_idx: int) -> dict: ...


class Ephys(Stream):
    """(n_samples, n_channels) zarr array + timestamps array (n_samples, )"""
    def __init__(
            self,
            name: str,
            values: Array,
            ts: Array,
            fs: float,
            *,
            chunk_samples: int | None = None,
        ):
        self.kind = 'ephys'
        self.name = name
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
        return self.ts[-1] - self.ts[0]

    def chunk_at(self, t: float) -> int:
        idx = np.searchsorted(self.chunk_times, t, side='right')
        return max(int(idx) - 1, 0)

    def load_chunk(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)

        data = np.asarray(self.values[start:stop])
        if data.ndim == 1:
            data = data[:, np.newaxis]
        
        t_start = self.ts[start]

        return {
            't_start': float(t_start),
            'data': data,
        }