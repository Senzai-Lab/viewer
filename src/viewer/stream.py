from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from zarr import Array

class Stream(Protocol):
    name: str
    kind: str

    t_min: float
    t_max: float
    n_chunks: int

    # values: Array # some kind of nd array
    # ts: Array # 1D array of timestamps coming from hardware

    def chunk_at(self, t: float) -> int: ...
    def load_chunk(self, chunk_idx: int) -> dict: ...


class TimeSeries(Stream):
    """Continuous signal: (n_samples, n_channels) values array and timestamps."""
    def __init__(
            self,
            name: str,
            values: Array,
            ts: Array,
            fs: float,
            *,
            chunk_samples: int | None = None,
        ):
        self.kind = 'timeseries'
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
            't_stop': float(self.ts[stop-1]),
            'n_bytes': self.chunk_nbytes,
            'data': data,
        }

class Units(Stream):
    """Irregular sampled (spike_times, spike_units) arrays"""
    def __init__(
        self,
        name: str,
        ts: Array,
        values: Array,
        metadata: dict,
        *,
        chunk_duration: float = 10.0,
    ):
        self.kind = 'units'
        self.name = name
        self.metadata = metadata # nested dict with unit id maps

        self.values = values # spike units (N,) unique integers
        self.ts = ts # spike times (N,) monotonic timestamps

        self._n_spikes = int(ts.shape[0])
        self.t_min = float(ts[0])
        self.t_max = float(ts[-1])
        self.chunk_duration = chunk_duration

        self.n_chunks = int(np.ceil((self.t_max - self.t_min) / self.chunk_duration))

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min
    
    def chunk_at(self, t: float) -> int:
        idx = int((t - self.t_min) / self.chunk_duration)
        return max(0, min(idx, self.n_chunks - 1))
    
    def load_chunk(self, chunk_idx: int) -> dict:
        t0 = self.t_min + chunk_idx * self.chunk_duration
        t1 = t0 + self.chunk_duration

        i0 = int(np.searchsorted(self.ts, t0, side='left'))
        i1 = int(np.searchsorted(self.ts, t1, side='left'))
        
        times = np.asarray(self.ts[i0:i1])
        units = np.asarray(self.values[i0:i1])

        return {
            't_start': t0,
            't_stop': t1,
            'ts': times,
            'data': units,
            'n_bytes': times.nbytes + units.nbytes
        }
