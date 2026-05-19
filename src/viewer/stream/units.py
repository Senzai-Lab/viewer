from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import BaseStream


class Spikes(BaseStream):
    """Irregular sampled spike times with per-spike unit labels."""

    def __init__(
        self,
        name: str,
        ts: Any,
        spike_units: Any,
        *,
        chunk_duration: float = 10.0,
        unit_ids=None,
    ):
        self.name = name
        self.spike_units = spike_units
        # Load spike timestamps for fast searchsorted.
        self.ts = np.asarray(ts)

        self._n_spikes = len(self.ts)
        self.t_min = float(self.ts[0])
        self.t_max = float(self.ts[-1])
        self.duration = self.t_max - self.t_min

        self.chunk_duration = chunk_duration
        self.n_chunks = max(1, math.ceil((self.t_max - self.t_min) / self.chunk_duration))
        spikes_per_chunk = math.ceil(self._n_spikes / self.n_chunks)
        self.chunk_nbytes = spikes_per_chunk * (
            self.ts.dtype.itemsize + spike_units.dtype.itemsize
        )

        ids = unit_ids
        if ids is None:
            ids = np.unique(spike_units[:])

        self.unit_ids = np.asarray([int(uid) for uid in ids], dtype=np.int64)
        self.n_units = len(self.unit_ids)

    def iter_visible(self, chunks, t0: float, t1: float, width_px: float):
        if t1 <= t0:
            return

        n_bins = max(1, int(width_px))
        bin_scale = n_bins / (t1 - t0)

        for chunk in chunks:
            times = chunk["ts"]
            unit_ids = chunk["data"]

            i0 = np.searchsorted(times, t0, side="left")
            i1 = np.searchsorted(times, t1, side="left")
            if i0 >= i1:
                continue

            times = times[i0:i1]
            unit_ids = unit_ids[i0:i1]

            bins = ((times - t0) * bin_scale).astype(np.int64)
            np.minimum(bins, n_bins - 1, out=bins)

            codes = unit_ids * n_bins + bins
            keep = np.unique(codes, return_index=True)[1]

            yield (
                times[keep],
                unit_ids[keep],
            )

    def chunk_at(self, t: float) -> int:
        idx = math.floor((t - self.t_min) / self.chunk_duration)
        return max(0, min(idx, self.n_chunks - 1))

    def read(self, chunk_idx: int) -> dict:
        t0 = self.t_min + chunk_idx * self.chunk_duration
        t1 = t0 + self.chunk_duration

        i0 = np.searchsorted(self.ts, t0, side="left")
        i1 = np.searchsorted(self.ts, t1, side="left")

        times = self.ts[i0:i1]
        units = np.asarray(self.spike_units[i0:i1], dtype=np.int64)

        return {
            "t_start": t0,
            "t_stop": t1,
            "ts": times,
            "data": units,
            "nbytes": times.nbytes + units.nbytes,
        }
