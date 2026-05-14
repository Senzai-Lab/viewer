from __future__ import annotations

import math
from typing import Any

import numpy as np


class Units:
    """Irregular sampled (spike_times, spike_units) arrays."""

    def __init__(
        self,
        name: str,
        ts: Any,
        values: Any,
        metadata: dict | None = None,
        *,
        chunk_duration: float = 10.0,
        unit_ids=None,
    ):
        self.name = name
        raw_metadata = dict(metadata or {})

        self.values = values
        # Load spike timestamps for fast searchsorted.
        self.ts = np.asarray(ts)

        self._n_spikes = len(self.ts)
        self.t_min = float(self.ts[0])
        self.t_max = float(self.ts[-1])
        self.chunk_duration = chunk_duration

        self.n_chunks = max(1, math.ceil((self.t_max - self.t_min) / self.chunk_duration))
        spikes_per_chunk = math.ceil(self._n_spikes / self.n_chunks)
        self.chunk_nbytes = spikes_per_chunk * (ts.dtype.itemsize + values.dtype.itemsize)

        ids = unit_ids
        if ids is None:
            ids = raw_metadata.pop("unit_ids", None)
        if ids is None:
            ids = raw_metadata["rate"].keys() if "rate" in raw_metadata else np.unique(values[:])

        self.unit_ids = [int(uid) for uid in ids]
        self.n_units = len(self.unit_ids)
        self.metadata = {
            key: np.array([values[str(uid)] for uid in self.unit_ids], dtype=float)
            for key, values in raw_metadata.items()
        }
        self.metadata_keys = list(self.metadata)

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min

    def iter_visible(self, chunks, t0: float, t1: float, width_px: float):
        n_bins = max(1, int(width_px))
        bin_width = (t1 - t0) / n_bins

        for chunk in chunks:
            times = chunk["ts"]
            unit_ids = chunk["data"]

            i0 = np.searchsorted(times, t0, side="left")
            i1 = np.searchsorted(times, t1, side="right")
            if i0 >= i1:
                continue

            visible_t = times[i0:i1]
            visible_u = unit_ids[i0:i1]

            bins = ((visible_t - t0) / bin_width).astype(int)
            np.clip(bins, 0, n_bins - 1, out=bins)

            # Use bins only to reduce overdraw; draw selected spikes at their
            # original timestamps so playback does not quantize them to a moving grid.
            codes = visible_u * n_bins + bins
            _, keep = np.unique(codes, return_index=True)

            yield (
                visible_t[keep],
                visible_u[keep],
            )

    def chunk_at(self, t: float) -> int:
        idx = math.floor((t - self.t_min) / self.chunk_duration)
        return max(0, min(idx, self.n_chunks - 1))

    def load_chunk(self, chunk_idx: int) -> dict:
        t0 = self.t_min + chunk_idx * self.chunk_duration
        t1 = t0 + self.chunk_duration

        i0 = np.searchsorted(self.ts, t0, side="left")
        i1 = np.searchsorted(self.ts, t1, side="left")

        times = self.ts[i0:i1]
        units = np.asarray(self.values[i0:i1], dtype=int)

        return {
            "t_start": t0,
            "t_stop": t1,
            "ts": times,
            "data": units,
            "n_bytes": times.nbytes + units.nbytes,
        }
