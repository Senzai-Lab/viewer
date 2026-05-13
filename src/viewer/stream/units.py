from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zarr import Array


class Units:
    """Irregular sampled (spike_times, spike_units) arrays."""
    kind = "units"

    def __init__(
        self,
        name: str,
        values: Array,
        ts: Array,
        metadata: dict,
        *,
        chunk_duration: float = 10.0,
    ):
        self.name = name
        self.metadata = metadata

        self.values = values
        self.ts = ts

        self._n_spikes = int(ts.shape[0])
        self.t_min = float(ts[0])
        self.t_max = float(ts[-1])
        self.chunk_duration = chunk_duration

        self.n_chunks = max(1, int(np.ceil((self.t_max - self.t_min) / self.chunk_duration)))
        spikes_per_chunk = int(np.ceil(self._n_spikes / self.n_chunks))
        self.chunk_nbytes = spikes_per_chunk * (ts.dtype.itemsize + values.dtype.itemsize)
        self.unit_ids = list(metadata["rate"].keys())
        self.n_units = len(self.unit_ids)

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min

    def view(self, chunks, t0: float, t1: float, width_px: float):
        n_bins = max(1, int(width_px))
        bin_width = (t1 - t0) / n_bins
        if bin_width <= 0:
            return

        for chunk in chunks:
            times = chunk["ts"]
            unit_ids = chunk["data"]
            if times.size == 0:
                continue

            i0 = int(np.searchsorted(times, t0, side="left"))
            i1 = int(np.searchsorted(times, t1, side="right"))
            if i0 >= i1:
                continue

            visible_t = times[i0:i1]
            visible_u = unit_ids[i0:i1]

            bins = ((visible_t - t0) / bin_width).astype(np.int32)
            np.clip(bins, 0, n_bins - 1, out=bins)

            pairs = np.column_stack((visible_u, bins))
            pairs = np.unique(pairs, axis=0)

            yield (
                t0 + (pairs[:, 1] + 0.5) * bin_width,
                pairs[:, 0],
            )

    def chunk_at(self, t: float) -> int:
        idx = int((t - self.t_min) / self.chunk_duration)
        return max(0, min(idx, self.n_chunks - 1))

    def load_chunk(self, chunk_idx: int) -> dict:
        t0 = self.t_min + chunk_idx * self.chunk_duration
        t1 = t0 + self.chunk_duration

        i0 = int(np.searchsorted(self.ts, t0, side="left"))
        i1 = int(np.searchsorted(self.ts, t1, side="left"))

        times = np.asarray(self.ts[i0:i1])
        units = np.asarray(self.values[i0:i1])

        return {
            "t_start": t0,
            "t_stop": t1,
            "ts": times,
            "data": units,
            "n_bytes": times.nbytes + units.nbytes,
        }
