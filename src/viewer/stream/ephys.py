from __future__ import annotations

from typing import Any

import math
import numpy as np


class Ephys:
    """Sample-clocked high-density electrophysiology data with probe geometry."""

    def __init__(
        self,
        name: str,
        values: Any,
        geometry: dict,
        *,
        fs: float = 30_000,
        chunk_samples: int | None = None,
        scale: float = 1.0,
        offset: float = 0.0,
        units: str = "a.u.",
    ):
        self.name = name
        self.values = values
        self.ts = None

        self.fs = float(fs)
        self.n_samples = values.shape[0]
        self.n_channels = values.shape[1]
        self.geometry = geometry

        self.t_min = 0.0
        self.t_max = (self.n_samples - 1) / self.fs
        self.duration = self.t_max - self.t_min

        if chunk_samples is None:
            chunk_samples = int(5 * self.fs)
        self.chunk_samples = int(chunk_samples)

        self.n_chunks = -(-self.n_samples // self.chunk_samples)
        self.source_dtype = values.dtype
        self.dtype = np.result_type(self.source_dtype, np.float32)
        self.chunk_nbytes = self.chunk_samples * self.n_channels * self.dtype.itemsize

        self.scale = float(scale)
        self.offset = float(offset)
        self.units = units

    def chunk_at(self, t: float) -> int:
        sample_idx = math.floor(t * self.fs)
        chunk_idx = sample_idx // self.chunk_samples
        return max(0, min(chunk_idx, self.n_chunks - 1))

    def load_chunk(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)

        data = np.array(self.values[start:stop], dtype=self.dtype, copy=True, order="F")
        data *= self.scale
        data += self.offset

        return {
            "sample_start": start,
            "sample_stop": stop,
            "n_bytes": data.nbytes,
            "data": data,
        }

    def iter_visible_channels(
        self,
        chunks,
        t0: float,
        t1: float,
        width_px: float,
        channel_indices: np.ndarray,
        *,
        envelope_threshold: float = 2.0,
    ):
        """
        Yield raw samples or min/max envelopes for selected channels.
        """
        samples_per_px = self.fs * (t1 - t0) / width_px

        if samples_per_px <= envelope_threshold:
            yield from self._iter_raw(chunks, t0, t1)
            return

        samples_per_bin = math.ceil(samples_per_px)
        yield from self._iter_envelope(
            chunks,
            t0,
            t1,
            channel_indices,
            samples_per_bin,
        )

    def _iter_raw(self, chunks, t0: float, t1: float):
        for chunk in chunks:
            chunk_start = chunk["sample_start"]
            n = chunk["data"].shape[0]

            i0 = math.floor(t0 * self.fs) - chunk_start - 1
            i1 = math.ceil(t1 * self.fs) - chunk_start + 2
            i0 = max(0, i0)
            i1 = min(n, i1)

            if i0 >= i1:
                continue

            sample_start = chunk["sample_start"] + i0
            yield {
                "mode": "raw",
                "data": chunk["data"][i0:i1],
                "sample_start": sample_start,
                "sample_stop": chunk["sample_start"] + i1,
                "dt": 1.0 / self.fs,
            }

    def _iter_envelope(
        self,
        chunks,
        t0: float,
        t1: float,
        channel_indices: np.ndarray,
        samples_per_bin: int,
    ):
        visible_start = math.floor(t0 * self.fs)
        visible_stop = math.ceil(t1 * self.fs) + 1

        runs = []
        run_start = None
        run_stop = None
        parts = []
        for chunk in chunks:
            chunk_sample_start = chunk["sample_start"]
            n = chunk["data"].shape[0]

            i0 = visible_start - chunk_sample_start
            i1 = visible_stop - chunk_sample_start
            i0 = max(0, i0)
            i1 = min(n, i1)

            if i0 >= i1:
                continue

            global_start = chunk_sample_start + i0
            global_stop = chunk_sample_start + i1

            if parts and global_start != run_stop:
                runs.append((run_start, run_stop, parts))
                run_start = None
                parts = []

            if run_start is None:
                run_start = global_start
            run_stop = global_stop

            parts.append(
                np.asfortranarray(chunk["data"][i0:i1, channel_indices])
            )

        if parts:
            runs.append((run_start, run_stop, parts))

        for run_start, run_stop, parts in runs:
            data = np.asfortranarray(np.concatenate(parts, axis=0))
            first_bin = (run_start // samples_per_bin) * samples_per_bin
            bin_starts = np.arange(first_bin, run_stop, samples_per_bin)
            starts = np.maximum(bin_starts, run_start)
            stops = np.minimum(bin_starts + samples_per_bin, run_stop)

            block_start = starts[0] - run_start
            block_stop = stops[-1] - run_start
            block = data[block_start:block_stop]
            offsets = starts - starts[0]

            # reduceat starts one reduction at each offset and stops at the
            # next offset, so each output row is one envelope bin.
            y_min = np.minimum.reduceat(block, offsets, axis=0)
            y_max = np.maximum.reduceat(block, offsets, axis=0)
            times = 0.5 * (starts + stops - 1) / self.fs

            yield {
                "mode": "envelope",
                "t": times,
                "y_min": np.asfortranarray(y_min),
                "y_max": np.asfortranarray(y_max),
                "samples_per_bin": samples_per_bin,
            }
