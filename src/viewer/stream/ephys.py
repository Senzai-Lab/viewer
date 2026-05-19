from __future__ import annotations

from typing import Any

import math
import numpy as np

from .base import chunks_in_span, read_key
from viewer.span import Span


class Ephys:
    """Sample-clocked high-density electrophysiology data with probe geometry."""

    def __init__(
        self,
        name: str,
        values: Any,
        geometry: dict,
        *,
        fs: float = 1250.0,
        chunk_samples: int | None = None,
        scale: float = 1.0,
        offset: float = 0.0,
        units: str = "a.u.",
    ):
        self.name = name
        self.values = values
        self.ts = None
        self.source = self
        self.transforms = None

        self.source_fs = float(fs)
        self.fs = self.source_fs
        self.n_samples = values.shape[0]
        self.source_n_channels = values.shape[1]
        self.n_channels = self.source_n_channels
        self.geometry = geometry

        self.t_min = 0.0
        self.t_max = (self.n_samples - 1) / self.source_fs
        self.duration = self.t_max - self.t_min

        if chunk_samples is None:
            chunk_samples = int(5 * self.source_fs)
        self.chunk_samples = int(chunk_samples)

        self.n_chunks = -(-self.n_samples // self.chunk_samples)
        self.source_dtype = values.dtype
        self.dtype = np.dtype(np.result_type(self.source_dtype, np.float32))
        self.y = None
        self.chunk_nbytes = 0

        self.scale = float(scale)
        self.offset = float(offset)
        self.units = units
        self.setup_pipe()

    def pipe(self, *transforms, name: str) -> "Ephys":
        from viewer.transforms import Compose

        stream = Ephys(
            name,
            self.values,
            self.geometry,
            fs=self.source_fs,
            chunk_samples=self.chunk_samples,
            scale=self.scale,
            offset=self.offset,
            units=self.units,
        )
        stream.source = self.source
        stream.transforms = (
            Compose(*transforms)
            if self.transforms is None
            else Compose(self.transforms, *transforms)
        )
        stream.setup_pipe()
        return stream

    def setup_pipe(self):
        self.fs = self.source_fs
        self.n_channels = self.source_n_channels
        self.dtype = np.dtype(np.result_type(self.source_dtype, np.float32))
        self.y = None
        self.chunk_nbytes = (
            self.chunk_samples * self.n_channels * np.dtype(self.dtype).itemsize
        )

        if self.transforms is None:
            return

        setup = getattr(self.transforms, "setup", None)
        meta = setup(self) if setup is not None else {}
        meta = {} if meta is None else meta

        self.fs = float(meta.get("fs", self.fs))
        self.n_channels = int(meta.get("n_channels", self.n_channels))
        if "dtype" in meta:
            self.dtype = np.dtype(meta["dtype"])
        self.y = meta.get("y", self.y)

        output_nbytes = getattr(self.transforms, "output_nbytes", None)
        if "chunk_nbytes" in meta:
            self.chunk_nbytes = int(meta["chunk_nbytes"])
        elif output_nbytes is not None:
            self.chunk_nbytes = int(output_nbytes(self, self.chunk_samples))
        else:
            self.chunk_nbytes = (
                self.chunk_samples * self.n_channels * np.dtype(self.dtype).itemsize
            )

    def chunk_at(self, t: float) -> int:
        sample_idx = math.floor(t * self.source_fs)
        chunk_idx = sample_idx // self.chunk_samples
        return max(0, min(chunk_idx, self.n_chunks - 1))

    @property
    def span(self) -> Span:
        return Span(self.t_min, self.t_max)

    def chunks_in(self, span: Span) -> range:
        return chunks_in_span(self, span)

    def at(self, t: float) -> dict:
        return self.read(self.chunk_at(t))

    def in_span(self, span: Span) -> list[dict]:
        return [self.read(i) for i in self.chunks_in(span)]

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return read_key(self, key)

    def read(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)
        pad = 0
        if self.transforms is not None:
            pad = math.ceil(float(getattr(self.transforms, "pad_s", 0.0)) * self.source_fs)
        read_start = max(0, start - pad)
        read_stop = min(self.n_samples, stop + pad)

        data = np.array(self.values[read_start:read_stop], dtype=self.dtype, copy=True, order="F")
        data *= self.scale
        data += self.offset

        if self.transforms is not None:
            ctx = {
                "stream": self,
                "source_fs": self.source_fs,
                "source_t_min": self.t_min,
                "request_start": start,
                "request_stop": stop,
                "read_start": read_start,
                "read_stop": read_stop,
            }
            output = self.transforms(data, ctx)
            return self._chunk_from_transform(output, start, stop, read_start, read_stop)

        return {
            "t_start": self.t_min + start / self.source_fs,
            "t_stop": self.t_min + stop / self.source_fs,
            "sample_start": start,
            "sample_stop": stop,
            "fs": self.source_fs,
            "dt": 1.0 / self.source_fs,
            "nbytes": data.nbytes,
            "data": data,
        }

    def _chunk_from_transform(
        self,
        output,
        start: int,
        stop: int,
        read_start: int,
        read_stop: int,
    ) -> dict:
        if not isinstance(output, dict):
            output = {"data": output}

        data = output["data"]
        if "sample_start" not in output:
            i0 = start - read_start
            i1 = i0 + (stop - start)
            data = data[i0:i1]
            output = {
                **output,
                "data": data,
                "sample_start": start,
                "sample_stop": stop,
                "t_start": self.t_min + start / self.fs,
                "t_stop": self.t_min + stop / self.fs,
                "fs": self.fs,
                "dt": 1.0 / self.fs,
            }

        payload = {
            "data": output["data"],
            "t_start": output["t_start"],
            "t_stop": output["t_stop"],
            "sample_start": output["sample_start"],
            "sample_stop": output["sample_stop"],
            "fs": output["fs"],
            "dt": output["dt"],
            "source_sample_start": read_start,
            "source_sample_stop": read_stop,
        }
        if "y" in output:
            payload["y"] = output["y"]
        payload["nbytes"] = output.get("nbytes", payload["data"].nbytes)
        return payload

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
        if not chunks:
            return

        fs = chunks[0]["fs"]
        samples_per_px = fs * (t1 - t0) / width_px

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
            chunk_t0 = chunk["t_start"]
            fs = chunk["fs"]
            dt = chunk["dt"]
            n = chunk["data"].shape[0]

            i0 = math.floor((t0 - chunk_t0) * fs) - 1
            i1 = math.ceil((t1 - chunk_t0) * fs) + 2
            i0 = max(0, i0)
            i1 = min(n, i1)

            if i0 >= i1:
                continue

            sample_start = chunk["sample_start"] + i0
            yield {
                "mode": "raw",
                "data": chunk["data"][i0:i1],
                "t_start": chunk_t0 + i0 * dt,
                "sample_start": sample_start,
                "sample_stop": chunk["sample_start"] + i1,
                "dt": dt,
            }

    def _iter_envelope(
        self,
        chunks,
        t0: float,
        t1: float,
        channel_indices: np.ndarray,
        samples_per_bin: int,
    ):
        if not chunks:
            return

        fs = chunks[0]["fs"]
        dt = chunks[0]["dt"]
        t_origin = chunks[0]["t_start"] - chunks[0]["sample_start"] * dt
        visible_start = math.floor((t0 - t_origin) * fs)
        visible_stop = math.ceil((t1 - t_origin) * fs) + 1

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
            times = t_origin + 0.5 * (starts + stops - 1) * dt

            yield {
                "mode": "envelope",
                "t": times,
                "y_min": np.asfortranarray(y_min),
                "y_max": np.asfortranarray(y_max),
                "samples_per_bin": samples_per_bin,
            }
