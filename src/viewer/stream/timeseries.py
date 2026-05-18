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
            transform=None,
    ):
        self.name = name
        self.values = values
        self.ts = ts
        self.transform = transform

        self.source_fs = float(fs)
        self.fs = self.source_fs
        self.t_min = float(ts[0])
        self.t_max = float(ts[-1])
        self.duration = self.t_max - self.t_min

        self.n_samples = values.shape[0]
        self.source_n_channels = values.shape[1] if values.ndim > 1 else 1
        self.n_channels = self.source_n_channels
        self.source_dtype = values.dtype
        self.dtype = self.source_dtype
        self.y = None
        self.chunk_samples = int(chunk_samples)
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self.n_chunks = -(-self.n_samples // self.chunk_samples)

        self.chunk_nbytes = 0
        self.setup_transform()

    def setup_transform(self):
        self.fs = self.source_fs
        self.n_channels = self.source_n_channels
        self.dtype = self.source_dtype
        self.y = None
        self.chunk_nbytes = (
            self.chunk_samples * self.n_channels * np.dtype(self.dtype).itemsize
        )

        if self.transform is None:
            return

        setup = getattr(self.transform, "setup", None)
        meta = setup(self) if setup is not None else {}
        meta = {} if meta is None else meta

        self.fs = float(meta.get("fs", self.fs))
        self.n_channels = int(meta.get("n_channels", self.n_channels))
        if "dtype" in meta:
            self.dtype = np.dtype(meta["dtype"])
        else:
            self.dtype = np.dtype(np.result_type(self.source_dtype, np.float32))
        self.y = meta.get("y", self.y)

        output_nbytes = getattr(self.transform, "output_nbytes", None)
        if "chunk_nbytes" in meta:
            self.chunk_nbytes = int(meta["chunk_nbytes"])
        elif output_nbytes is not None:
            self.chunk_nbytes = int(output_nbytes(self, self.chunk_samples))
        else:
            self.chunk_nbytes = (
                self.chunk_samples * self.n_channels * np.dtype(self.dtype).itemsize
            )

    def iter_visible(self, chunks, t0: float, t1: float, width_px: float):
        for chunk in chunks:
            chunk_t0 = chunk["t_start"]
            fs = chunk["fs"]
            dt = chunk["dt"]
            stride = max(1, math.floor(fs * (t1 - t0) / max(width_px, 1)))
            data = chunk["data"]
            n = data.shape[0]

            i0 = max(0, math.floor((t0 - chunk_t0) * fs) - 1)
            i1 = min(n, math.ceil((t1 - chunk_t0) * fs) + 1)

            if i0 >= i1:
                continue

            if stride > 1:
                offset = (stride - ((chunk["sample_start"] + i0) % stride)) % stride
                i0 += offset
                if i0 >= i1:
                    continue

            yield (
                data[i0:i1:stride],
                chunk_t0 + i0 * dt,
                stride * dt,
            )

    def chunk_at(self, t: float) -> int:
        sample_idx = math.floor((t - self.t_min) * self.source_fs)
        chunk_idx = sample_idx // self.chunk_samples
        return max(0, min(chunk_idx, self.n_chunks - 1))

    def load_chunk(self, chunk_idx: int) -> dict:
        start = chunk_idx * self.chunk_samples
        stop = min(start + self.chunk_samples, self.n_samples)
        pad = 0
        if self.transform is not None:
            pad = math.ceil(float(getattr(self.transform, "pad_s", 0.0)) * self.source_fs)
        read_start = max(0, start - pad)
        read_stop = min(self.n_samples, stop + pad)

        if self.transform is None:
            data = np.asarray(self.values[start:stop])
        else:
            data = np.array(self.values[read_start:read_stop], dtype=self.dtype, copy=True)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if self.transform is not None:
            ctx = {
                "stream": self,
                "source_fs": self.source_fs,
                "source_t_min": self.t_min,
                "request_start": start,
                "request_stop": stop,
                "read_start": read_start,
                "read_stop": read_stop,
            }
            output = self.transform(data, ctx)
            return self._chunk_from_transform(output, start, stop, read_start, read_stop)

        return {
            "t_start": self.t_min + start / self.source_fs,
            "t_stop": self.t_min + stop / self.source_fs,
            "sample_start": start,
            "sample_stop": stop,
            "fs": self.source_fs,
            "dt": 1.0 / self.source_fs,
            "n_bytes": data.nbytes,
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
        payload["n_bytes"] = output.get("n_bytes", payload["data"].nbytes)
        return payload
