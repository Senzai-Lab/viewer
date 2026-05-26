from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np

from .base import BaseStream


class Video(BaseStream):
    """File-backed video stream decoded into small time chunks."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        *,
        chunk_duration: float = 0.5,
        stream_index: int = 0,
        scale: float = 1.0,
    ):
        self.name = name
        self.path = Path(path)
        self.chunk_duration = float(chunk_duration)
        self.stream_index = int(stream_index)
        self.scale = float(scale)

        with av.open(str(self.path)) as container:
            stream = container.streams.video[self.stream_index]
            self.width = round(int(stream.width) * self.scale)
            self.height = round(int(stream.height) * self.scale)
            self.fs = float(stream.average_rate)
            self.time_base = float(stream.time_base)
            self.n_frames = int(stream.frames)
            self.first_frame_s = float(next(container.decode(stream)).time)
            self.duration = (self.n_frames - 1) / self.fs

        self.t_min = 0.0
        self.t_max = self.duration
        self.n_chunks = max(1, math.ceil(self.duration / self.chunk_duration))

        frames_per_chunk = math.ceil(self.fs * self.chunk_duration) + 1
        frame_nbytes = self.width * self.height * 3
        self.chunk_nbytes = frames_per_chunk * (frame_nbytes + 8)

    def at(self, t: float) -> int:
        chunk_idx = math.floor((t - self.t_min) / self.chunk_duration)
        return max(0, min(chunk_idx, self.n_chunks - 1))

    def read(self, chunk_idx: int) -> dict:
        t0 = self.t_min + chunk_idx * self.chunk_duration
        t1 = min(self.t_max, t0 + self.chunk_duration)
        i0 = max(0, int(math.ceil(t0 * self.fs - 1e-9)))
        i1 = (
            self.n_frames
            if chunk_idx == self.n_chunks - 1
            else min(self.n_frames, int(math.ceil(t1 * self.fs - 1e-9)))
        )

        frames = []
        ts = []
        frame_idx = []
        past_stop = 0

        with av.open(str(self.path)) as container:
            stream = container.streams.video[self.stream_index]
            seek_s = max(0.0, self.first_frame_s + t0 - 1.0 / self.fs)
            seek_ts = int(seek_s / self.time_base)
            container.seek(seek_ts, stream=stream, backward=True)

            for frame in container.decode(stream):
                idx = int(round((float(frame.time) - self.first_frame_s) * self.fs))
                if idx < i0:
                    continue
                if idx >= i1:
                    past_stop += 1
                    if past_stop >= 12:
                        break
                    continue

                frames.append(
                    frame.reformat(
                        width=self.width,
                        height=self.height,
                        format="rgb24",
                    ).to_ndarray()
                )
                ts.append(idx / self.fs)
                frame_idx.append(idx)

        if frames:
            order = np.argsort(frame_idx)
            data = np.stack(frames, axis=0)[order]
            ts = np.asarray(ts, dtype=np.float64)[order]
            frame_idx = np.asarray(frame_idx, dtype=np.int64)[order]
        else:
            data = np.empty((0, self.height, self.width, 3), dtype=np.uint8)
            ts = np.empty(0, dtype=np.float64)
            frame_idx = np.empty(0, dtype=np.int64)

        return {
            "t_start": t0,
            "t_stop": t1,
            "ts": ts,
            "frame_idx": frame_idx,
            "fs": self.fs,
            "dt": 1.0 / self.fs,
            "data": data,
            "nbytes": data.nbytes + ts.nbytes,
        }
