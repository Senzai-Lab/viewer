from __future__ import annotations

import os

import numpy as np

from .dense_signal import DenseSignalAdapter


class BinarySignalAdapter(DenseSignalAdapter):
    """
    File-backed dense signal adapter for headerless interleaved binary data.

    The file layout is sample-major: ``sample0/ch0, sample0/ch1, ..., sample1/ch0, ...``.
    Metadata such as dtype, channel count, and sample rate is provided
    explicitly, while chunk planning and materialization reuse
    ``DenseSignalAdapter``.
    """

    def __init__(
        self,
        stream_id: str,
        file_path: str | os.PathLike[str],
        *,
        dtype: str | np.dtype,
        channel_count: int,
        sample_rate_hz: float,
        start_time: float = 0.0,
        target_chunk_bytes: int = 8 * 1024 * 1024,
        byte_offset: int = 0,
    ):
        if channel_count <= 0:
            raise ValueError("channel_count must be > 0")
        if byte_offset < 0:
            raise ValueError("byte_offset must be >= 0")

        self.file_path = os.fspath(file_path)
        self.sample_dtype = np.dtype(dtype)
        self.byte_offset = int(byte_offset)

        file_size = os.path.getsize(self.file_path)
        if self.byte_offset > file_size:
            raise ValueError("byte_offset exceeds file size")

        frame_bytes = int(channel_count) * self.sample_dtype.itemsize
        data_bytes = file_size - self.byte_offset
        if frame_bytes <= 0:
            raise ValueError("frame size must be > 0")
        if data_bytes % frame_bytes != 0:
            raise ValueError(
                "file size is not divisible by channel_count * dtype.itemsize"
            )

        self.file_sample_count = data_bytes // frame_bytes
        raw = np.memmap(
            self.file_path,
            dtype=self.sample_dtype,
            mode="r",
            offset=self.byte_offset,
            shape=(self.file_sample_count, int(channel_count)),
        )

        # The on-disk layout is sample-major, while DenseSignalAdapter expects
        # channel-first data.
        channel_first = raw.T
        super().__init__(
            stream_id,
            channel_first,
            sample_rate_hz,
            start_time=start_time,
            target_chunk_bytes=target_chunk_bytes,
        )