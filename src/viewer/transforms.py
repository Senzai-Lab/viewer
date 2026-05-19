from __future__ import annotations

import math

import numpy as np
from scipy import signal


class Compose:
    def __init__(self, *transforms):
        self.transforms = list(transforms)

    @property
    def pad_s(self) -> float:
        return max(float(t.pad_s) for t in self.transforms)

    def setup(self, stream) -> dict:
        meta = {}
        for transform in self.transforms:
            meta.update(transform.setup(stream))
        return meta

    def output_nbytes(self, stream, source_samples: int) -> int:
        nbytes = source_samples * stream.source_n_channels * np.dtype(stream.dtype).itemsize
        for transform in self.transforms:
            nbytes = int(transform.output_nbytes(stream, source_samples))
        return nbytes

    def __call__(self, data: np.ndarray, ctx: dict) -> dict:
        result = {"data": data}
        for transform in self.transforms:
            call_ctx = dict(ctx)
            call_ctx.update(result)
            output = transform(result["data"], call_ctx)
            result.update(output)
        return result

    def draw_settings(self, stream) -> bool:
        changed = False
        for transform in self.transforms:
            changed |= transform.draw_settings(stream)
        return changed


class CAR:
    pad_s = 0.0

    def __init__(self, mode: str = "median"):
        self.mode = mode

    def setup(self, stream) -> dict:
        return {"dtype": np.result_type(stream.source_dtype, np.float32)}

    def output_nbytes(self, stream, source_samples: int) -> int:
        dtype = np.result_type(stream.source_dtype, np.float32)
        return source_samples * stream.source_n_channels * np.dtype(dtype).itemsize

    def __call__(self, data: np.ndarray, ctx: dict) -> dict:
        if self.mode == "mean":
            reference = np.mean(data, axis=1, keepdims=True)
        else:
            reference = np.median(data, axis=1, keepdims=True)
        return {"data": data - reference}

    def draw_settings(self, stream) -> bool:
        from imgui_bundle import imgui

        modes = ["median", "mean"]
        idx = modes.index(self.mode)
        imgui.text("CAR mode")
        imgui.set_next_item_width(-1)
        changed, idx = imgui.combo(f"##car_mode_{stream.name}", idx, modes)
        if changed:
            self.mode = modes[idx]
        return changed


class _SOSFilter:
    def __init__(
        self,
        cutoff,
        btype: str,
        *,
        order: int = 4,
        pad_s: float | None = None,
        zero_phase: bool = True,
    ):
        self.cutoff = cutoff
        self.btype = btype
        self.order = int(order)
        self._pad_s = None if pad_s is None else float(pad_s)
        self.zero_phase = zero_phase
        self.sos = None

    @property
    def pad_s(self) -> float:
        if self._pad_s is not None:
            return self._pad_s
        return 0.0

    def setup(self, stream) -> dict:
        cutoff = self._cutoff()
        self.sos = signal.butter(
            self.order,
            cutoff,
            btype=self.btype,
            fs=stream.source_fs,
            output="sos",
        )
        if self._pad_s is None:
            self._pad_s = self._default_pad_s(cutoff)
        return {"dtype": np.result_type(stream.source_dtype, np.float32)}

    def output_nbytes(self, stream, source_samples: int) -> int:
        dtype = np.result_type(stream.source_dtype, np.float32)
        return source_samples * stream.source_n_channels * np.dtype(dtype).itemsize

    def draw_settings(self, stream) -> bool:
        return False

    def __call__(self, data: np.ndarray, ctx: dict) -> dict:
        if self.zero_phase:
            data = signal.sosfiltfilt(self.sos, data, axis=0, padtype=None)
        else:
            data = signal.sosfilt(self.sos, data, axis=0)
        return {"data": data.astype(np.float32, copy=False)}

    def _cutoff(self):
        return self.cutoff

    def _default_pad_s(self, cutoff) -> float:
        if np.ndim(cutoff) == 0:
            low = float(cutoff)
        else:
            low = float(np.min(cutoff))
        return max(1.0, 3.0 * self.order / low)


class Lowpass(_SOSFilter):
    def __init__(
        self,
        freq: float,
        *,
        order: int = 4,
        pad_s: float | None = None,
        zero_phase: bool = True,
    ):
        super().__init__(
            float(freq),
            "lowpass",
            order=order,
            pad_s=pad_s,
            zero_phase=zero_phase,
        )


class Highpass(_SOSFilter):
    def __init__(
        self,
        freq: float,
        *,
        order: int = 4,
        pad_s: float | None = None,
        zero_phase: bool = True,
    ):
        super().__init__(
            float(freq),
            "highpass",
            order=order,
            pad_s=pad_s,
            zero_phase=zero_phase,
        )


class Bandpass(_SOSFilter):
    def __init__(
        self,
        low: float,
        high: float,
        *,
        order: int = 4,
        pad_s: float | None = None,
        zero_phase: bool = True,
    ):
        super().__init__(
            (float(low), float(high)),
            "bandpass",
            order=order,
            pad_s=pad_s,
            zero_phase=zero_phase,
        )


class FFT:
    def __init__(
        self,
        channel: int = 0,
        window_s: float = 0.5,
        step_s: float = 0.05,
        power: bool = True,
        freq_min: float = 0.0,
        freq_max: float | None = None,
        log_power: bool = False,
    ):
        self.channel = int(channel)
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self.power = power
        self.freq_min = float(freq_min)
        self.freq_max = None if freq_max is None else float(freq_max)
        self.log_power = log_power
        self.freqs: np.ndarray | None = None

    @property
    def pad_s(self) -> float:
        return 0.5 * self.window_s

    def _params(self, source_fs: float) -> tuple[int, int, np.ndarray]:
        window_samples = max(1, int(round(self.window_s * source_fs)))
        step_samples = max(1, int(round(self.step_s * source_fs)))
        freqs = np.fft.rfftfreq(window_samples, d=1.0 / source_fs)
        return window_samples, step_samples, freqs

    def _freq_mask(self, freqs: np.ndarray) -> np.ndarray:
        keep = freqs >= self.freq_min
        if self.freq_max is not None:
            keep &= freqs <= self.freq_max
        return keep

    def setup(self, stream) -> dict:
        _, step_samples, freqs = self._params(stream.source_fs)
        self.freqs = freqs[self._freq_mask(freqs)].astype(np.float32)
        return {
            "fs": stream.source_fs / step_samples,
            "n_channels": len(self.freqs),
            "y": self.freqs,
            "dtype": np.float32,
            "chunk_nbytes": self.output_nbytes(stream, stream.chunk_samples),
        }

    def output_nbytes(self, stream, source_samples: int) -> int:
        _, step_samples, freqs = self._params(stream.source_fs)
        freqs = freqs[self._freq_mask(freqs)]
        n_frames = max(1, math.ceil(source_samples / step_samples))
        return n_frames * len(freqs) * np.dtype(np.float32).itemsize

    def __call__(self, data: np.ndarray, ctx: dict) -> dict:
        source_fs = float(ctx["source_fs"])
        source_t_min = float(ctx["source_t_min"])
        read_start = int(ctx["read_start"])
        request_start = int(ctx["request_start"])
        request_stop = int(ctx["request_stop"])
        window_samples, step_samples, freqs = self._params(source_fs)
        dt = step_samples / source_fs

        first_k = math.ceil(request_start / step_samples)
        stop_k = math.ceil(request_stop / step_samples)
        ks = np.arange(first_k, stop_k, dtype=np.int64)
        centers = ks * step_samples
        window_starts = centers - window_samples // 2
        window_stops = window_starts + window_samples

        keep = (window_starts >= read_start) & (window_stops <= read_start + data.shape[0])
        ks = ks[keep]
        window_starts = window_starts[keep] - read_start

        if len(ks) == 0:
            y = freqs[self._freq_mask(freqs)].astype(np.float32)
            empty = np.empty((0, len(y)), dtype=np.float32)
            t_start = source_t_min + first_k * dt
            return {
                "data": empty,
                "sample_start": first_k,
                "sample_stop": first_k,
                "t_start": t_start,
                "t_stop": t_start,
                "fs": 1.0 / dt,
                "dt": dt,
                "y": y,
                "nbytes": empty.nbytes + y.nbytes,
            }

        signal = np.asarray(data[:, self.channel], dtype=np.float32)
        frames = np.stack(
            [signal[start:start + window_samples] for start in window_starts],
            axis=0,
        )
        frames *= np.hanning(window_samples).astype(np.float32)
        spectrum = np.fft.rfft(frames, axis=1)
        values = np.abs(spectrum)
        if self.power:
            values *= values
        keep = self._freq_mask(freqs)
        values = values[:, keep]
        freqs = freqs[keep]
        if self.log_power:
            values = 10.0 * np.log10(np.maximum(values, np.finfo(np.float32).tiny))
        values = values.astype(np.float32)

        sample_start = int(ks[0])
        sample_stop = int(ks[-1] + 1)
        t_start = source_t_min + sample_start * dt
        t_stop = source_t_min + sample_stop * dt
        y = freqs.astype(np.float32)
        return {
            "data": values,
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "t_start": t_start,
            "t_stop": t_stop,
            "fs": 1.0 / dt,
            "dt": dt,
            "y": y,
            "nbytes": values.nbytes + y.nbytes,
        }

    def draw_settings(self, stream) -> bool:
        from imgui_bundle import imgui

        imgui.text("FFT channel")
        imgui.set_next_item_width(-1)
        max_channel = stream.source_n_channels - 1
        changed, channel = imgui.drag_int(
            f"##fft_channel_{stream.name}",
            self.channel,
            1.0,
            0,
            max_channel,
            "Channel: %d",
        )
        channel = max(0, min(int(channel), max_channel))
        if changed:
            self.channel = channel
        return changed
