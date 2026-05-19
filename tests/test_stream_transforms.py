from __future__ import annotations

from concurrent.futures import Future

import numpy as np

from viewer import Cache, Ephys, HeatmapView, RasterView, Span, Spikes, TimeSeries, TraceView, show
from viewer.render.heatmap import iter_heatmap_visible
from viewer.transforms import Bandpass, CAR, Compose, FFT, Highpass, Lowpass


class _TinyStream:
    def __init__(self, name: str, n_chunks: int = 5):
        self.name = name
        self.chunk_nbytes = 1
        self.n_chunks = n_chunks

    def chunk_at(self, t: float) -> int:
        return max(0, min(int(t), self.n_chunks - 1))

    def read(self, i: int) -> dict:
        return {"i": i, "nbytes": 1}


def test_public_imports():
    assert HeatmapView is not None
    assert Cache is not None
    assert RasterView is not None
    assert Span is not None
    assert Spikes is not None
    assert TraceView is not None
    assert show is not None
    assert Compose is not None
    assert CAR is not None
    assert FFT is not None
    assert Lowpass is not None
    assert Highpass is not None
    assert Bandpass is not None


def test_timeseries_chunk_without_pipe_has_local_timing():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    ts = np.arange(10, dtype=np.float32) / 10.0
    stream = TimeSeries("raw", values, ts, 10.0, chunk_samples=4)

    chunk = stream[1]

    np.testing.assert_array_equal(chunk["data"], values[4:8])
    assert chunk["sample_start"] == 4
    assert chunk["sample_stop"] == 8
    assert chunk["t_start"] == 0.4
    assert chunk["t_stop"] == 0.8
    assert chunk["fs"] == 10.0
    assert chunk["dt"] == 0.1
    assert chunk["nbytes"] == values[4:8].nbytes
    assert len(stream) == 3
    assert [item["sample_start"] for item in stream[0:2]] == [0, 4]
    assert list(stream.chunks_in(Span(0.1, 0.6))) == [0, 1]
    assert stream.at(0.45)["sample_start"] == 4


def test_car_pipe_removes_per_sample_average():
    values = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 9.0, 12.0],
            [10.0, 12.0, 14.0],
        ],
        dtype=np.float32,
    )
    ts = np.arange(len(values), dtype=np.float32)
    raw = TimeSeries("raw", values, ts, 1.0, chunk_samples=len(values))
    stream = raw.pipe(CAR(mode="mean"), name="car")

    chunk = stream[0]
    expected = values - values.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(chunk["data"], expected)
    np.testing.assert_allclose(chunk["data"].mean(axis=1), 0.0, atol=1e-6)
    assert chunk["sample_start"] == 0
    assert chunk["sample_stop"] == len(values)
    assert chunk["source_sample_start"] == 0
    assert chunk["source_sample_stop"] == len(values)
    assert chunk["fs"] == 1.0


def test_ephys_pipe_returns_standard_chunk():
    values = np.array(
        [
            [1, 3],
            [2, 6],
            [4, 8],
            [5, 11],
        ],
        dtype=np.int16,
    )
    geometry = {
        "channel_ids": np.arange(2),
        "x": np.zeros(2),
        "y": np.arange(2),
        "shank_ids": np.zeros(2),
    }
    raw = Ephys(
        "ephys",
        values,
        geometry,
        fs=2.0,
        chunk_samples=2,
        scale=2.0,
    )
    stream = raw.pipe(CAR(mode="mean"), name="ephys/car")

    chunk = stream[1]
    scaled = values[2:4].astype(np.float32) * 2.0
    expected = scaled - scaled.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(chunk["data"], expected)
    assert chunk["sample_start"] == 2
    assert chunk["sample_stop"] == 4
    assert chunk["t_start"] == 1.0
    assert chunk["t_stop"] == 2.0
    assert chunk["fs"] == 2.0
    assert chunk["dt"] == 0.5


def test_fft_pipe_outputs_spectrogram_metadata_and_channel_selection():
    fs = 100.0
    t = np.arange(200, dtype=np.float32) / fs
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 20.0 * t),
        ]
    ).astype(np.float32)

    fft = FFT(channel=1, window_s=0.2, step_s=0.1)
    raw = TimeSeries("raw", values, t, fs, chunk_samples=100)
    stream = raw.pipe(fft, name="fft")
    chunk = stream[0]

    assert chunk["data"].shape == (9, 11)
    assert chunk["sample_start"] == 1
    assert chunk["sample_stop"] == 10
    assert chunk["fs"] == 10.0
    assert chunk["dt"] == 0.1
    assert chunk["source_sample_start"] == 0
    assert chunk["source_sample_stop"] == 110
    np.testing.assert_allclose(chunk["y"], np.fft.rfftfreq(20, d=1.0 / fs))
    assert chunk["nbytes"] == chunk["data"].nbytes + chunk["y"].nbytes

    peak_freq = chunk["y"][np.argmax(chunk["data"].mean(axis=0))]
    assert peak_freq == 20.0

    fft.channel = 0
    stream.setup_pipe()
    chunk = stream[0]
    peak_freq = chunk["y"][np.argmax(chunk["data"].mean(axis=0))]
    assert peak_freq == 5.0


def test_filter_pipes_are_sample_preserving():
    fs = 500.0
    t = np.arange(1000, dtype=np.float32) / fs
    values = (
        np.sin(2.0 * np.pi * 10.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)
    ).astype(np.float32)

    raw = TimeSeries("raw", values, t, fs, chunk_samples=len(values))
    low = raw.pipe(Lowpass(30.0), name="low")[0]["data"][:, 0]
    high = raw.pipe(Highpass(60.0), name="high")[0]["data"][:, 0]
    band = raw.pipe(Bandpass(8.0, 20.0), name="band")[0]["data"][:, 0]

    freqs = np.fft.rfftfreq(len(values), d=1.0 / fs)
    low_power = np.abs(np.fft.rfft(low))
    high_power = np.abs(np.fft.rfft(high))
    band_power = np.abs(np.fft.rfft(band))
    f10 = np.argmin(np.abs(freqs - 10.0))
    f120 = np.argmin(np.abs(freqs - 120.0))

    assert low.shape == values.shape
    assert high.shape == values.shape
    assert band.shape == values.shape
    assert low_power[f10] > 5 * low_power[f120]
    assert high_power[f120] > 5 * high_power[f10]
    assert band_power[f10] > 5 * band_power[f120]


def test_pipe_chains_filter_before_fft():
    fs = 500.0
    t = np.arange(1000, dtype=np.float32) / fs
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 10.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * 120.0 * t),
        ]
    ).astype(np.float32)
    raw = TimeSeries("raw", values, t, fs, chunk_samples=len(values))
    stream = raw.pipe(
        Lowpass(30.0),
        FFT(channel=0, window_s=0.25, step_s=0.05, freq_max=150.0),
        name="spec",
    )

    chunk = stream[0]
    mean_power = chunk["data"].mean(axis=0)
    peak_freq = chunk["y"][np.argmax(mean_power)]

    assert chunk["data"].shape[1] == len(chunk["y"])
    assert abs(float(peak_freq) - 10.0) <= 2.0


def test_pipe_from_derived_stream_keeps_raw_source():
    fs = 500.0
    t = np.arange(1000, dtype=np.float32) / fs
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 10.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * 120.0 * t),
        ]
    ).astype(np.float32)
    raw = TimeSeries("raw", values, t, fs, chunk_samples=len(values))
    lfp = raw.pipe(Lowpass(30.0), name="lfp")
    spec = lfp.pipe(FFT(channel=0, window_s=0.25, step_s=0.05), name="spec")

    assert lfp.source is raw
    assert spec.source is raw
    assert raw.transforms is None
    assert lfp.name == "lfp"
    assert spec.name == "spec"
    assert spec[0]["data"].shape[1] == len(spec.y)


def test_cache_drop_stream_only_clears_named_stream():
    cache = Cache(workers=1)
    stream_a = _TinyStream("a")
    stream_b = _TinyStream("b")
    cache.add(stream_a)
    cache.add(stream_b)
    future = Future()
    cache.cache[("a", 0)] = {"nbytes": 1}
    cache.cache[("b", 0)] = {"nbytes": 1}
    cache.pending[("a", 1)] = future
    cache.wanted["a"] = {0, 1}
    cache.wanted["b"] = {0}

    cache.drop(stream_a)

    assert ("a", 0) not in cache.cache
    assert ("b", 0) in cache.cache
    assert ("a", 1) not in cache.pending
    assert future.cancelled()
    assert cache.wanted["a"] == set()
    assert cache.wanted["b"] == {0}
    cache.close()


def test_cache_request_and_chunks_are_cursor_centered():
    stream = _TinyStream("a")
    cache = Cache([stream], workers=1)

    cache.request(stream, 2.4)
    for future in list(cache.pending.values()):
        future.result()
    cache.poll()

    assert cache.wanted["a"] == {1, 2, 3}
    assert [chunk["i"] for chunk in cache.chunks(stream, 2.4)] == [1, 2, 3]
    assert (stream, 2) in cache
    assert ("a", 2) in cache
    assert cache[stream, 2]["i"] == 2
    assert cache["a", 2]["i"] == 2
    assert cache.nbytes == 3
    cache.close()


def test_heatmap_visible_uses_chunk_local_fs():
    chunk = {
        "data": np.arange(30, dtype=np.float32).reshape(10, 3),
        "t_start": 10.0,
        "sample_start": 20,
        "sample_stop": 30,
        "fs": 2.0,
        "dt": 0.5,
        "nbytes": 120,
    }

    items = list(iter_heatmap_visible(None, [chunk], 11.0, 12.0))

    assert len(items) == 1
    item = items[0]
    np.testing.assert_array_equal(item["data"], chunk["data"][1:6])
    assert item["sample_start"] == 21
    assert item["sample_stop"] == 26
    assert item["x0"] == 10.25
    assert item["x1"] == 12.75
