from __future__ import annotations

from concurrent.futures import Future

import numpy as np

from viewer import Ephys, HeatmapSettings, TimeSeries
from viewer.cache import ChunkCache
from viewer.render.heatmap import iter_heatmap_visible
from viewer.transforms import Bandpass, CAR, Compose, FFT, Highpass, Lowpass


def test_public_imports():
    assert HeatmapSettings is not None
    assert Compose is not None
    assert CAR is not None
    assert FFT is not None
    assert Lowpass is not None
    assert Highpass is not None
    assert Bandpass is not None


def test_timeseries_chunk_without_transform_has_local_timing():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    ts = np.arange(10, dtype=np.float32) / 10.0
    stream = TimeSeries("raw", values, ts, 10.0, chunk_samples=4)

    chunk = stream.load_chunk(1)

    np.testing.assert_array_equal(chunk["data"], values[4:8])
    assert chunk["sample_start"] == 4
    assert chunk["sample_stop"] == 8
    assert chunk["t_start"] == 0.4
    assert chunk["t_stop"] == 0.8
    assert chunk["fs"] == 10.0
    assert chunk["dt"] == 0.1
    assert chunk["n_bytes"] == values[4:8].nbytes


def test_car_transform_removes_per_sample_average():
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
    stream = TimeSeries(
        "car",
        values,
        ts,
        1.0,
        chunk_samples=len(values),
        transform=CAR(mode="mean"),
    )

    chunk = stream.load_chunk(0)
    expected = values - values.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(chunk["data"], expected)
    np.testing.assert_allclose(chunk["data"].mean(axis=1), 0.0, atol=1e-6)
    assert chunk["sample_start"] == 0
    assert chunk["sample_stop"] == len(values)
    assert chunk["source_sample_start"] == 0
    assert chunk["source_sample_stop"] == len(values)
    assert chunk["fs"] == 1.0


def test_ephys_accepts_transform_and_returns_standard_chunk():
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
    stream = Ephys(
        "ephys",
        values,
        geometry,
        fs=2.0,
        chunk_samples=2,
        scale=2.0,
        transform=CAR(mode="mean"),
    )

    chunk = stream.load_chunk(1)
    scaled = values[2:4].astype(np.float32) * 2.0
    expected = scaled - scaled.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(chunk["data"], expected)
    assert chunk["sample_start"] == 2
    assert chunk["sample_stop"] == 4
    assert chunk["t_start"] == 1.0
    assert chunk["t_stop"] == 2.0
    assert chunk["fs"] == 2.0
    assert chunk["dt"] == 0.5


def test_fft_transform_outputs_spectrogram_metadata_and_channel_selection():
    fs = 100.0
    t = np.arange(200, dtype=np.float32) / fs
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 20.0 * t),
        ]
    ).astype(np.float32)

    transform = FFT(channel=1, window_s=0.2, step_s=0.1)
    stream = TimeSeries("fft", values, t, fs, chunk_samples=100, transform=transform)
    chunk = stream.load_chunk(0)

    assert chunk["data"].shape == (9, 11)
    assert chunk["sample_start"] == 1
    assert chunk["sample_stop"] == 10
    assert chunk["fs"] == 10.0
    assert chunk["dt"] == 0.1
    assert chunk["source_sample_start"] == 0
    assert chunk["source_sample_stop"] == 110
    np.testing.assert_allclose(chunk["y"], np.fft.rfftfreq(20, d=1.0 / fs))
    assert chunk["n_bytes"] == chunk["data"].nbytes + chunk["y"].nbytes

    peak_freq = chunk["y"][np.argmax(chunk["data"].mean(axis=0))]
    assert peak_freq == 20.0

    transform.channel = 0
    stream.setup_transform()
    chunk = stream.load_chunk(0)
    peak_freq = chunk["y"][np.argmax(chunk["data"].mean(axis=0))]
    assert peak_freq == 5.0


def test_filter_transforms_are_sample_preserving():
    fs = 500.0
    t = np.arange(1000, dtype=np.float32) / fs
    values = (
        np.sin(2.0 * np.pi * 10.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)
    ).astype(np.float32)

    low = TimeSeries(
        "low",
        values,
        t,
        fs,
        chunk_samples=len(values),
        transform=Lowpass(30.0),
    ).load_chunk(0)["data"][:, 0]
    high = TimeSeries(
        "high",
        values,
        t,
        fs,
        chunk_samples=len(values),
        transform=Highpass(60.0),
    ).load_chunk(0)["data"][:, 0]
    band = TimeSeries(
        "band",
        values,
        t,
        fs,
        chunk_samples=len(values),
        transform=Bandpass(8.0, 20.0),
    ).load_chunk(0)["data"][:, 0]

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


def test_compose_filter_before_fft():
    fs = 500.0
    t = np.arange(1000, dtype=np.float32) / fs
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 10.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * 120.0 * t),
        ]
    ).astype(np.float32)
    stream = TimeSeries(
        "compose",
        values,
        t,
        fs,
        chunk_samples=len(values),
        transform=Compose(
            Lowpass(30.0),
            FFT(channel=0, window_s=0.25, step_s=0.05, freq_max=150.0),
        ),
    )

    chunk = stream.load_chunk(0)
    mean_power = chunk["data"].mean(axis=0)
    peak_freq = chunk["y"][np.argmax(mean_power)]

    assert chunk["data"].shape[1] == len(chunk["y"])
    assert abs(float(peak_freq) - 10.0) <= 2.0


def test_cache_reset_stream_only_clears_named_stream():
    cache = ChunkCache(max_workers=1)
    future = Future()
    cache.cache[("a", 0)] = {"n_bytes": 1}
    cache.cache[("b", 0)] = {"n_bytes": 1}
    cache.pending[("a", 1)] = future
    cache.desired["a"] = {0, 1}
    cache.desired["b"] = {0}

    cache.reset_stream("a")

    assert ("a", 0) not in cache.cache
    assert ("b", 0) in cache.cache
    assert ("a", 1) not in cache.pending
    assert future.cancelled()
    assert cache.desired["a"] == set()
    assert cache.desired["b"] == {0}
    cache.close()


def test_heatmap_visible_uses_chunk_local_fs():
    chunk = {
        "data": np.arange(30, dtype=np.float32).reshape(10, 3),
        "t_start": 10.0,
        "sample_start": 20,
        "sample_stop": 30,
        "fs": 2.0,
        "dt": 0.5,
        "n_bytes": 120,
    }

    items = list(iter_heatmap_visible(None, [chunk], 11.0, 12.0))

    assert len(items) == 1
    item = items[0]
    np.testing.assert_array_equal(item["data"], chunk["data"][1:6])
    assert item["sample_start"] == 21
    assert item["sample_stop"] == 26
    assert item["x0"] == 10.25
    assert item["x1"] == 12.75
