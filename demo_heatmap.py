import numpy as np

from viewer import HeatmapView, TimeSeries, show


if __name__ == "__main__":
    duration = 30.0
    dt = 0.025
    times = np.arange(0.0, duration, dt)
    fs = 1.0 / dt

    freqs = np.linspace(1.0, 120.0, 96)
    ridge = 35.0 + 22.0 * np.sin(2.0 * np.pi * times / 9.0)
    wide = 0.6 * np.exp(-0.5 * ((freqs[None, :] - ridge[:, None]) / 9.0) ** 2)
    slow = 0.25 * np.sin(
        2.0 * np.pi * freqs[None, :] / 38.0 + times[:, None] * 0.8
    )
    bursts = np.zeros_like(wide)
    for center in (6.0, 14.0, 23.0):
        bursts += 0.45 * np.exp(-0.5 * ((times[:, None] - center) / 0.45) ** 2)
    spectrogram = 10.0 * np.log10(1.0 + wide + slow + bursts)

    values = np.linspace(-4.0, 4.0, 80)
    mean = 1.4 * np.sin(2.0 * np.pi * times / 12.0)
    sigma = 0.45 + 0.25 * (1.0 + np.sin(2.0 * np.pi * times / 7.0))
    density = np.exp(-0.5 * ((values[None, :] - mean[:, None]) / sigma[:, None]) ** 2)
    density /= density.sum(axis=1, keepdims=True)

    spec_stream = TimeSeries(
        "spectrogram",
        spectrogram.astype(np.float32),
        times,
        fs,
        chunk_samples=240,
    )
    density_stream = TimeSeries(
        "probability density",
        density.astype(np.float32),
        times,
        fs,
        chunk_samples=240,
    )

    show(
        [
            (
                spec_stream,
                HeatmapView(freqs, y_label="Frequency (Hz)", cmap="Viridis"),
            ),
            (
                density_stream,
                HeatmapView(values, y_label="Value", cmap="Plasma"),
            ),
        ],
        title="Heatmap Demo",
        span=8.0,
    )
