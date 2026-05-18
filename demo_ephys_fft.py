from pathlib import Path

import numpy as np

from viewer import Ephys, EphysSettings, HeatmapSettings, run_viewer
from viewer.utils import env_path, load_probeinterface
from viewer.transforms import Bandpass, Compose, FFT


def load_bin(filename: Path):
    dtype = np.dtype("int16")
    n_channels = 384

    n_values = filename.stat().st_size // dtype.itemsize
    assert n_values % n_channels == 0

    n_samples = n_values // n_channels
    return np.memmap(
        filename=filename,
        dtype=dtype,
        mode="r",
        shape=(n_samples, n_channels),
        order="C",
    )


if __name__ == "__main__":
    data_path = env_path("VIEWER_EPHYS_DATA_PATH")
    geometry = load_probeinterface(data_path / "concat" / "probe.json")
    data = load_bin(data_path / "eeg.dat")

    fs = 1250.0
    raw_stream = Ephys(
        "raw eeg",
        data,
        geometry,
        fs=fs,
        chunk_samples=int(10 * fs),
        units="uV",
        scale=0.195,
    )
    fft_stream = Ephys(
        "fft1",
        data,
        geometry,
        fs=fs,
        chunk_samples=int(10 * fs),
        units="uV",
        scale=0.195,
        transform=Compose(
            Bandpass(1.0, 125.0),
            FFT(
                channel=12,
                window_s=0.5,
                step_s=0.025,
                freq_min=1.0,
                freq_max=100.0,
                log_power=True,
            ),
        ),
    )
    fft_stream2 = Ephys(
        "fft2",
        data,
        geometry,
        fs=fs,
        chunk_samples=int(10 * fs),
        units="uV",
        scale=0.195,
        transform=Compose(
            Bandpass(1.0, 125.0),
            FFT(
                channel=2,
                window_s=0.5,
                step_s=0.025,
                freq_min=1.0,
                freq_max=100.0,
                log_power=True,
            ),
        ),
    )

    run_viewer(
        [
            (raw_stream, EphysSettings(geometry, gain=1 / 40)),
            (
                fft_stream,
                HeatmapSettings(y_label="Frequency (Hz)", cmap="Viridis", auto_scale=True),
            ),
            (
                fft_stream2,
                HeatmapSettings(y_label="Frequency (Hz)", cmap="Viridis", auto_scale=True),
            )

        ],
        title="Ephys FFT Heatmap",
        span=2.0,
    )
