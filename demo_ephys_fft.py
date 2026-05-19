from pathlib import Path

import numpy as np
import pandas as pd

from viewer import Ephys, EphysView, HeatmapView, RasterView, Spikes, show
from viewer.utils import load_env, load_probeinterface
from viewer.transforms import Bandpass, FFT


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
    data_path = Path(load_env()["EPHYS_DATA_PATH"])
    geometry = load_probeinterface(data_path / "concat" / "probe.json")
    data = load_bin(data_path / "eeg" / "eeg.dat")
    units_metadata = pd.read_csv("./scripts/units.csv").to_dict()

    fs = 1250.0
    
    raw_stream = Ephys(
        "Probe A",
        data,
        geometry,
        fs=fs,
        chunk_samples=int(10 * fs),
        units="uV",
        scale=0.195,
    )
    fft_stream = raw_stream.pipe(
        Bandpass(1.0, 125.0),
        FFT(
            channel=100,
            window_s=0.2,
            step_s=0.05,
            freq_min=1.0,
            freq_max=100.0,
            log_power=True,
        ),
        name="fft1",
    )

    spikes = Spikes(
        "units",
        ts=np.load("spike_times.npy"),
        spike_units=np.load("spike_units.npy", mmap_mode="r"),
        unit_ids=units_metadata['unit_ids'],)

    show(
        [
            (raw_stream, EphysView(geometry, gain=1 / 40)),
            (fft_stream, HeatmapView(y_label="Frequency (Hz)", cmap="Viridis", auto_scale=True)),
            (spikes, RasterView(metadata=units_metadata, sort_by='unit_display_y')),
        ],
        title="Ephys FFT Heatmap",
        span=2.0,
    )
