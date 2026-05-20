from pathlib import Path

import numpy as np
import pandas as pd

from viewer import (
    Ephys,
    EphysView,
    HeatmapView,
    ProbeView,
    RasterView,
    Spikes,
    show,
)
from viewer.utils import load_env, load_probeinterface, load_bin
from viewer.transforms import Bandpass, FFT


if __name__ == "__main__":
    data_path = Path(load_env()["EPHYS_DATA_PATH"])
    geometry = load_probeinterface(data_path / "concat" / "probe.json")
    data = load_bin(data_path / "eeg" / "eeg.dat")
    units = pd.read_csv("./scripts/units.csv")

    units_metadata = {col: units[col].to_numpy() for col in units.columns}

    fs = 1250.0
    fft_channel = 100

    raw_stream = Ephys("Probe A", data, geometry, fs=fs, chunk_samples=int(10 * fs), units="uV", scale=0.195,)
    probe_view = ProbeView(geometry, visible_channels=np.arange(fft_channel - 4, fft_channel + 5),)

    fft_stream = raw_stream.pipe(
        Bandpass(1.0, 125.0),
        FFT(channel=fft_channel, window_s=0.2, step_s=0.01, freq_min=1.0, freq_max=100.0, log_power=True),
        name="fft1",
        )
    
    spikes = Spikes(
        "units",
        ts=np.load("spike_times.npy") / 30_000.0,
        spike_units=np.load("spike_units.npy", mmap_mode="r"),
        unit_ids=units["unit_ids"].to_numpy(),
    )

    show(
        [
            (raw_stream, EphysView(probe=probe_view, gain=1 / 40)),
            (fft_stream, HeatmapView(y_label="Frequency (Hz)", cmap="Viridis", auto_scale=True)),
            (spikes, RasterView(metadata=units_metadata, sort_by="unit_display_y")),
        ],
        title="sSC view",
        span=2.0,
    )
