from viewer import Ephys, EphysView, show
from viewer.utils import load_env, load_probeinterface
from pathlib import Path
import numpy as np

def load_bin(filename: Path):
    dtype = np.dtype('int16')
    n_channels = 384

    n_values = filename.stat().st_size // dtype.itemsize
    assert n_values % n_channels == 0

    n_samples = n_values // n_channels
    return np.memmap(
        filename=filename,
        dtype=dtype,
        mode='r',
        shape=(n_samples, n_channels),
        order='C'
    )

if __name__ == "__main__":
    data_path = Path(load_env()["EPHYS_DATA_PATH"])
    geometry = load_probeinterface(data_path / 'concat' / 'probe.json')
    data = load_bin(data_path / 'eeg' / 'eeg.dat')

    fs = 1250.0
    ephys = Ephys(
        "probe demo",
        data,
        geometry,
        fs=fs,
        chunk_samples=10 * fs,
        units="uV",
        scale=0.195,
    )

    show(
        [(ephys, EphysView(gain=1 / 40))],
        title="eeg",
        span=1,
    )
