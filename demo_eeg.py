from viewer import Ephys, EphysSettings, run_viewer
from viewer.utils import load_probeinterface
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
    data_path = Path('/Volumes/fsmresfiles/Basic_Sciences/Phys/SenzaiLab/pipeline_output/aa005/ProbeA')
    geometry = load_probeinterface(data_path / 'concat' / 'probe.json')
    data = load_bin(data_path / 'eeg.dat')

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

    run_viewer(
        [(ephys, EphysSettings(geometry, gain=1 / 40))],
        title="eeg",
        span=1,
    )
