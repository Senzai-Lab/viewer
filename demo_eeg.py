from viewer import Ephys, EphysSettings, run_viewer
from viewer.utils import load_prb

from pathlib import Path
import numpy as np

if __name__ == "__main__":
    data_path = Path('/Volumes/fsmresfiles/Basic_Sciences/Phys/SenzaiLab/pipeline_output/aa005/ProbeA')
    geometry = load_prb(ks_path=data_path / 'kilosort')
    data = np.memmap(data_path / 'eeg.dat',
                    dtype='int16',
                    mode='r',
                    shape=(20745665, 384),
                    order='C',
                    )

    fs = 1250
    ephys = Ephys(
        "probe demo",
        data,
        geometry,
        fs=fs,
        chunk_samples=10 * fs,
        units="values",
    )

    run_viewer([(ephys, EphysSettings(ephys))], title="eeg")