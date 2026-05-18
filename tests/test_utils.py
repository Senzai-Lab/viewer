from __future__ import annotations

import numpy as np

from viewer.utils import save_filtered_units


def test_save_filtered_units_filters_parallel_spike_arrays(tmp_path):
    spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    spike_units = np.array([4, 2, 7, 4, 9])
    unit_ids = np.array([4, 7])

    summary = save_filtered_units(
        spike_times,
        spike_units,
        unit_ids,
        tmp_path,
        spike_units_filename="spike_clusters.npy",
    )

    np.testing.assert_array_equal(
        np.load(tmp_path / "spike_times.npy"),
        np.array([0.1, 0.3, 0.4]),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "spike_clusters.npy"),
        np.array([4, 7, 4]),
    )
    np.testing.assert_array_equal(np.load(tmp_path / "unit_ids.npy"), unit_ids)
    assert summary["n_spikes"] == 3
    assert summary["n_units"] == 2
