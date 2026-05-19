import numpy as np
import pytest

from viewer import EphysView, ProbeView


def geometry():
    return {
        "channel_ids": np.array([10, 20, 30, 40]),
        "shank_ids": np.array([0, 0, 1, 1]),
        "x": np.array([0.0, 20.0, 0.0, 20.0]),
        "y": np.array([0.0, 0.0, 20.0, 20.0]),
    }


def test_probe_view_initial_visible_channels():
    probe = ProbeView(geometry(), visible_channels=[20, 40])

    np.testing.assert_array_equal(probe.visible, [False, True, False, True])


def test_ephys_view_accepts_probe_view():
    probe = ProbeView(geometry(), visible_channels=[10])
    view = EphysView(probe=probe)

    assert view.probe is probe


def test_ephys_view_requires_probe_view():
    with pytest.raises(TypeError):
        EphysView()
