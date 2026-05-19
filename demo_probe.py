import numpy as np

from viewer import Ephys, EphysView, show


def make_geometry(n_shanks=4, contacts_per_shank=96):
    shank_ids = np.repeat(np.arange(n_shanks), contacts_per_shank)
    rows = np.tile(np.arange(contacts_per_shank), n_shanks)
    shanks = np.repeat(np.arange(n_shanks), contacts_per_shank)

    x = shanks * 120.0 + (rows % 2) * 24.0
    y = rows * 22.0
    channel_ids = np.arange(shank_ids.size)
    return {
        'channel_ids': channel_ids,
        'x': x,
        'y': y,
        'shank_ids': shank_ids,
        }


def make_values(n_samples, n_channels, fs):
    rng = np.random.default_rng(3)
    t = np.arange(n_samples, dtype=np.float32) / fs
    channels = np.arange(n_channels, dtype=np.float32)

    fast = np.sin(2.0 * np.pi * t[:, None] * (4.0 + channels[None, :] * 0.08))
    slow = np.sin(2.0 * np.pi * t[:, None] * 0.35 + channels[None, :] * 0.23)
    noise = rng.standard_normal((n_samples, n_channels))
    return (0.35 * fast + 0.12 * slow + 0.05 * noise).astype(np.float32)

if __name__ == "__main__":
    fs = 1_000
    geometry = make_geometry()
    values = make_values(
        n_samples=20 * fs,
        n_channels=len(geometry["channel_ids"]),
        fs=fs,
    )
    ephys = Ephys(
        "probe demo",
        values,
        geometry,
        fs=fs,
        chunk_samples=2 * fs,
        units="a.u.",
    )

    show([(ephys, EphysView())], title="Probe Widget Demo")
