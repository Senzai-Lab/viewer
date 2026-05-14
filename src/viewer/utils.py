from __future__ import annotations

from pathlib import Path

import numpy as np


def load_prb(ks_path: str | Path) -> dict[str, np.ndarray]:
    ks_path = Path(ks_path)
    ch_map = np.load(ks_path / "channel_map.npy").astype(np.uint16)
    ch_pos = np.load(ks_path / "channel_positions.npy")
    ch_shanks = np.load(ks_path / "channel_shanks.npy").astype(np.uint8)

    return {
        "channel_ids": ch_map,
        "shank_ids": ch_shanks,
        "x": ch_pos[:, 0],
        "y": ch_pos[:, 1],
    }


def format_bytes(nbytes: int | None) -> str:
    if nbytes is None:
        return "unbounded"

    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def format_seconds(seconds: float) -> str:
    value = abs(seconds)

    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"

    for unit, scale in (("s", 60.0), ("min", 60.0), ("h", 24.0)):
        if value < scale:
            return f"{value:.1f} {unit}"
        value /= scale
    return f"{value:.1f} d"
