from __future__ import annotations

from pathlib import Path

import numpy as np


SHANK_COLORS = {
    1: "#5081eb",
    2: "#f47c20",
    3: "#31a354",
    4: "#8e63ce",
}


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


def load_probeinterface(
    path: str | Path,
    *,
    shank_gap: float = 4.0,
) -> dict[str, np.ndarray]:
    from probeinterface import read_probeinterface

    df = read_probeinterface(path).to_dataframe().reset_index(drop=True)
    shank_ids = df["shank_ids"].to_numpy(dtype=int)
    df["shank_ids"] = shank_ids
    display_index = np.empty(len(df), dtype=np.float32)
    colors = {
        shank: SHANK_COLORS[i + 1]
        for i, shank in enumerate(np.unique(shank_ids))
    }
    cursor = 0.0

    for _, group in df.groupby("shank_ids", sort=True):
        order = group.sort_values("y", ascending=False).index.to_numpy()
        display_index[order] = cursor + np.arange(len(order), dtype=np.float32)
        cursor += len(order) + shank_gap

    return {
        "channel_ids": np.arange(len(df)),
        "shank_ids": shank_ids,
        "x": df["x"].to_numpy(),
        "y": df["y"].to_numpy(),
        "display_index": display_index,
        "colors": np.array(
            [colors[shank] for shank in shank_ids],
            dtype=object,
        ),
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
