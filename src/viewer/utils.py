from __future__ import annotations

from pathlib import Path

import numpy as np


SHANK_COLORS = {
    1: "#5081eb",
    2: "#f47c20",
    3: "#31a354",
    4: "#8e63ce",
}


def load_env(path: str | Path = ".env") -> dict[str, str]:
    path = Path(path)
    values = {}

    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno} is not KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        values[key] = value

    return values

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


def filter_units(
    spike_times,
    spike_units,
    unit_ids,
    output_path: str | Path,
    *,
    spike_times_filename: str = "spike_times.npy",
    spike_units_filename: str = "spike_units.npy",
    unit_ids_filename: str | None = "unit_ids.npy",
) -> dict[str, object]:
    # GPT5.5 made
    # TODO: Verify
    spike_times = np.asarray(spike_times)
    spike_units = np.asarray(spike_units).reshape(-1)
    unit_ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)

    if len(spike_times) != len(spike_units):
        raise ValueError("spike_times and spike_units must have the same length")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    keep = np.isin(spike_units, unit_ids)
    spike_times_path = output_path / spike_times_filename
    spike_units_path = output_path / spike_units_filename

    np.save(spike_times_path, spike_times[keep])
    np.save(spike_units_path, spike_units[keep])

    paths: dict[str, Path] = {
        "spike_times": spike_times_path,
        "spike_units": spike_units_path,
    }
    if unit_ids_filename is not None:
        unit_ids_path = output_path / unit_ids_filename
        np.save(unit_ids_path, unit_ids)
        paths["unit_ids"] = unit_ids_path

    return {
        "paths": paths,
        "n_spikes": int(np.count_nonzero(keep)),
        "n_units": int(len(unit_ids)),
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
