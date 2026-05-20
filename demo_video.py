from pathlib import Path

import numpy as np
import zarr

from viewer import RasterView, Spikes, Video, VideoView, show


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[0]
    video_path = root / "scripts" / "data" / "A5044-240404A_wake.avi"
    zarr_path = root / "scripts" / "exp1.zarr"

    video = Video("wake video", video_path, chunk_duration=1.0, scale=0.5)

    data = zarr.open(zarr_path, mode="r")
    units = data["units"]
    metadata = dict(units.attrs)
    unit_ids = metadata.get("unit_ids")
    if unit_ids is None and "rate" in metadata:
        unit_ids = [int(uid) for uid in metadata["rate"]]

    spike_times = np.asarray(units["spike_times"])
    spike_units = np.asarray(units["spike_units"])
    keep = spike_times <= video.t_max
    spikes = Spikes(
        name="units",
        ts=spike_times[keep],
        spike_units=spike_units[keep],
        chunk_duration=5.0,
        unit_ids=unit_ids,
    )

    show(
        [
            (video, VideoView()),
            (spikes, RasterView(metadata=metadata, unit_ids=unit_ids)),
        ],
        title="Video + Raster Demo",
        span=8.0,
        max_workers=2,
    )
