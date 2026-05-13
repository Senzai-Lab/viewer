import zarr

from viewer import TimeSeries, run_viewer


if __name__ == "__main__":
    root = zarr.open("/Users/iii9781/viewer/scripts/exp1.zarr", mode="r")

    ephys_grp = root["ephys"]
    ephys = TimeSeries(
        name="ephys",
        values=ephys_grp["values"],
        ts=ephys_grp["ts"],
        fs=ephys_grp.attrs["fs"],
        chunk_samples=ephys_grp["values"].chunks[0],
    )

    pupil_grp = root["behavior/pupil"]
    pupil = TimeSeries(
        name="pupil",
        values=pupil_grp["values"],
        ts=pupil_grp["ts"],
        fs=pupil_grp.attrs["fs"],
        chunk_samples=pupil_grp["values"].chunks[0],
    )

    run_viewer([ephys, pupil])
