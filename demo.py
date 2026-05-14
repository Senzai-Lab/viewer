import zarr

from viewer import TimeSeries, TimeSeriesSettings, Units, UnitsSettings, run_viewer

root = zarr.open("/Users/iii9781/viewer/scripts/exp1.zarr", mode="r")

ephys = TimeSeries(
    name="ephys",
    values=root["ephys/values"],
    ts=root["ephys/ts"],
    fs=root["ephys"].attrs["fs"],
    chunk_samples=root["ephys/values"].chunks[0],
)

pupil = TimeSeries(
    name="sine",
    values=root["behavior/pupil/values"],
    ts=root["behavior/pupil/ts"],
    fs=root["behavior/pupil"].attrs["fs"],
    chunk_samples=root["behavior/pupil/values"].chunks[0],
)

units = Units(
    name="units",
    ts=root["units/spike_times"],
    values=root["units/spike_units"],
    metadata=dict(root["units"].attrs),
)

run_viewer(
    [
        (ephys, TimeSeriesSettings()),
        (pupil, TimeSeriesSettings()),
        (units, UnitsSettings(units)),
    ],
    span=5.0,
)
