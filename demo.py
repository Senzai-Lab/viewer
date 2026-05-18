import zarr

from viewer import EventBars, TimeSeries, TimeSeriesSettings, Units, UnitsSettings, run_viewer

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

units_metadata = dict(root["units"].attrs)
units = Units(
    name="units",
    ts=root["units/spike_times"],
    spike_units=root["units/spike_units"],
    unit_ids=units_metadata.get("unit_ids"),
)

t0 = float(root["ephys/ts"][0])
t1 = float(root["ephys/ts"][-1])
step = (t1 - t0) / 6.0
behavior = EventBars(
    starts=[t0 + i * step for i in range(6)],
    ends=[t0 + (i + 1) * step for i in range(6)],
    labels=["quiet", "run", "groom", "quiet", "run", "rest"],
    label_order=["quiet", "run", "groom", "rest"],
    colors={
        "quiet": "#6b8cff",
        "run": "#47d18c",
        "groom": "#f26d7d",
        "rest": "#ffc857",
    },
)

run_viewer(
    [
        (ephys, TimeSeriesSettings()),
        (pupil, TimeSeriesSettings()),
        (units, UnitsSettings(metadata=units_metadata)),
    ],
    event_bars=behavior,
    span=5.0,
)
