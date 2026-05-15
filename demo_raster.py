import zarr

from viewer import Units, UnitsSettings, run_viewer


if __name__ == "__main__":
    root = zarr.open("/Users/iii9781/viewer/scripts/exp1.zarr", mode="r")

    units_grp = root["units"]
    metadata = dict(units_grp.attrs)
    units = Units(
        name="units",
        ts=units_grp["spike_times"],
        values=units_grp["spike_units"],
        metadata=metadata,
    )

    run_viewer([(units, UnitsSettings(metadata=metadata))], title="Raster Viewer")
