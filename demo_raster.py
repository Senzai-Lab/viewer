import zarr

from viewer import Units, run_viewer


if __name__ == "__main__":
    root = zarr.open("/Users/iii9781/viewer/scripts/exp1.zarr", mode="r")

    units_grp = root["units"]
    units = Units(
        name="units",
        ts=units_grp["spike_times"],
        values=units_grp["spike_units"],
        metadata=dict(units_grp.attrs),
    )

    run_viewer([units], title="Raster Viewer")
