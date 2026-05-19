import zarr

from viewer import RasterView, Spikes, show


if __name__ == "__main__":
    root = zarr.open("/Users/iii9781/viewer/scripts/exp1.zarr", mode="r")

    units_grp = root["units"]
    metadata = dict(units_grp.attrs)
    spikes = Spikes(
        name="units",
        ts=units_grp["spike_times"],
        spike_units=units_grp["spike_units"],
        unit_ids=metadata.get("unit_ids"),
    )

    show([(spikes, RasterView(metadata=metadata))], title="Raster Viewer")
