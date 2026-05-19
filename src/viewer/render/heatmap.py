from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui, implot

from viewer.ui import draw_cursor, setup_time_axis


def iter_heatmap_visible(stream, chunks, t0: float, t1: float):
    """Clip continuous chunks as image columns instead of line samples."""
    for chunk in chunks:
        chunk_t0 = chunk["t_start"]
        fs = chunk["fs"]
        dt = chunk["dt"]
        data = chunk["data"]
        n = data.shape[0]

        i0 = math.floor((t0 - chunk_t0) * fs) - 1
        i1 = math.ceil((t1 - chunk_t0) * fs) + 2
        i0 = max(0, i0)
        i1 = min(n, i1)

        if i0 >= i1:
            continue

        sample_start = chunk["sample_start"] + i0
        sample_stop = chunk["sample_start"] + i1
        yield {
            "data": data[i0:i1],
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "x0": chunk_t0 + (i0 - 0.5) * dt,
            "x1": chunk_t0 + (i1 - 0.5) * dt,
        }


class HeatmapView:
    def __init__(
        self,
        y=None,
        *,
        y_edges=None,
        y_label: str = "Bin",
        cmap: str = "Viridis",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
        auto_scale: bool = True,
        label_fmt: str = "",
    ):
        self.y_edges = None
        if y_edges is not None:
            self.y_edges = np.asarray(y_edges, dtype=float)
        elif y is not None:
            self.y_edges = _centers_to_edges(np.asarray(y, dtype=float))

        self.y_label = y_label
        self.cmap = cmap
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.auto_scale = auto_scale
        self.label_fmt = label_fmt

    def _y_edges(self, stream, chunks, n_bins: int) -> np.ndarray:
        if self.y_edges is not None:
            return self.y_edges

        for chunk in chunks:
            y = chunk.get("y")
            if y is not None:
                return _centers_to_edges(np.asarray(y, dtype=float))

        y = getattr(stream, "y", None)
        if y is not None:
            return _centers_to_edges(np.asarray(y, dtype=float))

        return np.arange(n_bins + 1, dtype=float) - 0.5

    def draw_settings(self, stream, cache):
        name = stream.name
        imgui.text("Colormap")
        imgui.set_next_item_width(-1)
        _, self.cmap = imgui.input_text(f"##cmap_{name}", self.cmap)

        imgui.text("Cell labels")
        imgui.set_next_item_width(-1)
        _, self.label_fmt = imgui.input_text(f"##label_fmt_{name}", self.label_fmt)

        _, self.auto_scale = imgui.checkbox(f"Auto scale##autoscale_{name}", self.auto_scale)
        if not self.auto_scale:
            imgui.set_next_item_width(-1)
            _, self.scale_min = imgui.drag_float(
                f"##scale_min_{name}", self.scale_min, 0.01, 0.0, 0.0, "Min: %.3f"
            )
            imgui.set_next_item_width(-1)
            _, self.scale_max = imgui.drag_float(
                f"##scale_max_{name}", self.scale_max, 0.01, 0.0, 0.0, "Max: %.3f"
            )

    def draw_plot(
        self,
        stream,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
        overlays=(),
        *,
        time_axis: str = "clock",
    ):
        items = list(iter_heatmap_visible(stream, chunks, view_t0.value, view_t1.value))
        if items:
            n_bins = items[0]["data"].shape[1]
        else:
            n_bins = getattr(stream, "n_channels", 1)
        y_edges = self._y_edges(stream, chunks, n_bins)
        scale_min, scale_max = self._scale(items)

        if implot.begin_plot(f"{stream.name}", flags=implot.Flags_.no_legend):
            setup_time_axis(self.y_label, time_axis=time_axis)
            implot.setup_axis_limits(
                implot.ImAxis_.y1,
                float(y_edges[0]),
                float(y_edges[-1]),
                imgui.Cond_.once,
            )
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

            if not chunks:
                implot.plot_text("Loading...", t, float(0.5 * (y_edges[0] + y_edges[-1])))

            implot.push_colormap(self.cmap)
            try:
                for item in items:
                    # ImPlot heatmaps draw the first matrix row at the top.
                    values = np.ascontiguousarray(item["data"].T[::-1])
                    implot.plot_heatmap(
                        f"##heatmap_{stream.name}_{item['sample_start']}",
                        values,
                        scale_min,
                        scale_max,
                        self.label_fmt,
                        implot.Point(float(item["x0"]), float(y_edges[0])),
                        implot.Point(float(item["x1"]), float(y_edges[-1])),
                    )
            finally:
                implot.pop_colormap()

            for overlay in overlays:
                overlay.draw_overlay()
            draw_cursor(t)
            implot.end_plot()

    def _scale(self, items) -> tuple[float, float]:
        if not self.auto_scale or not items:
            return self.scale_min, self.scale_max

        mins = [float(np.nanmin(item["data"])) for item in items]
        maxs = [float(np.nanmax(item["data"])) for item in items]
        self.scale_min = min(mins)
        self.scale_max = max(maxs)
        return self.scale_min, self.scale_max


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    mids = 0.5 * (values[:-1] + values[1:])
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = values[0] - (mids[0] - values[0])
    edges[-1] = values[-1] + (values[-1] - mids[-1])
    return edges
