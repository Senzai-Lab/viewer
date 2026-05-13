from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from viewer.stream import TimeSeries
from viewer.ui import draw_cursor


class TimeSeriesSpec:
    def __init__(self):
        self.visible = True
        self.line_weight = 1.0
        self.gain = 1.0
        self.ch_offset = 1.0
        self.ch_colors = []

    def to_implot_spec(self) -> implot.Spec:
        return implot.Spec(line_weight=self.line_weight)


def make_spec(stream: TimeSeries) -> TimeSeriesSpec:
    return TimeSeriesSpec()


def draw_settings(name: str, spec: TimeSeriesSpec):
    _, spec.visible = imgui.checkbox(f"Visible##{name}", spec.visible)

    imgui.set_next_item_width(-1)
    _, spec.line_weight = imgui.slider_float(
        f"##lw_{name}", spec.line_weight, 0.5, 3.0, "Line weight: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.gain = imgui.slider_float(
        f"##gain_{name}", spec.gain, 0.1, 20.0, "Gain: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.ch_offset = imgui.slider_float(
        f"##offset_{name}", spec.ch_offset, 0.0, 100.0, "Ch offset: %.0f"
    )


def draw_plot(
        stream: TimeSeries,
        chunks: list[dict],
        spec: TimeSeriesSpec,
        t: float,
        view_t0: implot.BoxedValue | None = None,
        view_t1: implot.BoxedValue | None = None,
):
    if implot.begin_plot(f"{stream.name}"):
        y_flags = implot.AxisFlags_.auto_fit if chunks else 0
        implot.setup_axes("Time (s)", "Value", 0, y_flags)
        implot.setup_axis_zoom_constraints(implot.ImAxis_.y1, 1e-6, 1e12)
        if view_t0 and view_t1:
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

        if not chunks:
            implot.plot_text("Loading...", t, 0.0)

        width_px = implot.get_plot_size().x
        spec.ch_colors = [implot.get_colormap_color(ch) for ch in range(stream.n_channels)]
        imspec = spec.to_implot_spec()

        for data, xstart, xscale in stream.view(chunks, view_t0.value, view_t1.value, width_px):
            for ch in range(stream.n_channels):
                ys = np.ascontiguousarray(data[:, ch])
                ys = ys * spec.gain + ch * spec.ch_offset
                imspec.line_color = spec.ch_colors[ch]
                implot.plot_line(
                    f"ch{ch}",
                    ys,
                    xscale=xscale,
                    xstart=xstart,
                    spec=imspec,
                )

        draw_cursor(t)
        implot.end_plot()
