from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from viewer.stream import TimeSeries
from viewer.ui import draw_cursor


class TimeSeriesSettings:
    def __init__(
        self,
        *,
        width: float = 1.0,
        gain: float = 1.0,
        spacing: float = 1.0,
    ):
        self.width = width
        self.gain = gain
        self.spacing = spacing

    def y_limits(self, n_channels: int) -> tuple[float, float]:
        row_max = (n_channels - 1) * self.spacing
        pad = max(self.gain, 0.5 * self.spacing, 0.25)
        return -pad, row_max + pad

    def draw_settings(self, name: str):
        imgui.set_next_item_width(-1)
        _, self.width = imgui.slider_float(
            f"##width_{name}", self.width, 0.5, 3.0, "Line width: %.1f"
        )
        imgui.set_next_item_width(-1)
        _, self.gain = imgui.slider_float(
            f"##gain_{name}", self.gain, 0.1, 20.0, "Gain: %.1f"
        )
        imgui.set_next_item_width(-1)
        _, self.spacing = imgui.slider_float(
            f"##spacing_{name}", self.spacing, 0.0, 100.0, "Channel spacing: %.0f"
        )

    def draw_plot(
        self,
        stream: TimeSeries,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
    ):
        y_min, y_max = self.y_limits(stream.n_channels)

        if implot.begin_plot(f"{stream.name}"):
            implot.setup_axes("Time (s)", "Value", 0, 0)
            implot.setup_axis_limits(implot.ImAxis_.y1, y_min, y_max, imgui.Cond_.once)
            implot.setup_axis_zoom_constraints(implot.ImAxis_.y1, 1e-6, 1e12)
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

            if not chunks:
                implot.plot_text("Loading...", t, 0.0)

            width_px = implot.get_plot_size().x
            ch_colors = [implot.get_colormap_color(ch) for ch in range(stream.n_channels)]
            imspec = implot.Spec(line_weight=self.width)

            for data, xstart, xscale in stream.iter_visible(
                chunks, view_t0.value, view_t1.value, width_px
            ):
                for ch in range(stream.n_channels):
                    ys = np.ascontiguousarray(data[:, ch])
                    ys = ys * self.gain + ch * self.spacing
                    imspec.line_color = ch_colors[ch]
                    implot.plot_line(
                        f"ch{ch}",
                        ys,
                        xscale=xscale,
                        xstart=xstart,
                        spec=imspec,
                    )

            draw_cursor(t)
            implot.end_plot()
