from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from viewer.ui import draw_cursor, setup_time_axis


class EventBars:
    def __init__(self, starts, ends, labels, *, colors=None, label_order=None):
        self.starts = np.asarray(starts, dtype=float)
        self.ends = np.asarray(ends, dtype=float)
        labels = [str(label) for label in labels]

        self.label_names = list(dict.fromkeys(labels))
        if label_order is not None:
            ordered = list(dict.fromkeys(str(label) for label in label_order))
            self.label_names = ordered + [
                label for label in self.label_names if label not in ordered
            ]

        self.colors = {
            str(label): _color_vec4(color)
            for label, color in (colors or {}).items()
        }
        row_by_label = {label: row for row, label in enumerate(self.label_names)}
        self.rows = np.asarray([row_by_label[label] for label in labels], dtype=int)
        self.y_ticks = [float(row) for row in range(len(self.label_names))]

        self.t_min = float(self.starts.min()) if self.starts.size else 0.0
        self.t_max = float(self.ends.max()) if self.ends.size else 1.0

    def draw(
        self,
        name: str,
        *,
        t: float | None = None,
        view_t0=None,
        view_t1=None,
        height: float = 140.0,
        bar_height: float = 0.45,
        legend: bool = False,
        time_axis: str = "clock",
    ):
        flags = (
            implot.Flags_.no_title
            | implot.Flags_.no_box_select
            | implot.Flags_.no_mouse_text
        )
        x_flags = (
            implot.AxisFlags_.no_label
            | implot.AxisFlags_.no_tick_marks
            | implot.AxisFlags_.no_tick_labels
            )
        y_flags = (
            implot.AxisFlags_.no_label
            | implot.AxisFlags_.no_grid_lines
        )

        if not implot.begin_plot(name, imgui.ImVec2(-1.0, height), flags):
            return

        setup_time_axis("", time_axis=time_axis, x_label="", x_flags=x_flags, y_flags=y_flags)
        implot.setup_axis_limits(
            implot.ImAxis_.y1,
            -0.5,
            max(0.5, len(self.label_names) - 0.5),
            imgui.Cond_.always,
        )

        if self.label_names:
            implot.setup_axis_ticks(
                implot.ImAxis_.y1,
                self.y_ticks,
                self.label_names,
                False,
            )

        if view_t0 is not None and view_t1 is not None:
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)
        else:
            pad = max((self.t_max - self.t_min) * 0.02, 1e-6)
            implot.setup_axis_limits(
                implot.ImAxis_.x1,
                self.t_min - pad,
                self.t_max + pad,
                imgui.Cond_.once,
            )

        row_colors = []
        for row, label in enumerate(self.label_names):
            color = self.colors.get(label)
            if color is None:
                color = implot.get_colormap_color(row)
            row_colors.append(imgui.ImVec4(color.x, color.y, color.z, 0.88))
        fill_colors = [imgui.color_convert_float4_to_u32(color) for color in row_colors]

        if legend:
            implot.setup_legend(
                implot.Location_.west,
                implot.LegendFlags_.outside | implot.LegendFlags_.no_buttons,
            )
            for row, label in enumerate(self.label_names):
                implot.plot_dummy(
                    label,
                    implot.Spec(line_color=row_colors[row], line_weight=6.0),
                )

        limits = implot.get_plot_limits()
        x_min = float(limits.x.min)
        x_max = float(limits.x.max)
        half_h = max(0.01, float(bar_height) * 0.5)
        draw_list = implot.get_plot_draw_list()
        edge_color = imgui.IM_COL32(255, 255, 255, 42)

        implot.push_plot_clip_rect()
        try:
            visible = np.flatnonzero((self.ends > x_min) & (self.starts < x_max))
            for idx in visible:
                row = int(self.rows[idx])
                x0 = max(float(self.starts[idx]), x_min)
                x1 = min(float(self.ends[idx]), x_max)
                p0 = implot.plot_to_pixels(x0, row - half_h)
                p1 = implot.plot_to_pixels(x1, row + half_h)
                a = imgui.ImVec2(min(p0.x, p1.x), min(p0.y, p1.y))
                b = imgui.ImVec2(max(p0.x, p1.x), max(p0.y, p1.y))
                draw_list.add_rect_filled(a, b, fill_colors[row], 2.0)
                draw_list.add_rect(a, b, edge_color, 3.0, 0, 1.0)
        finally:
            implot.pop_plot_clip_rect()

        #if t is not None:
        #    draw_cursor(float(t))
        implot.end_plot()

    def draw_overlay(self, alpha: float = 0.16):
        limits = implot.get_plot_limits()
        x_min = float(limits.x.min)
        x_max = float(limits.x.max)
        y_min = float(limits.y.min)
        y_max = float(limits.y.max)
        draw_list = implot.get_plot_draw_list()

        implot.push_plot_clip_rect()
        try:
            visible = np.flatnonzero((self.ends > x_min) & (self.starts < x_max))
            for idx in visible:
                row = int(self.rows[idx])
                color = self.colors.get(self.label_names[row])
                if color is None:
                    color = implot.get_colormap_color(row)
                color = imgui.color_convert_float4_to_u32(
                    imgui.ImVec4(color.x, color.y, color.z, alpha)
                )
                p0 = implot.plot_to_pixels(max(float(self.starts[idx]), x_min), y_min)
                p1 = implot.plot_to_pixels(min(float(self.ends[idx]), x_max), y_max)
                draw_list.add_rect_filled(
                    imgui.ImVec2(min(p0.x, p1.x), min(p0.y, p1.y)),
                    imgui.ImVec2(max(p0.x, p1.x), max(p0.y, p1.y)),
                    color,
                )
        finally:
            implot.pop_plot_clip_rect()


def _color_vec4(color) -> imgui.ImVec4:
    if isinstance(color, int):
        return imgui.color_convert_u32_to_float4(color)

    if isinstance(color, str):
        text = color.removeprefix("#")
        if len(text) == 6:
            text += "ff"
        rgba = [int(text[i:i + 2], 16) / 255.0 for i in range(0, 8, 2)]
    else:
        rgba = list(color)
        if len(rgba) == 3:
            rgba.append(1.0)
        if max(rgba) > 1.0:
            rgba = [value / 255.0 for value in rgba]

    return imgui.ImVec4(*rgba[:4])
