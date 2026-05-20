from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from imgui_bundle import imgui, immvision, implot

if TYPE_CHECKING:
    from viewer.stream import Video


class VideoView:
    def __init__(self, *, fill: bool = False):
        self.fill = fill
        self.texture = None
        self.frame_idx = None

    def draw_settings(self, stream: Video, cache):
        _, self.fill = imgui.checkbox(f"Fill frame##fill_{stream.name}", self.fill)
        imgui.text_disabled(f"{stream.width} x {stream.height} px")
        imgui.text_disabled(f"{stream.fs:.3g} fps")

    def draw_plot(
        self,
        stream: Video,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
        overlays=(),
        *,
        time_axis: str = "clock",
    ):
        flags = (
            implot.Flags_.no_title
            | implot.Flags_.no_legend
            | implot.Flags_.no_menus
            | implot.Flags_.no_mouse_text
            | implot.Flags_.no_box_select
        )
        x_axis_flags = (
            implot.AxisFlags_.no_decorations
            | implot.AxisFlags_.no_grid_lines
            | implot.AxisFlags_.no_menus
        )
        y_axis_flags = x_axis_flags | implot.AxisFlags_.lock

        if implot.begin_plot(f"{stream.name}", flags=flags):
            implot.setup_axes("", "", x_axis_flags, y_axis_flags)
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 1.0, imgui.Cond_.always)

            item = self._frame_at(chunks, t)
            if item is None:
                implot.plot_text("Loading...", t, 0.5)
            else:
                frame, frame_idx = item
                self._draw_frame(frame, frame_idx)

            implot.end_plot()

    def _frame_at(self, chunks: list[dict], t: float):
        best = None
        best_t = -math.inf

        for chunk in chunks:
            times = chunk["ts"]
            if len(times) == 0:
                continue

            candidate = int(np.searchsorted(times, t, side="right")) - 1
            if candidate < 0:
                continue

            frame_t = float(times[candidate])
            if frame_t > best_t:
                best_t = frame_t
                best = (
                    chunk["data"][candidate],
                    int(chunk["frame_idx"][candidate]),
                )

        return best

    def _draw_frame(self, frame: np.ndarray, frame_idx: int):
        if self.texture is None:
            self.texture = immvision.GlTexture(frame)
        elif self.frame_idx != frame_idx:
            self.texture.update_from_image(frame)
        self.frame_idx = frame_idx

        plot_pos = implot.get_plot_pos()
        plot_size = implot.get_plot_size()

        frame_h, frame_w = frame.shape[:2]
        sx = plot_size.x / frame_w
        sy = plot_size.y / frame_h
        scale = max(sx, sy) if self.fill else min(sx, sy)
        draw_w = frame_w * scale
        draw_h = frame_h * scale

        x0 = plot_pos.x + 0.5 * (plot_size.x - draw_w)
        y0 = plot_pos.y + 0.5 * (plot_size.y - draw_h)
        p0 = imgui.ImVec2(x0, y0)
        p1 = imgui.ImVec2(x0 + draw_w, y0 + draw_h)

        draw = implot.get_plot_draw_list()
        bg0 = imgui.ImVec2(plot_pos.x, plot_pos.y)
        bg1 = imgui.ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y)
        draw.add_rect_filled(bg0, bg1, imgui.IM_COL32(0, 0, 0, 255))

        implot.push_plot_clip_rect()
        try:
            draw.add_image(imgui.ImTextureRef(self.texture.texture_id), p0, p1)
        finally:
            implot.pop_plot_clip_rect()
