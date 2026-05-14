from __future__ import annotations

import numpy as np
from cmap import Colormap
from imgui_bundle import imgui, implot

from viewer.stream import Units
from viewer.ui import draw_cursor

INDEX_LABEL = "(index)"
DEFAULT_CMAP = "cmocean:phase"


class UnitsSettings:
    def __init__(
        self,
        stream: Units | None = None,
        *,
        tick_height: float = 5.0,
        width: float = 1.0,
        spacing: float = 1.0,
        cmap: str = DEFAULT_CMAP,
        sort_by: str = "",
        color_by: str = "",
    ):
        self.tick_height = tick_height
        self.width = width
        self.spacing = spacing
        self.cmap = cmap
        self.sort_by = sort_by
        self.color_by = color_by
        if stream is not None and not color_by and "rate" in stream.metadata_keys:
            self.color_by = "rate"

    def unit_order(self, stream: Units) -> list[int]:
        if self.sort_by:
            return np.argsort(stream.metadata[self.sort_by]).tolist()
        return list(range(stream.n_units))

    def unit_colors(self, stream: Units, order: list[int]) -> np.ndarray:
        if self.color_by:
            values = stream.metadata[self.color_by][order]
            values = (values - values.min()) / max(values.max() - values.min(), 1e-9)
        else:
            values = np.linspace(0, 1, len(order), dtype=np.float32)

        try:
            rgba = Colormap(self.cmap)(values)
        except Exception:
            rgba = Colormap(DEFAULT_CMAP)(values)

        rgba = np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        return rgba

    def y_limits(self, n_units: int) -> tuple[float, float]:
        row_max = (n_units - 1) * self.spacing
        pad = max(0.5 * self.spacing, 0.25)
        return -pad, row_max + pad

    def draw_settings(self, name: str, stream: Units):
        meta_keys = [INDEX_LABEL] + stream.metadata_keys

        imgui.text("Sort by")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo(
            f"##sort_by_{name}",
            meta_keys.index(self.sort_by or INDEX_LABEL),
            meta_keys,
        )
        if changed:
            self.sort_by = "" if meta_keys[new_idx] == INDEX_LABEL else meta_keys[new_idx]

        imgui.text("Color by")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo(
            f"##color_by_{name}",
            meta_keys.index(self.color_by or INDEX_LABEL),
            meta_keys,
        )
        if changed:
            self.color_by = "" if meta_keys[new_idx] == INDEX_LABEL else meta_keys[new_idx]

        imgui.text("Colormap")
        imgui.set_next_item_width(-1)
        changed, new_cmap = imgui.input_text(f"##cmap_{name}", self.cmap)
        if changed:
            self.cmap = new_cmap

        imgui.separator()

        imgui.text("Tick height")
        imgui.set_next_item_width(-1)
        _, self.tick_height = imgui.slider_float(
            f"##tick_{name}", self.tick_height, 1.0, 20.0, "%.1f px"
        )
        imgui.text("Tick width")
        imgui.set_next_item_width(-1)
        _, self.width = imgui.slider_float(
            f"##width_{name}", self.width, 0.5, 4.0, "%.1f"
        )
        imgui.text("Unit spacing")
        imgui.set_next_item_width(-1)
        _, self.spacing = imgui.slider_float(
            f"##spacing_{name}", self.spacing, 0.1, 5.0, "%.1f"
        )

    def draw_plot(
        self,
        stream: Units,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
    ):
        order = self.unit_order(stream)
        unit_ids = [stream.unit_ids[i] for i in order]
        row_by_uid = {uid: row for row, uid in enumerate(unit_ids)}
        colors = self.unit_colors(stream, order)
        y_min, y_max = self.y_limits(len(unit_ids))

        if implot.begin_plot(f"{stream.name}"):
            implot.setup_axes("Time (s)", "Unit", 0, 0)
            implot.setup_axis_limits(implot.ImAxis_.y1, y_min, y_max, imgui.Cond_.once)
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

            if not chunks:
                implot.plot_text("Loading...", t, len(unit_ids) / 2)

            t0 = view_t0.value
            t1 = view_t1.value
            _plot_bounds(t0, t1, y_min, y_max)

            width_px = max(1, int(implot.get_plot_size().x))
            _draw_lines(stream, chunks, self, row_by_uid, colors, t0, t1, width_px)

            draw_cursor(t)
            implot.end_plot()


def _rgba_u32(rgba: np.ndarray) -> int:
    return imgui.color_convert_float4_to_u32(
        imgui.ImVec4(*(float(value) / 255.0 for value in rgba))
    )


def _draw_lines(
    stream: Units,
    chunks: list[dict],
    settings: UnitsSettings,
    row_by_uid: dict[int, int],
    colors: np.ndarray,
    t0: float,
    t1: float,
    width_px: int,
):
    colors_u32 = [_rgba_u32(color) for color in colors]
    draw_list = implot.get_plot_draw_list()
    half_height = settings.tick_height * 0.5

    implot.push_plot_clip_rect()
    try:
        for times, unit_ids in stream.iter_visible(chunks, t0, t1, width_px):
            for timestamp, uid in zip(times, unit_ids):
                row_idx = row_by_uid[uid]
                row = row_idx * settings.spacing
                p = implot.plot_to_pixels(timestamp, row)
                draw_list.add_line(
                    imgui.ImVec2(p.x, p.y - half_height),
                    imgui.ImVec2(p.x, p.y + half_height),
                    colors_u32[row_idx],
                    settings.width,
                )
    finally:
        implot.pop_plot_clip_rect()


def _plot_bounds(t0: float, t1: float, y_min: float, y_max: float):
    spec = implot.Spec()
    spec.marker = implot.Marker_.none
    spec.line_weight = 0.0
    xs = np.array([t0, t1])
    ys = np.array([y_min, y_max])
    implot.plot_scatter("##unit_bounds", xs, ys, spec=spec)