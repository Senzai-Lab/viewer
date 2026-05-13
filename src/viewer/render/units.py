from __future__ import annotations

import numpy as np
from cmap import Colormap
from imgui_bundle import imgui, implot

from viewer.stream import Units
from viewer.ui import draw_cursor


class UnitsSpec:
    def __init__(self, stream: Units):
        self.visible = True
        self.tick_height = 5.0
        self.thickness = 1.0
        self.unit_offset = 1.0
        self.cmap_name = "cmocean:phase"
        self.color_key = "rate"
        self.sort_key = ""
        self.unit_ids = np.asarray(stream.unit_ids, dtype=np.int64)
        self.metadata = stream.metadata

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def sorted_ids(self) -> np.ndarray:
        if self.sort_key and self.sort_key in self.metadata:
            meta_dict = self.metadata[self.sort_key]
            vals = np.array([float(meta_dict.get(str(uid), 0.0)) for uid in self.unit_ids])
            return self.unit_ids[np.argsort(vals)]
        return self.unit_ids

    def colors(self, sorted_ids: np.ndarray) -> list:
        if len(sorted_ids) == 0:
            return []

        if self.color_key and self.color_key in self.metadata:
            meta_dict = self.metadata[self.color_key]
            vals = np.array([float(meta_dict.get(str(uid), 0.0)) for uid in sorted_ids])
            vmin, vmax = vals.min(), vals.max()
            norm = (vals - vmin) / max(vmax - vmin, 1e-9)
        else:
            norm = np.linspace(0, 1, len(sorted_ids), dtype=np.float32)

        cm = Colormap(self.cmap_name)
        rgba = cm(norm.astype(np.float32))
        return [
            imgui.color_convert_float4_to_u32(
                imgui.ImVec4(float(r), float(g), float(b), float(a))
            )
            for r, g, b, a in rgba
        ]

    def y_limits(self) -> tuple[float, float]:
        if self.n_units == 0:
            return -1.0, 1.0

        row_max = (self.n_units - 1) * self.unit_offset
        pad = max(0.5 * self.unit_offset, 0.25)
        return -pad, row_max + pad


def make_spec(stream: Units) -> UnitsSpec:
    return UnitsSpec(stream)


def draw_settings(name: str, spec: UnitsSpec):
    _, spec.visible = imgui.checkbox(f"Visible##{name}", spec.visible)

    imgui.set_next_item_width(-1)
    _, spec.tick_height = imgui.slider_float(
        f"##tick_{name}", spec.tick_height, 1.0, 20.0, "Tick height: %.1f px"
    )
    imgui.set_next_item_width(-1)
    _, spec.thickness = imgui.slider_float(
        f"##thick_{name}", spec.thickness, 0.5, 4.0, "Thickness: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.unit_offset = imgui.slider_float(
        f"##offset_{name}", spec.unit_offset, 0.1, 5.0, "Unit offset: %.1f"
    )

    imgui.set_next_item_width(-1)
    changed, new_cmap = imgui.input_text(f"##cmap_{name}", spec.cmap_name)
    if imgui.is_item_hovered(imgui.HoveredFlags_.stationary):
        imgui.set_tooltip("Colormap name")
    if changed:
        spec.cmap_name = new_cmap

    meta_keys = ["(index)"] + [k for k in spec.metadata.keys()]

    sort_current = spec.sort_key if spec.sort_key else "(index)"
    sort_idx = meta_keys.index(sort_current) if sort_current in meta_keys else 0
    imgui.set_next_item_width(-1)
    changed, new_idx = imgui.combo(f"Sort by##{name}", sort_idx, meta_keys)
    if changed:
        spec.sort_key = "" if meta_keys[new_idx] == "(index)" else meta_keys[new_idx]

    color_current = spec.color_key if spec.color_key else "(index)"
    color_idx = meta_keys.index(color_current) if color_current in meta_keys else 0
    imgui.set_next_item_width(-1)
    changed, new_idx = imgui.combo(f"Color by##{name}", color_idx, meta_keys)
    if changed:
        spec.color_key = "" if meta_keys[new_idx] == "(index)" else meta_keys[new_idx]


def draw_plot(
        stream: Units,
        chunks: list[dict],
        spec: UnitsSpec,
        t: float,
        view_t0: implot.BoxedValue | None = None,
        view_t1: implot.BoxedValue | None = None,
):
    sorted_ids = spec.sorted_ids()
    uid_to_row = {int(uid): i for i, uid in enumerate(sorted_ids)}
    colors = spec.colors(sorted_ids)
    y_min, y_max = spec.y_limits()

    if implot.begin_plot(f"{stream.name}"):
        implot.setup_axes("Time (s)", "Unit", 0, 0)
        implot.setup_axis_limits(implot.ImAxis_.y1, y_min, y_max, imgui.Cond_.always)

        if view_t0 and view_t1:
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

        if not chunks:
            implot.plot_text("Loading...", t, spec.n_units / 2)

        width_px = max(implot.get_plot_size().x, 1.0)
        draw_list = implot.get_plot_draw_list()
        implot.push_plot_clip_rect()
        try:
            for times, unit_ids in stream.view(chunks, view_t0.value, view_t1.value, width_px):
                for timestamp, uid in zip(times, unit_ids):
                    row_idx = uid_to_row.get(int(uid))
                    if row_idx is None:
                        continue

                    row = float(row_idx) * spec.unit_offset
                    color = colors[row_idx]
                    half = spec.tick_height * 0.5
                    p = implot.plot_to_pixels(float(timestamp), row)
                    draw_list.add_line(
                        imgui.ImVec2(p.x, p.y - half),
                        imgui.ImVec2(p.x, p.y + half),
                        color,
                        spec.thickness,
                    )
        finally:
            implot.pop_plot_clip_rect()

        draw_cursor(t)
        implot.end_plot()
