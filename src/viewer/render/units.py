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
        metadata: dict | None = None,
        unit_ids=None,
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
        self.metadata = {} if metadata is None else metadata
        self.unit_ids = self._unit_ids(unit_ids)
        self.metadata_values = {
            key: _metadata_values(values, self.unit_ids)
            for key, values in self.metadata.items()
            if key != "unit_ids"
        }
        self.metadata_keys = list(self.metadata_values)

        if not self.color_by and "rate" in self.metadata_values:
            self.color_by = "rate"

        self.order = np.arange(len(self.unit_ids))
        self.plot_unit_ids = self.unit_ids
        self.row_by_uid = {int(uid): int(row) for row, uid in enumerate(self.plot_unit_ids)}
        self.colors = np.zeros((len(self.unit_ids), 4), dtype=np.uint8)
        self.colors_u32: list[int] = []
        self._dirty = True
        self._build_plot_data()

    def _unit_ids(self, unit_ids) -> np.ndarray:
        ids = unit_ids
        if ids is None:
            ids = self.metadata.get("unit_ids")
        if ids is None and "rate" in self.metadata:
            ids = self.metadata["rate"].keys()
        if ids is None:
            return np.array([], dtype=int)
        return np.asarray([int(uid) for uid in ids])

    def _build_plot_data(self):
        if len(self.unit_ids) == 0:
            self.order = np.array([], dtype=int)
            self.plot_unit_ids = self.unit_ids
            self.row_by_uid = {}
            self.colors = np.zeros((0, 4), dtype=np.uint8)
            self.colors_u32 = []
            self._dirty = False
            return

        if self.sort_by:
            self.order = np.argsort(self.metadata_values[self.sort_by])
        else:
            self.order = np.arange(len(self.unit_ids))

        self.plot_unit_ids = self.unit_ids[self.order]
        self.row_by_uid = {
            int(uid): int(row)
            for row, uid in enumerate(self.plot_unit_ids)
        }

        if self.color_by:
            values = self.metadata_values[self.color_by][self.order]
            values = (values - values.min()) / max(values.max() - values.min(), 1e-9)
        else:
            values = np.linspace(0, 1, len(self.order), dtype=np.float32)

        try:
            rgba = Colormap(self.cmap)(values)
        except Exception:
            rgba = Colormap(DEFAULT_CMAP)(values)

        rgba = np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        self.colors = rgba
        self.colors_u32 = [_rgba_u32(color) for color in rgba]
        self._dirty = False

    def _ensure_stream_units(self, stream: Units):
        if len(self.unit_ids):
            return
        self.unit_ids = np.asarray(stream.unit_ids)
        self.order = np.arange(len(self.unit_ids))
        self.plot_unit_ids = self.unit_ids
        self.row_by_uid = {int(uid): int(row) for row, uid in enumerate(self.plot_unit_ids)}
        self._dirty = True

    def y_limits(self, n_units: int) -> tuple[float, float]:
        row_max = (n_units - 1) * self.spacing
        pad = max(0.5 * self.spacing, 0.25)
        return -pad, row_max + pad

    def draw_settings(self, name: str):
        meta_keys = [INDEX_LABEL] + self.metadata_keys

        imgui.text("Sort by")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo(
            f"##sort_by_{name}",
            meta_keys.index(self.sort_by) if self.sort_by in self.metadata_keys else 0,
            meta_keys,
        )
        if changed:
            self.sort_by = "" if meta_keys[new_idx] == INDEX_LABEL else meta_keys[new_idx]
            self._dirty = True

        imgui.text("Color by")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo(
            f"##color_by_{name}",
            meta_keys.index(self.color_by) if self.color_by in self.metadata_keys else 0,
            meta_keys,
        )
        if changed:
            self.color_by = "" if meta_keys[new_idx] == INDEX_LABEL else meta_keys[new_idx]
            self._dirty = True

        imgui.text("Colormap")
        imgui.set_next_item_width(-1)
        changed, new_cmap = imgui.input_text(f"##cmap_{name}", self.cmap)
        if changed:
            self.cmap = new_cmap
            self._dirty = True

        imgui.separator()

        imgui.text("Tick height")
        imgui.set_next_item_width(-1)
        _, self.tick_height = imgui.drag_float(
            f"##tick_{name}", self.tick_height, 0.2, 1.0, 20.0, "%.1f px"
        )
        imgui.text("Tick width")
        imgui.set_next_item_width(-1)
        _, self.width = imgui.drag_float(
            f"##width_{name}", self.width, 0.05, 0.5, 4.0, "%.1f"
        )
        imgui.text("Unit spacing")
        imgui.set_next_item_width(-1)
        _, self.spacing = imgui.drag_float(
            f"##spacing_{name}", self.spacing, 0.05, 0.1, 5.0, "%.1f"
        )

    def draw_plot(
        self,
        stream: Units,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
    ):
        self._ensure_stream_units(stream)
        if self._dirty:
            self._build_plot_data()

        y_min, y_max = self.y_limits(len(self.plot_unit_ids))

        if implot.begin_plot(f"{stream.name}"):
            implot.setup_axes("Time (s)", "Unit", 0, 0)
            implot.setup_axis_limits(implot.ImAxis_.y1, y_min, y_max, imgui.Cond_.always)
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

            if not chunks:
                implot.plot_text("Loading...", t, len(self.plot_unit_ids) / 2)

            t0 = view_t0.value
            t1 = view_t1.value

            width_px = max(1, int(implot.get_plot_size().x))
            _draw_lines(stream, chunks, self, t0, t1, width_px)

            draw_cursor(t)
            implot.end_plot()


def _metadata_values(values, unit_ids: np.ndarray) -> np.ndarray:
    if hasattr(values, "keys"):
        return np.array([
            values[int(uid)] if int(uid) in values else values[str(uid)]
            for uid in unit_ids
        ], dtype=float)
    return np.asarray(values, dtype=float)


def _rgba_u32(rgba: np.ndarray) -> int:
    return imgui.color_convert_float4_to_u32(
        imgui.ImVec4(*(float(value) / 255.0 for value in rgba))
    )


def _draw_lines(
    stream: Units,
    chunks: list[dict],
    settings: UnitsSettings,
    t0: float,
    t1: float,
    width_px: int,
):
    draw_list = implot.get_plot_draw_list()
    half_height = settings.tick_height * 0.5

    implot.push_plot_clip_rect()
    try:
        for times, unit_ids in stream.iter_visible(chunks, t0, t1, width_px):
            for timestamp, uid in zip(times, unit_ids):
                row_idx = settings.row_by_uid[int(uid)]
                row = row_idx * settings.spacing
                p = implot.plot_to_pixels(timestamp, row)
                draw_list.add_line(
                    imgui.ImVec2(p.x, p.y - half_height),
                    imgui.ImVec2(p.x, p.y + half_height),
                    settings.colors_u32[row_idx],
                    settings.width,
                )
    finally:
        implot.pop_plot_clip_rect()
