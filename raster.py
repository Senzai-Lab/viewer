from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pynapple as nap
from cmap import Colormap
from imgui_bundle import imgui, immapp, implot

from viewer import PlaybackController, ns_to_seconds, seconds_to_ns
from viewer.ui_old import TRANSPORT_HEIGHT, draw_cursor, draw_transport, handle_shortcuts, sync_axis_links
from viewer.utils import MIN_VIEW_SPAN_S, max_view_span_s


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "A5044-240404A_wake.nwb"
DEFAULT_VIEW_SPAN_S = 8.0
DEFAULT_COLORMAP = "cmocean:phase"
RASTER_TICK_HEIGHT_PX = 10.0
RASTER_TICK_THICKNESS = 1.0


def load_units(data_path: Path, *, units_key: str = "units") -> nap.TsGroup:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing NWB file: {data_path}")
    data = nap.load_file(str(data_path))
    units = data[units_key]
    if not isinstance(units, nap.TsGroup):
        raise TypeError(f"Expected {units_key!r} to be a TsGroup, got {type(units)!r}")
    return units


def make_unit_colors(n_units: int, cmap_name: str = DEFAULT_COLORMAP) -> tuple[int, ...]:
    cmap = Colormap(cmap_name)
    rgba = np.ascontiguousarray(cmap(np.linspace(0.0, 1.0, n_units, dtype=np.float32)))
    return tuple(
        imgui.color_convert_float4_to_u32(
            imgui.ImVec4(float(r), float(g), float(b), float(a))
        )
        for r, g, b, a in rgba
    )


def time_bounds(spike_times_s: tuple[np.ndarray, ...]) -> tuple[float, float]:
    t_min_s = float("inf")
    t_max_s = float("-inf")
    for times_s in spike_times_s:
        if times_s.size == 0:
            continue
        t_min_s = min(t_min_s, float(times_s[0]))
        t_max_s = max(t_max_s, float(times_s[-1]))

    if not np.isfinite(t_min_s) or not np.isfinite(t_max_s):
        return 0.0, 1.0
    if t_max_s <= t_min_s:
        return t_min_s, t_min_s + 1e-3
    return t_min_s, t_max_s


class AppState:
    def __init__(
        self,
        data_path: Path | None = DEFAULT_DATA_PATH,
        *,
        units: nap.TsGroup | None = None,
        cmap_name: str = DEFAULT_COLORMAP,
    ) -> None:
        self.data_path = data_path
        self.units = units if units is not None else load_units(data_path or DEFAULT_DATA_PATH)
        self.unit_ids = tuple(self.units.keys())
        if not self.unit_ids:
            raise ValueError("units is empty")

        self.spike_times_s = tuple(
            np.ascontiguousarray(self.units[unit_id].t, dtype=np.float64)
            for unit_id in self.unit_ids
        )
        self.colors_u32 = make_unit_colors(len(self.unit_ids), cmap_name=cmap_name)
        self.unit_count = len(self.unit_ids)
        self.data_label = (
            data_path.name if data_path is not None and data_path.exists() else "in-memory TsGroup"
        )

        t_min_s, t_max_s = time_bounds(self.spike_times_s)
        t_min_ns = seconds_to_ns(t_min_s)
        t_max_ns = seconds_to_ns(t_max_s)
        full_span_ns = max(1, t_max_ns - t_min_ns)

        self.max_view_span_ns = full_span_ns
        self.initial_view_span_ns = min(seconds_to_ns(DEFAULT_VIEW_SPAN_S), full_span_ns)
        self.reset_cursor_ns = t_min_ns + self.initial_view_span_ns // 2

        self.controller = PlaybackController(
            t_min_ns,
            t_max_ns,
            initial_time_ns=self.reset_cursor_ns,
            view_span_ns=self.initial_view_span_ns,
            playback_speed=1.0,
        )

    def close(self) -> None:
        return


def draw_timestamp_array(
    times_s: np.ndarray,
    *,
    y_offset: float,
    x_min_s: float,
    x_max_s: float,
    color_u32: int,
    thickness: float = RASTER_TICK_THICKNESS,
    height_px: float = RASTER_TICK_HEIGHT_PX,
    time_offset_s: float = 0.0,
) -> None:
    if times_s.size == 0:
        return

    visible_start = x_min_s - time_offset_s
    visible_stop = x_max_s - time_offset_s
    start = int(np.searchsorted(times_s, visible_start, side="left"))
    stop = int(np.searchsorted(times_s, visible_stop, side="right"))
    if start >= stop:
        return

    draw_list = implot.get_plot_draw_list()
    half_height_px = 0.5 * height_px
    for timestamp_s in times_s[start:stop]:
        point = implot.plot_to_pixels(implot.Point(float(timestamp_s + time_offset_s), y_offset))
        draw_list.add_line(
            imgui.ImVec2(point.x, point.y - half_height_px),
            imgui.ImVec2(point.x, point.y + half_height_px),
            color_u32,
            thickness,
        )


def draw_raster_plot(
    state: AppState,
    x_min_box,
    x_max_box,
    *,
    tick_thickness: float = RASTER_TICK_THICKNESS,
    tick_height_px: float = RASTER_TICK_HEIGHT_PX,
) -> None:
    max_span_s = max_view_span_s(state.max_view_span_ns)
    if not implot.begin_plot("Raster##plot", None):
        return

    implot.setup_axes("time (s)", "unit")
    implot.setup_axis_links(implot.ImAxis_.x1, x_min_box, x_max_box)
    implot.setup_axis_limits_constraints(
        implot.ImAxis_.x1,
        state.controller.t_min_s,
        state.controller.t_max_s,
    )
    implot.setup_axis_zoom_constraints(
        implot.ImAxis_.x1,
        min(MIN_VIEW_SPAN_S, max_span_s),
        max_span_s,
    )
    implot.setup_axis_limits(
        implot.ImAxis_.y1,
        -0.5,
        max(0.5, state.unit_count - 0.5),
        imgui.Cond_.always,
    )

    limits = implot.get_plot_limits()
    implot.push_plot_clip_rect()
    try:
        for row, (times_s, color_u32) in enumerate(zip(state.spike_times_s, state.colors_u32)):
            draw_timestamp_array(
                times_s,
                y_offset=float(row),
                x_min_s=float(limits.x.min),
                x_max_s=float(limits.x.max),
                color_u32=color_u32,
                thickness=tick_thickness,
                height_px=tick_height_px,
            )

        draw_cursor(state.controller.cursor_s)
    finally:
        implot.pop_plot_clip_rect()
        implot.end_plot()


def build_app(state: AppState):
    def app() -> None:
        controller = state.controller
        controller.tick(time.monotonic())
        imgui.style_colors_classic()

        io = imgui.get_io()
        window_margin = 12.0
        default_width = max(960.0, io.display_size.x - 2.0 * window_margin)
        default_height = max(540.0, io.display_size.y - 2.0 * window_margin)
        imgui.set_next_window_pos(
            imgui.ImVec2(window_margin, window_margin),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(default_width, default_height),
            imgui.Cond_.first_use_ever,
        )
        main_visible, _ = imgui.begin(
            "##",
            flags=imgui.WindowFlags_.no_collapse,
        )
        if not main_visible:
            imgui.end()
            return

        handle_shortcuts(state, enable_debug_popup=False)

        if imgui.begin_child("plots", imgui.ImVec2(0, -TRANSPORT_HEIGHT)):
            plot_width = max(1, int(imgui.get_content_region_avail().x))
            viewport = controller.viewport(width_px=plot_width)
            x_min_box = implot.BoxedValue(ns_to_seconds(viewport.time.start_ns))
            x_max_box = implot.BoxedValue(ns_to_seconds(viewport.time.stop_ns))

            draw_raster_plot(state, x_min_box, x_max_box)
            sync_axis_links(state, x_min_box, x_max_box)
        imgui.end_child()

        imgui.separator()
        imgui.text_disabled(state.data_label)
        draw_transport(state, show_debug_button=False)
        imgui.end()

    return app


def main(data_path: Path = DEFAULT_DATA_PATH) -> None:
    state = AppState(data_path)
    try:
        immapp.run(
            build_app(state),
            window_size=(1480, 760),
            with_implot=True,
            fps_idle=0.0,
        )
    finally:
        state.close()


if __name__ == "__main__":
    main()