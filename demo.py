import time
from collections import deque

import numpy as np
from imgui_bundle import imgui, immapp, implot

from viewer import (
    ChunkManager,
    DenseSignalAdapter,
    PlaybackController,
    Selection,
    StreamView,
    ns_to_seconds,
    seconds_to_ns,
)
from viewer.ui_old import (
    TRANSPORT_HEIGHT,
    draw_cursor,
    draw_debug_popup,
    draw_transport,
    handle_shortcuts,
    sync_axis_links,
)
from viewer.utils import MIN_VIEW_SPAN_S, max_view_span_ns, max_view_span_s


DEFAULT_Y_LIMITS = (-1.5, 1.5)
FPS_HISTORY_LEN = 120
HOVER_MARKER_RADIUS_PX = 4.0
HOVER_Y_TOLERANCE_FRAC = 0.04


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        sample_rate_hz = 1000.0
        duration_s = 120.0
        times = np.arange(0.0, duration_s, 1.0 / sample_rate_hz, dtype=np.float64)

        stream_a = np.vstack([
            np.sin(2 * np.pi * 1.2 * times),
            0.5 * np.cos(2 * np.pi * 8.0 * times),
        ]).astype(np.float32)
        stream_b = np.vstack([
            0.75 * np.sin(2 * np.pi * 0.35 * times + 0.4),
            0.25 * np.cos(2 * np.pi * 14.0 * times),
        ]).astype(np.float32)

        self.adapters = [
            DenseSignalAdapter("signal-a", stream_a, sample_rate_hz, target_chunk_bytes=512 * 1024),
            DenseSignalAdapter("signal-b", stream_b, sample_rate_hz, target_chunk_bytes=512 * 1024),
        ]
        self.stream_views = (
            StreamView("signal-a", Selection(channels=(0, 1))),
            StreamView("signal-b", Selection(channels=(0,))),
        )

        memory_budget = 3 * 1024 * 1024
        prefetch_distance = 1
        self.max_view_span_ns = max_view_span_ns(self.adapters, memory_budget, prefetch_distance)
        self.initial_view_span_ns = min(seconds_to_ns(8.0), self.max_view_span_ns)
        self.initial_cursor_ns = seconds_to_ns(5.0)
        self.reset_cursor_ns = min(a.time_range.start_ns for a in self.adapters)

        self.controller = PlaybackController(
            min(a.time_range.start_ns for a in self.adapters),
            max(a.time_range.stop_ns for a in self.adapters),
            initial_time_ns=self.initial_cursor_ns,
            view_span_ns=self.initial_view_span_ns,
            playback_speed=1.0,
        )
        self.session = ChunkManager(
            self.adapters,
            max_workers=2,
            prefetch_distance=prefetch_distance,
            memory_budget_bytes=memory_budget,
        )

        self.y_limits = {sv.stream_id: DEFAULT_Y_LIMITS for sv in self.stream_views}
        self.default_y_limits = dict(self.y_limits)

        self.frames: dict = {}
        self.stream_cache_stats = self.session.stream_debug_stats()
        self.fps_history: deque[float] = deque(maxlen=FPS_HISTORY_LEN)

    def close(self) -> None:
        self.session.close()


def _nearest_sample_index(xs: np.ndarray, x_value: float) -> int:
    index = int(np.searchsorted(xs, x_value))
    if index <= 0:
        return 0
    if index >= xs.size:
        return int(xs.size - 1)
    before = index - 1
    if abs(float(xs[index]) - x_value) < abs(float(xs[before]) - x_value):
        return index
    return before


def _hovered_line_info(
    xs: np.ndarray,
    values: np.ndarray,
    channel_indices: np.ndarray,
    y_limits: tuple[float, float],
) -> dict | None:
    if xs.size == 0 or values.size == 0 or not implot.is_plot_hovered():
        return None

    mouse = implot.get_plot_mouse_pos()
    sample_index = _nearest_sample_index(xs, float(mouse.x))
    span = max(1e-6, y_limits[1] - y_limits[0])
    y_tolerance = span * HOVER_Y_TOLERANCE_FRAC

    hovered = None
    best_distance = None
    for row, ys in enumerate(values):
        value = float(ys[sample_index])
        distance = abs(value - float(mouse.y))
        if distance > y_tolerance:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            hovered = {
                "row": row,
                "sample_index": sample_index,
                "channel": int(channel_indices[row]),
                "x": float(xs[sample_index]),
                "y": value,
            }
    return hovered


def _boost_color(color: imgui.ImVec4, amount: float = 0.18) -> imgui.ImVec4:
    return imgui.ImVec4(
        min(1.0, color.x + amount),
        min(1.0, color.y + amount),
        min(1.0, color.z + amount),
        color.w,
    )


def _draw_hover_guide(hovered: dict, base_color: imgui.ImVec4) -> None:
    """Vertical guide + dot at the hovered sample. Cheap: a few draw_list ops."""
    limits = implot.get_plot_limits()
    color = imgui.color_convert_float4_to_u32(_boost_color(base_color))
    guide_color = imgui.color_convert_float4_to_u32(
        imgui.ImVec4(base_color.x, base_color.y, base_color.z, 0.35)
    )
    draw_list = implot.get_plot_draw_list()
    top = implot.plot_to_pixels(implot.Point(hovered["x"], limits.y.max))
    bottom = implot.plot_to_pixels(implot.Point(hovered["x"], limits.y.min))
    draw_list.add_line(
        imgui.ImVec2(top.x, top.y),
        imgui.ImVec2(bottom.x, bottom.y),
        guide_color,
        1.0,
    )
    point = implot.plot_to_pixels(implot.Point(hovered["x"], hovered["y"]))
    draw_list.add_circle_filled(
        imgui.ImVec2(point.x, point.y),
        HOVER_MARKER_RADIUS_PX,
        color,
    )


def _draw_hover_details(stream_id: str, hovered: dict, base_color: imgui.ImVec4) -> None:
    implot.annotation(
        hovered["x"],
        hovered["y"],
        _boost_color(base_color),
        imgui.ImVec2(10, -10),
        True,
        f"ch{hovered['channel']}  {hovered['y']:.3f}",
    )

    if imgui.begin_tooltip():
        imgui.text_disabled(stream_id)
        imgui.separator()
        imgui.text(f"channel: {hovered['channel']}")
        imgui.text(f"time:    {hovered['x']:.4f} s")
        imgui.text(f"value:   {hovered['y']:.4f}")
        imgui.text(f"sample:  {hovered['sample_index']}")
        imgui.end_tooltip()


def draw_stream_plot(
    state: AppState,
    stream_id: str,
    request,
    payload,
    x_min_box,
    x_max_box,
) -> None:
    y_min, y_max = state.y_limits[stream_id]
    y_min_box = implot.BoxedValue(y_min)
    y_max_box = implot.BoxedValue(y_max)
    max_span_s = max_view_span_s(state.max_view_span_ns)

    if implot.begin_plot(f"{stream_id}##plot", imgui.ImVec2(-1, -1)):
        implot.setup_axes("time (s)", "value")
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
        implot.setup_axis_links(implot.ImAxis_.y1, y_min_box, y_max_box)

        if payload is not None and payload["time_ns"].size:
            xs = payload["time_ns"].astype(np.float64) / 1_000_000_000.0
            values = np.ascontiguousarray(payload["values"], dtype=np.float64)
            channel_indices = np.ascontiguousarray(payload["channel_indices"], dtype=np.int64)
            hovered = _hovered_line_info(
                xs,
                values,
                channel_indices,
                (float(y_min_box.value), float(y_max_box.value)),
            )
            hovered_color = None
            for i in range(values.shape[0]):
                ys = values[i]
                label = f"ch{int(channel_indices[i])}"
                implot.plot_line(label, xs, ys)
                if hovered is not None and hovered["row"] == i:
                    hovered_color = implot.get_last_item_color()

            if hovered is not None and hovered_color is not None:
                _draw_hover_guide(hovered, hovered_color)
                _draw_hover_details(stream_id, hovered, hovered_color)
        else:
            implot.plot_text("loading…", ns_to_seconds(request.cursor_ns), 0.0)

        draw_cursor(state.controller.cursor_s)

        implot.end_plot()
        state.y_limits[stream_id] = (float(y_min_box.value), float(y_max_box.value))


# ---------------------------------------------------------------------------
# App loop
# ---------------------------------------------------------------------------

def build_app(state: AppState):
    def app() -> None:
        controller = state.controller
        controller.tick(time.monotonic())
        imgui.style_colors_classic()

        io = imgui.get_io()
        state.fps_history.append(io.framerate)

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
            draw_debug_popup(state)
            imgui.end()
            return

        handle_shortcuts(state)

        if imgui.begin_child("plots", imgui.ImVec2(0, -TRANSPORT_HEIGHT)):
            plot_width = max(1, int(imgui.get_content_region_avail().x))
            viewport = controller.viewport(width_px=plot_width, streams=state.stream_views)
            state.frames = state.session.update(viewport)
            state.stream_cache_stats = state.session.stream_debug_stats()

            x_min_box = implot.BoxedValue(ns_to_seconds(viewport.time.start_ns))
            x_max_box = implot.BoxedValue(ns_to_seconds(viewport.time.stop_ns))
            subplot_flags = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_menus

            if implot.begin_subplots(
                "##signals",
                len(state.stream_views),
                1,
                imgui.ImVec2(-1, -1),
                subplot_flags,
            ):
                for stream_view in state.stream_views:
                    frame = state.frames[stream_view.stream_id]
                    draw_stream_plot(
                        state,
                        stream_view.stream_id,
                        viewport,
                        frame.data,
                        x_min_box,
                        x_max_box,
                    )
                implot.end_subplots()

            sync_axis_links(state, x_min_box, x_max_box)
        imgui.end_child()

        imgui.separator()
        draw_transport(state)
        draw_debug_popup(state)
        imgui.end()

    return app


def main() -> None:
    state = AppState()
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
