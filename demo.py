import math
import time
from collections import deque

import numpy as np
from imgui_bundle import imgui, immapp, implot, icons_fontawesome_4

from viewer import (
    ChunkManager,
    DenseSignalAdapter,
    PlaybackController,
    Selection,
    StreamView,
    ns_to_seconds,
    seconds_to_ns,
)
from viewer.utils import format_bytes


FA_PLAY = icons_fontawesome_4.ICON_FA_PLAY
FA_PAUSE = icons_fontawesome_4.ICON_FA_PAUSE
FA_BACKWARD = icons_fontawesome_4.ICON_FA_STEP_BACKWARD
FA_FORWARD = icons_fontawesome_4.ICON_FA_STEP_FORWARD
FA_RESET = icons_fontawesome_4.ICON_FA_UNDO
FA_BUG = icons_fontawesome_4.ICON_FA_BUG

JUMP_VIEW_FRACTION = 0.25
MIN_VIEW_SPAN_S = 0.25
PLOT_HEIGHT = 240
TRANSPORT_HEIGHT = 96.0
DEFAULT_Y_LIMITS = (-1.5, 1.5)
CURSOR_HALF_WIDTH_PX = 6.0
CURSOR_HEIGHT_PX = 9.0
FPS_HISTORY_LEN = 120


# ---------------------------------------------------------------------------
# Cache budget estimation
# ---------------------------------------------------------------------------

def _chunk_nbytes(adapter: DenseSignalAdapter) -> int:
    per_sample = adapter.channel_count * adapter.data.dtype.itemsize + np.dtype(np.int64).itemsize
    chunk_samples = min(adapter.chunk_samples, adapter.sample_count)
    return int(chunk_samples * per_sample)


def _chunk_span_ns(adapter: DenseSignalAdapter) -> int:
    chunk_samples = min(adapter.chunk_samples, adapter.sample_count)
    return max(1, int(round(chunk_samples * adapter.sample_period_ns)))


def _required_bytes(adapters, span_ns: int, prefetch_distance: int) -> int:
    total = 0
    for adapter in adapters:
        chunk_span = _chunk_span_ns(adapter)
        visible = max(1, math.ceil(span_ns / chunk_span) + 1)
        total += (visible + max(0, prefetch_distance)) * _chunk_nbytes(adapter)
    return total


def _max_view_span_ns(adapters, budget: int | None, prefetch_distance: int) -> int:
    full = max(a.time_range.stop_ns for a in adapters) - min(
        a.time_range.start_ns for a in adapters
    )
    if budget is None or _required_bytes(adapters, full, prefetch_distance) <= budget:
        return full
    lo, hi = 1, full
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _required_bytes(adapters, mid, prefetch_distance) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def apply_theme_once(state: "AppState") -> None:
    if state.theme_applied:
        return

    style = imgui.get_style()
    imgui.style_colors_dark()

    style.window_rounding = 6.0
    style.frame_rounding = 4.0
    style.grab_rounding = 4.0
    style.popup_rounding = 6.0
    style.scrollbar_rounding = 6.0
    style.tab_rounding = 4.0
    style.window_padding = imgui.ImVec2(10, 10)
    style.frame_padding = imgui.ImVec2(8, 5)
    style.item_spacing = imgui.ImVec2(8, 6)

    c = imgui.Col_
    set_color = style.set_color_
    set_color(c.window_bg.value, imgui.ImVec4(0.09, 0.09, 0.11, 1.00))
    set_color(c.child_bg.value, imgui.ImVec4(0.09, 0.09, 0.11, 0.00))
    set_color(c.popup_bg.value, imgui.ImVec4(0.11, 0.11, 0.13, 0.98))
    set_color(c.frame_bg.value, imgui.ImVec4(0.16, 0.16, 0.19, 1.00))
    set_color(c.frame_bg_hovered.value, imgui.ImVec4(0.22, 0.22, 0.26, 1.00))
    set_color(c.frame_bg_active.value, imgui.ImVec4(0.28, 0.28, 0.33, 1.00))
    set_color(c.button.value, imgui.ImVec4(0.20, 0.20, 0.24, 1.00))
    set_color(c.button_hovered.value, imgui.ImVec4(0.30, 0.42, 0.62, 1.00))
    set_color(c.button_active.value, imgui.ImVec4(0.24, 0.36, 0.56, 1.00))
    set_color(c.header.value, imgui.ImVec4(0.20, 0.20, 0.24, 1.00))
    set_color(c.header_hovered.value, imgui.ImVec4(0.30, 0.42, 0.62, 1.00))
    set_color(c.slider_grab.value, imgui.ImVec4(0.45, 0.65, 0.95, 1.00))
    set_color(c.slider_grab_active.value, imgui.ImVec4(0.55, 0.75, 1.00, 1.00))
    set_color(c.check_mark.value, imgui.ImVec4(0.55, 0.75, 1.00, 1.00))
    set_color(c.separator.value, imgui.ImVec4(0.20, 0.20, 0.24, 1.00))
    set_color(c.border.value, imgui.ImVec4(0.20, 0.20, 0.24, 1.00))

    state.theme_applied = True


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
        self.max_view_span_ns = _max_view_span_ns(self.adapters, memory_budget, prefetch_distance)
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

        self.show_cursor = True
        self.show_debug = False
        self.theme_applied = False
        self.y_limits = {sv.stream_id: DEFAULT_Y_LIMITS for sv in self.stream_views}
        self.default_y_limits = dict(self.y_limits)

        self.frames: dict = {}
        self.stream_cache_stats = self.session.stream_debug_stats()
        self.fps_history: deque[float] = deque(maxlen=FPS_HISTORY_LEN)

    def close(self) -> None:
        self.session.close()


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

def jump_step_ns(controller: PlaybackController) -> int:
    return max(1, int(round(controller.view_span_ns * JUMP_VIEW_FRACTION)))


def max_view_span_s(state: AppState) -> float:
    return ns_to_seconds(max(1, state.max_view_span_ns))


def min_view_span_s(state: AppState) -> float:
    return min(MIN_VIEW_SPAN_S, max_view_span_s(state))


def estimated_working_bytes(state: AppState) -> int:
    return _required_bytes(
        state.adapters,
        state.controller.view_span_ns,
        state.session.prefetch_distance,
    )


def reset_state(state: AppState) -> None:
    state.session.clear()
    state.controller.pause()
    state.controller.set_view_span(state.initial_view_span_ns)
    state.controller.jump_to(state.reset_cursor_ns)
    state.show_cursor = True
    state.y_limits = dict(state.default_y_limits)
    state.stream_cache_stats = state.session.stream_debug_stats()


def set_visible_window(state: AppState, start_s: float, stop_s: float) -> None:
    controller = state.controller
    t_min, t_max = controller.t_min_s, controller.t_max_s
    max_span = min(t_max - t_min, max_view_span_s(state))
    min_span = min(MIN_VIEW_SPAN_S, max_span)

    if stop_s < start_s:
        start_s, stop_s = stop_s, start_s
    span = min(max_span, max(min_span, stop_s - start_s))
    if span >= max_span:
        start_s, stop_s = t_min, t_max
    else:
        if start_s < t_min:
            start_s, stop_s = t_min, t_min + span
        if stop_s > t_max:
            stop_s, start_s = t_max, t_max - span

    controller.set_view_span(seconds_to_ns(stop_s - start_s))
    controller.jump_to(seconds_to_ns(0.5 * (start_s + stop_s)))


def sync_axis_links(state: AppState, x_min_box, x_max_box) -> None:
    visible = state.controller.visible_range
    cur_start = ns_to_seconds(visible.start_ns)
    cur_stop = ns_to_seconds(visible.stop_ns)
    new_start = float(x_min_box.value)
    new_stop = float(x_max_box.value)
    if abs(new_start - cur_start) <= 1e-6 and abs(new_stop - cur_stop) <= 1e-6:
        return
    set_visible_window(state, new_start, new_stop)


def handle_shortcuts(state: AppState) -> None:
    if not imgui.is_window_focused(imgui.FocusedFlags_.root_and_child_windows):
        return
    if imgui.is_any_item_active():
        return

    controller = state.controller
    if imgui.is_key_pressed(imgui.Key.space, repeat=False):
        controller.toggle()
    if imgui.is_key_pressed(imgui.Key.left_arrow):
        controller.jump_by(-jump_step_ns(controller))
    if imgui.is_key_pressed(imgui.Key.right_arrow):
        controller.jump_by(jump_step_ns(controller))
    if imgui.is_key_pressed(imgui.Key.r, repeat=False):
        reset_state(state)
    if imgui.is_key_pressed(imgui.Key.d, repeat=False):
        state.show_debug = not state.show_debug


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------

def draw_cursor(cursor_s: float) -> None:
    limits = implot.get_plot_limits()
    if not limits.x.contains(cursor_s):
        return

    top = implot.plot_to_pixels(implot.Point(cursor_s, limits.y.max))
    bottom = implot.plot_to_pixels(implot.Point(cursor_s, limits.y.min))
    draw_list = implot.get_plot_draw_list()
    color = imgui.IM_COL32(220, 230, 255, 230)

    draw_list.add_line(
        imgui.ImVec2(top.x, top.y + CURSOR_HEIGHT_PX),
        imgui.ImVec2(bottom.x, bottom.y),
        color,
        1.0,
    )
    draw_list.add_triangle_filled(
        imgui.ImVec2(top.x - CURSOR_HALF_WIDTH_PX, top.y),
        imgui.ImVec2(top.x + CURSOR_HALF_WIDTH_PX, top.y),
        imgui.ImVec2(top.x, top.y + CURSOR_HEIGHT_PX),
        color,
    )


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
    plot_flags = implot.Flags_.no_menus | implot.Flags_.no_mouse_text

    if implot.begin_plot(f"{stream_id}##plot", imgui.ImVec2(-1, PLOT_HEIGHT), plot_flags):
        implot.setup_axes("time (s)", "value")
        implot.setup_axis_links(implot.ImAxis_.x1, x_min_box, x_max_box)
        implot.setup_axis_limits_constraints(
            implot.ImAxis_.x1,
            state.controller.t_min_s,
            state.controller.t_max_s,
        )
        implot.setup_axis_zoom_constraints(
            implot.ImAxis_.x1,
            min_view_span_s(state),
            max_view_span_s(state),
        )
        implot.setup_axis_links(implot.ImAxis_.y1, y_min_box, y_max_box)

        if payload is not None and payload["time_ns"].size:
            xs = np.ascontiguousarray(payload["time_ns"], dtype=np.float64) / 1_000_000_000.0
            values = np.ascontiguousarray(payload["values"], dtype=np.float64)
            channel_indices = np.ascontiguousarray(payload["channel_indices"], dtype=np.int64)
            for i, channel_values in enumerate(values):
                ys = np.ascontiguousarray(channel_values, dtype=np.float64)
                implot.plot_line(f"ch{int(channel_indices[i])}", xs, ys)
        else:
            implot.plot_text("loading…", ns_to_seconds(request.cursor_ns), 0.0)

        if state.show_cursor:
            draw_cursor(state.controller.cursor_s)

        state.y_limits[stream_id] = (float(y_min_box.value), float(y_max_box.value))
        implot.end_plot()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def draw_transport(state: AppState) -> None:
    controller = state.controller
    step = jump_step_ns(controller)

    if imgui.button(FA_BACKWARD):
        controller.jump_by(-step)
    imgui.same_line()

    if imgui.button(FA_PAUSE if controller.is_playing else FA_PLAY):
        controller.toggle()
    imgui.same_line()

    if imgui.button(FA_FORWARD):
        controller.jump_by(step)
    imgui.same_line()

    imgui.dummy(imgui.ImVec2(8, 0))
    imgui.same_line()

    if imgui.button(f"{FA_RESET}"):
        reset_state(state)
    imgui.same_line()

    _, state.show_debug = imgui.checkbox(f"{FA_BUG}", state.show_debug)
    imgui.same_line()

    _, state.show_cursor = imgui.checkbox("Cursor", state.show_cursor)
    imgui.same_line()

    imgui.set_next_item_width(110)
    changed, new_speed = imgui.input_float("speed", controller.playback_speed, 0.1, 1.0, "%.2fx")
    if changed:
        controller.playback_speed = float(new_speed)

    imgui.set_next_item_width(-1)
    changed, new_t = imgui.slider_float(
        "##time",
        controller.cursor_s,
        controller.t_min_s,
        controller.t_max_s,
        f"%.3f s  /  {controller.t_max_s:.2f} s",
    )
    if changed:
        controller.jump_to(seconds_to_ns(new_t))


# ---------------------------------------------------------------------------
# Debug window (cache-focused, toggleable, movable, resizable)
# ---------------------------------------------------------------------------

def _kv(label: str, value: str) -> None:
    imgui.text_disabled(label)
    imgui.same_line(140)
    imgui.text(value)


def _progress(fraction: float, overlay: str) -> None:
    imgui.progress_bar(max(0.0, min(1.0, fraction)), imgui.ImVec2(-1, 0), overlay)


def draw_debug_window(state: AppState) -> None:
    if not state.show_debug:
        return

    imgui.set_next_window_size(imgui.ImVec2(420, 540), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos(imgui.ImVec2(60, 60), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(imgui.ImVec2(340, 240), imgui.ImVec2(800, 1200))

    expanded, state.show_debug = imgui.begin(f"{FA_BUG}  Cache Debug", state.show_debug)
    if not expanded:
        imgui.end()
        return

    session = state.session
    avg_fps = (sum(state.fps_history) / len(state.fps_history)) if state.fps_history else 0.0
    frame_ms = (1000.0 / avg_fps) if avg_fps > 0 else 0.0

    imgui.separator_text("Performance")
    _kv("FPS", f"{avg_fps:6.1f}")
    _kv("Frame time", f"{frame_ms:6.2f} ms")

    imgui.separator_text("Memory")
    budget = session.memory_budget_bytes
    loaded = session.loaded_bytes
    working = estimated_working_bytes(state)
    _progress(
        (loaded / budget) if budget else 0.0,
        f"loaded   {format_bytes(loaded)} / {format_bytes(budget)}",
    )
    _progress(
        (working / budget) if budget else 0.0,
        f"working ≈ {format_bytes(working)}",
    )

    imgui.separator_text("Chunks")
    _kv("Loaded",   str(session.loaded_chunk_count))
    _kv("Pending",  str(session.pending_chunk_count))
    _kv("Working",  str(session.working_chunk_count))
    _kv("Prefetch", f"±{session.prefetch_distance}")

    imgui.separator_text("View")
    _kv("Span",   f"{state.controller.view_span_s:.3f} s")
    _kv("Max",    f"{max_view_span_s(state):.3f} s")
    _kv("Cursor", f"{state.controller.cursor_s:.3f} s")
    _kv("State",  "playing" if state.controller.is_playing else "paused")

    imgui.separator_text("Per-stream")
    for adapter in state.adapters:
        stats = state.stream_cache_stats[adapter.stream_id]
        frame = state.frames.get(adapter.stream_id)
        coverage = frame.coverage if frame is not None else 0.0

        if not imgui.collapsing_header(
            f"{adapter.stream_id}##hdr", imgui.TreeNodeFlags_.default_open.value
        ):
            continue

        imgui.indent()
        _progress(coverage, f"coverage {coverage * 100:5.1f}%")
        _kv("Loaded",  f"{stats.loaded_chunk_count}  ({format_bytes(stats.loaded_bytes)})")
        _kv("Pending", str(stats.pending_chunk_count))
        _kv("Working", str(stats.working_chunk_count))
        _kv("Chunk",   f"{format_bytes(_chunk_nbytes(adapter))} / "
                       f"{ns_to_seconds(_chunk_span_ns(adapter)):.3f} s")
        if stats.loaded_keys:
            imgui.text_disabled("Loaded keys:")
            imgui.text_wrapped(", ".join(stats.loaded_keys))
        imgui.unindent()

    imgui.end()


# ---------------------------------------------------------------------------
# App loop
# ---------------------------------------------------------------------------

def build_app(state: AppState):
    def app() -> None:
        controller = state.controller
        controller.tick(time.monotonic())
        apply_theme_once(state)

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
            "Signal Viewer",
            flags=imgui.WindowFlags_.no_collapse,
        )
        if not main_visible:
            imgui.end()
            draw_debug_window(state)
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
        imgui.end()

        draw_debug_window(state)

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
