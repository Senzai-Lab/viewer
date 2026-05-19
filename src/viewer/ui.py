from __future__ import annotations
from imgui_bundle import imgui, implot, icons_fontawesome_4

from .utils import format_bytes, format_seconds

TIME_AXIS_CLOCK = "clock"
TIME_AXIS_SECONDS = "seconds"
TIME_AXIS_OPTIONS = [
    (TIME_AXIS_CLOCK, "Clock"),
    (TIME_AXIS_SECONDS, "Seconds"),
]

FA_PLAY = icons_fontawesome_4.ICON_FA_PLAY
FA_PAUSE = icons_fontawesome_4.ICON_FA_PAUSE
FA_BACKWARD = icons_fontawesome_4.ICON_FA_STEP_BACKWARD
FA_FORWARD = icons_fontawesome_4.ICON_FA_STEP_FORWARD
FA_RESET = icons_fontawesome_4.ICON_FA_UNDO
FA_BUG = icons_fontawesome_4.ICON_FA_BUG


def fmt_indices(indices: list[int]) -> str:
    if not indices:
        return "-"
    return ", ".join(str(i) for i in indices)


def _draw_chunk_boxes(name: str, current: int, cached: list[int], pending: list[int]):
    cached = set(cached)
    pending = set(pending)
    indices = [current - 1, current, current + 1]

    box = 10.0
    gap = 4.0
    width = len(indices) * box + (len(indices) - 1) * gap
    size = imgui.ImVec2(width, box)
    p0 = imgui.get_cursor_screen_pos()
    imgui.invisible_button(f"##chunk_boxes_{name}", size)

    draw = imgui.get_window_draw_list()
    for slot, idx in enumerate(indices):
        x = p0.x + slot * (box + gap)
        a = imgui.ImVec2(x, p0.y)
        b = imgui.ImVec2(x + box, p0.y + box)

        if idx in cached:
            color = imgui.IM_COL32(76, 175, 80, 255)
        elif idx in pending:
            color = imgui.IM_COL32(235, 190, 80, 255)
        else:
            color = imgui.IM_COL32(85, 90, 98, 255)

        draw.add_rect_filled(a, b, color, 2.0)
        border = imgui.IM_COL32(230, 235, 245, 220) if idx == current else imgui.IM_COL32(20, 22, 26, 180)
        draw.add_rect(a, b, border, 2.0, thickness=1.0)


def setup_style():
    imgui.style_colors_dark()

    style = imgui.get_style()
    style.window_rounding = 4.0
    style.child_rounding = 4.0
    style.frame_rounding = 3.0
    style.popup_rounding = 4.0
    style.grab_rounding = 3.0
    style.window_padding = imgui.ImVec2(10, 8)
    style.frame_padding = imgui.ImVec2(7, 4)
    style.item_spacing = imgui.ImVec2(8, 6)
    style.scrollbar_size = 10.0

    style.set_color_(imgui.Col_.window_bg, imgui.ImVec4(0.075, 0.080, 0.090, 1.0))
    style.set_color_(imgui.Col_.child_bg, imgui.ImVec4(0.060, 0.064, 0.072, 1.0))
    style.set_color_(imgui.Col_.frame_bg, imgui.ImVec4(0.120, 0.130, 0.145, 1.0))
    style.set_color_(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.170, 0.185, 0.205, 1.0))
    style.set_color_(imgui.Col_.button, imgui.ImVec4(0.145, 0.160, 0.180, 1.0))
    style.set_color_(imgui.Col_.button_hovered, imgui.ImVec4(0.220, 0.250, 0.285, 1.0))
    style.set_color_(imgui.Col_.check_mark, imgui.ImVec4(0.400, 0.760, 0.950, 1.0))
    style.set_color_(imgui.Col_.slider_grab, imgui.ImVec4(0.400, 0.760, 0.950, 1.0))

    plot_style = implot.get_style()
    plot_style.use24_hour_clock = True
    plot_style.use_iso8601 = False
    plot_style.use_local_time = False


def setup_time_axis(
    y_label: str,
    *,
    time_axis: str = TIME_AXIS_CLOCK,
    x_label: str | None = None,
    x_flags=0,
    y_flags=0,
) -> None:
    if x_label is None:
        x_label = {
            TIME_AXIS_CLOCK: "Time",
            TIME_AXIS_SECONDS: "Time (s)",
        }[time_axis]

    implot.setup_axes(x_label, y_label, x_flags, y_flags)
    scale = {
        TIME_AXIS_CLOCK: implot.Scale_.time,
        TIME_AXIS_SECONDS: implot.Scale_.linear,
    }[time_axis]
    implot.setup_axis_scale(implot.ImAxis_.x1, scale)


def format_time_value(seconds: float, time_axis: str) -> str:
    if time_axis == TIME_AXIS_CLOCK:
        return _format_elapsed_clock(seconds)
    if time_axis == TIME_AXIS_SECONDS:
        return f"{seconds:.1f} s"
    raise ValueError(f"Unknown time axis: {time_axis}")


def _format_elapsed_clock(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    total = int(round(abs(seconds)))
    days, rem = divmod(total, 24 * 60 * 60)
    hours, rem = divmod(rem, 60 * 60)
    minutes, seconds = divmod(rem, 60)
    text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if days:
        text = f"{days}d {text}"
    return f"{sign}{text}"


def draw_cursor(
    cursor_s: float,
    *,
    half_width_px: float = 6.0,
    height_px: float = 9.0,
    color_u32: int | None = None,
) -> None:
    limits = implot.get_plot_limits()
    if not limits.x.contains(cursor_s):
        return

    top = implot.plot_to_pixels(implot.Point(cursor_s, limits.y.max))
    bottom = implot.plot_to_pixels(implot.Point(cursor_s, limits.y.min))
    draw_list = implot.get_plot_draw_list()
    color = color_u32 if color_u32 is not None else imgui.IM_COL32(220, 230, 255, 230)

    draw_list.add_line(
        imgui.ImVec2(top.x, top.y + height_px),
        imgui.ImVec2(bottom.x, bottom.y),
        color,
        1.0,
    )
    draw_list.add_triangle_filled(
        imgui.ImVec2(top.x - half_width_px, top.y),
        imgui.ImVec2(top.x + half_width_px, top.y),
        imgui.ImVec2(top.x, top.y + height_px),
        color,
    )


def draw_stream_debug(cache, stream, t: float):
    current = stream.chunk_at(t)
    cached = cache.cached_indices(stream)
    pending = cache.pending_indices(stream)
    wanted = cache.wanted_indices(stream)

    imgui.text_disabled(
        f"chunk {current + 1}/{stream.n_chunks} | "
        f"{format_bytes(cache.cached_nbytes(stream))} cached"
    )
    _draw_chunk_boxes(stream.name, current, cached, pending)
    imgui.text_disabled(f"Time range: {stream.t_min:.3f} - {stream.t_max:.3f} s")
    imgui.text_disabled(f"Duration: {format_seconds(stream.duration)}")
    imgui.text_disabled(f"Chunk size: {format_bytes(stream.chunk_nbytes)}")

    if imgui.tree_node(f"Debug##debug_{stream.name}"):
        imgui.text(f"Wanted chunks: {fmt_indices(wanted)}")
        imgui.text(f"Loaded chunks: {fmt_indices(cached)}")
        imgui.text(f"Pending chunks: {fmt_indices(pending)}")
        imgui.tree_pop()


def gui_transport(state):
    """Dockable transport / scrubbing window."""
    ctrl = state.controller

    # --- transport buttons ---
    if imgui.button(FA_BACKWARD):
        ctrl.jump_by(-0.25)
    imgui.same_line()

    if imgui.button(FA_PAUSE if ctrl.is_playing else FA_PLAY):
        ctrl.toggle()
    imgui.same_line()

    if imgui.button(FA_FORWARD):
        ctrl.jump_by(0.25)
    imgui.same_line()

    if imgui.button(FA_RESET):
        state.reset()
    imgui.same_line()

    # --- speed slider ---
    imgui.set_next_item_width(70.0)
    changed, new_speed = imgui.drag_float(
        "##speed", ctrl.playback_speed, 0.05, -5.0, 10.0, "%.1fx"
    )
    if imgui.is_item_hovered(imgui.HoveredFlags_.stationary):
        imgui.set_tooltip("Playback speed")
    if changed:
        ctrl.playback_speed = new_speed
    imgui.same_line()

    modes = [mode for mode, _ in TIME_AXIS_OPTIONS]
    labels = [label for _, label in TIME_AXIS_OPTIONS]
    current_idx = modes.index(state.time_axis)
    imgui.set_next_item_width(95.0)
    changed, new_idx = imgui.combo("##time_axis", current_idx, labels)
    if imgui.is_item_hovered(imgui.HoveredFlags_.stationary):
        imgui.set_tooltip("Time axis")
    if changed:
        state.time_axis = modes[new_idx]
    imgui.same_line()

    # --- debug popup ---
    if imgui.button(FA_BUG):
        imgui.open_popup("debug_popup")
    if imgui.begin_popup("debug_popup"):
        cache = state.cache
        imgui.text(f"Cache: {format_bytes(cache.nbytes)} / {format_bytes(cache.max_nbytes)}")
        imgui.progress_bar(cache.nbytes / max(cache.max_nbytes, 1), imgui.ImVec2(200, 0))
        imgui.separator()
        imgui.text(f"Pending: {len(cache.pending)}")
        imgui.text(f"Cached chunks: {len(cache.cache)}")
        imgui.separator()
        imgui.end_popup()

    # --- global scrub bar ---
    imgui.set_next_item_width(-1)
    value_label = (
        f"{format_time_value(ctrl.t_cursor, state.time_axis)}  /  "
        f"{format_time_value(ctrl.t_max, state.time_axis)}"
    )
    changed, new_t = imgui.slider_float(
        "##time", ctrl.t_cursor, ctrl.t_min, ctrl.t_max,
        value_label,
    )
    if changed:
        ctrl.jump_to(new_t)

    # --- shortcuts ---
    if imgui.is_key_pressed(imgui.Key.space, repeat=False):
        ctrl.toggle()
    if imgui.is_key_pressed(imgui.Key.left_arrow):
        ctrl.jump_by(-0.25)
    if imgui.is_key_pressed(imgui.Key.right_arrow):
        ctrl.jump_by(0.25)
    if imgui.is_key_pressed(imgui.Key.r, repeat=False):
        state.reset()
