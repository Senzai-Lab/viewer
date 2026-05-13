from __future__ import annotations
from imgui_bundle import imgui, implot, icons_fontawesome_4

from .utils import format_bytes

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
    style.scrollbar_size = 12.0

    style.set_color_(imgui.Col_.window_bg, imgui.ImVec4(0.075, 0.080, 0.090, 1.0))
    style.set_color_(imgui.Col_.child_bg, imgui.ImVec4(0.060, 0.064, 0.072, 1.0))
    style.set_color_(imgui.Col_.frame_bg, imgui.ImVec4(0.120, 0.130, 0.145, 1.0))
    style.set_color_(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.170, 0.185, 0.205, 1.0))
    style.set_color_(imgui.Col_.button, imgui.ImVec4(0.145, 0.160, 0.180, 1.0))
    style.set_color_(imgui.Col_.button_hovered, imgui.ImVec4(0.220, 0.250, 0.285, 1.0))
    style.set_color_(imgui.Col_.check_mark, imgui.ImVec4(0.400, 0.760, 0.950, 1.0))
    style.set_color_(imgui.Col_.slider_grab, imgui.ImVec4(0.400, 0.760, 0.950, 1.0))


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
    cached = cache.cached_indices(stream.name)
    pending = cache.pending_indices(stream.name)
    desired = cache.desired_indices(stream.name)

    imgui.text_disabled(
        f"{stream.kind} | chunk {current + 1}/{stream.n_chunks} | "
        f"{format_bytes(cache.cached_bytes(stream.name))} cached"
    )

    if imgui.tree_node(f"Debug##debug_{stream.name}"):
        imgui.text(f"Time range: {stream.t_min:.3f} - {stream.t_max:.3f} s")
        imgui.text(f"Duration: {stream.t_max - stream.t_min:.3f} s")
        imgui.text(f"Chunk size: {format_bytes(stream.chunk_nbytes)}")
        imgui.text(f"Current chunk: {current}")
        imgui.text(f"Desired chunks: {fmt_indices(desired)}")
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

    # --- speed input ---
    imgui.set_next_item_width(35.0)
    changed, new_speed = imgui.input_float("##speed", ctrl.playback_speed, format="%.1f")
    if imgui.is_item_hovered(imgui.HoveredFlags_.stationary):
        imgui.set_tooltip("Playback speed")
    if changed:
        ctrl.playback_speed = new_speed
    imgui.same_line()

    # --- debug popup ---
    if imgui.button(FA_BUG):
        imgui.open_popup("debug_popup")
    if imgui.begin_popup("debug_popup"):
        cache = state.cache
        imgui.text(f"Cache: {format_bytes(cache.used_bytes)} / {format_bytes(cache.max_budget)}")
        imgui.progress_bar(cache.used_bytes / max(cache.max_budget, 1), imgui.ImVec2(200, 0))
        imgui.separator()
        imgui.text(f"Pending: {len(cache.pending)}")
        imgui.text(f"Cached chunks: {len(cache.cache)}")
        imgui.separator()
        imgui.end_popup()

    # --- global scrub bar ---
    imgui.set_next_item_width(-1)
    changed, new_t = imgui.slider_float(
        "##time", ctrl.t_cursor, ctrl.t_min, ctrl.t_max,
        f"%.1f s  /  {ctrl.t_max:.1f} s",
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
