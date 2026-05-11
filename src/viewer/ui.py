from __future__ import annotations
from imgui_bundle import imgui, implot, icons_fontawesome_4

from dataclasses import dataclass
from .utils import format_bytes

FA_PLAY = icons_fontawesome_4.ICON_FA_PLAY
FA_PAUSE = icons_fontawesome_4.ICON_FA_PAUSE
FA_BACKWARD = icons_fontawesome_4.ICON_FA_STEP_BACKWARD
FA_FORWARD = icons_fontawesome_4.ICON_FA_STEP_FORWARD
FA_RESET = icons_fontawesome_4.ICON_FA_UNDO
FA_BUG = icons_fontawesome_4.ICON_FA_BUG


@dataclass
class EphysSpec:
    line_weight: float = 1.0
    ch_offset: float = 1.0
    gain: float = 1.0
    visible: bool = True

    def to_implot_spec(self) -> implot.Spec:
        return implot.Spec(line_weight=self.line_weight)

def draw_ephys_spec(name: str, spec: EphysSpec):
    _, spec.visible = imgui.checkbox(f"Visible##{name}", spec.visible)

    imgui.set_next_item_width(-1)
    _, spec.line_weight = imgui.slider_float(
        f"##lw_{name}", spec.line_weight, 0.5, 3.0, "Line weight: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.gain = imgui.slider_float(
        f"##gain_{name}", spec.gain, 0.1, 20.0, "Gain: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.ch_offset = imgui.slider_float(
        f"##offset_{name}", spec.ch_offset, 0.0, 100.0, "Ch offset: %.0f"
    )

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
        imgui.text(f"Cache: {format_bytes(cache.used)} / {format_bytes(cache.budget)}")
        imgui.progress_bar(cache.used / max(cache.budget, 1), imgui.ImVec2(200, 0))
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

