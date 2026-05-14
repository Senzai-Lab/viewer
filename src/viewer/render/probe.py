from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

BG_U32 = 0xFF110F0E
OFF_U32 = 0xFF3C3836
EDGE_U32 = 0xE60D0B0A
HOVER_U32 = 0xFF6BE6FF
HOVER_BRIGHTEN = 0.45


class ProbeSettings:
    def __init__(
        self,
        geometry: dict,
        shank_gap: int = 4,
    ):
        self.channel_ids = np.asarray(geometry["channel_ids"])
        self.shank_ids = np.asarray(geometry["shank_ids"])
        self.x = np.asarray(geometry["x"])
        self.y = np.asarray(geometry["y"])
        self.n_channels = len(self.channel_ids)
        self.visible = np.zeros(self.n_channels, dtype=bool)
        self.hovered_idx = -1
        self.shanks = np.unique(self.shank_ids)
        self._shank_color_indices = {
            int(shank): i for i, shank in enumerate(self.shanks)
        }
        self.xg, self.yg = prb_to_grid(
            self.shank_ids,
            self.x,
            self.y,
            shank_gap,
        )

    def toggle_shank(self, shank: int):
        mask = self.shank_ids == shank
        self.visible[mask] = not self.visible[mask].all()

    def toggle_ch(self, ch: int):
        self.visible[ch] = ~self.visible[ch]

    def channel_label(self, ch: int) -> str:
        return f"ch {int(self.channel_ids[ch])}"

    def channel_color_vec(
        self,
        ch: int,
        *,
        alpha: float = 1.0,
        brighten: float = 0.0,
    ) -> imgui.ImVec4:
        shank = int(self.shank_ids[ch])
        color_idx = self._shank_color_indices[shank]
        return shank_color_vec(color_idx, alpha=alpha, brighten=brighten)

    def channel_color_u32(
        self,
        ch: int,
        *,
        alpha: float = 1.0,
        brighten: float = 0.0,
    ) -> int:
        return color_u32(self.channel_color_vec(ch, alpha=alpha, brighten=brighten))


def prb_to_grid(
    shank_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shank_gap: int = 5,
):
    xg = np.zeros(len(x), dtype=int)
    _, yg = np.unique(y, return_inverse=True)

    for shank in np.unique(shank_ids):
        mask = shank_ids == shank
        _, xg[mask] = np.unique(x[mask], return_inverse=True)
        xg[mask] += int(shank) * shank_gap

    # Screen y increases downward.
    yg = yg.max() - yg
    return xg, yg


def _hit_rect(mouse: imgui.ImVec2, p1: imgui.ImVec2, p2: imgui.ImVec2) -> bool:
    return p1.x <= mouse.x <= p2.x and p1.y <= mouse.y <= p2.y


def _contact_rects(
    settings: ProbeSettings,
    origin: imgui.ImVec2,
    cell: float,
    contact_px: float,
) -> list[tuple[imgui.ImVec2, imgui.ImVec2]]:
    rects = []
    for xg, yg in zip(settings.xg, settings.yg):
        x = origin.x + xg * cell
        y = origin.y + yg * cell
        rects.append(
            (imgui.ImVec2(x, y), imgui.ImVec2(x + contact_px, y + contact_px))
        )
    return rects


def _hit_contact(
    rects: list[tuple[imgui.ImVec2, imgui.ImVec2]],
    hovered: bool,
) -> int:
    if not hovered:
        return -1

    mouse = imgui.get_mouse_pos()
    for i, (p1, p2) in enumerate(rects):
        if _hit_rect(mouse, p1, p2):
            return i
    return -1


def shank_color_vec(
    color_idx: int,
    *,
    alpha: float = 1.0,
    brighten: float = 0.0,
) -> imgui.ImVec4:
    color = implot.get_colormap_color(color_idx)
    x, y, z, w = color.x, color.y, color.z, color.w
    if brighten > 0.0:
        amount = min(max(brighten, 0.0), 1.0)
        x += (1.0 - x) * amount
        y += (1.0 - y) * amount
        z += (1.0 - z) * amount
    return imgui.ImVec4(x, y, z, w * alpha)


def color_u32(color: imgui.ImVec4) -> int:
    return imgui.color_convert_float4_to_u32(color)


def _draw_contacts(
    draw: imgui.ImDrawList,
    settings: ProbeSettings,
    rects: list[tuple[imgui.ImVec2, imgui.ImVec2]],
):
    for i, (p1, p2) in enumerate(rects):
        color = (
            settings.channel_color_u32(i, alpha=0.95)
            if settings.visible[i]
            else OFF_U32
        )
        draw.add_rect_filled(p1, p2, color, 0.0)
        draw.add_rect(p1, p2, EDGE_U32, 0.0)


def draw_widget(
    name: str,
    settings: ProbeSettings,
    *,
    height: float = 480.0,
    contact_px: float = 12.0,
    gap_px: float = 1.0,
):
    settings.hovered_idx = -1
    avail = imgui.get_content_region_avail()

    pad = 10.0
    cell = contact_px + gap_px
    draw_w = (settings.xg.max() + 1) * cell
    draw_h = (settings.yg.max() + 1) * cell

    content_w = draw_w + 2 * pad
    canvas_w = max(1.0, min(avail.x, content_w))
    canvas_h = max(1.0, draw_h + 2 * pad)

    if not imgui.begin_child(
        f"##probe_scroll_{name}",
        imgui.ImVec2(canvas_w, height),
        0,
        imgui.WindowFlags_.no_scrollbar,
    ):
        imgui.end_child()
        return

    size = imgui.ImVec2(canvas_w, canvas_h)
    p0 = imgui.get_cursor_screen_pos()
    origin = imgui.ImVec2(
        p0.x + max(pad, (canvas_w - draw_w) * 0.5),
        p0.y + pad,
    )

    imgui.invisible_button(f"##probe_{name}", size)
    clicked = imgui.is_item_clicked()

    rects = _contact_rects(settings, origin, cell, contact_px)
    hit = _hit_contact(rects, imgui.is_item_hovered())
    settings.hovered_idx = hit

    if clicked and hit >= 0:
        settings.toggle_ch(hit)

    draw = imgui.get_window_draw_list()
    draw.add_rect_filled(
        p0,
        imgui.ImVec2(p0.x + size.x, p0.y + size.y),
        BG_U32,
        0.0,
    )
    _draw_contacts(draw, settings, rects)

    if hit >= 0:
        p1, p2 = rects[hit]
        if settings.visible[hit]:
            draw.add_rect_filled(
                p1,
                p2,
                settings.channel_color_u32(
                    hit,
                    alpha=1.0,
                    brighten=HOVER_BRIGHTEN,
                ),
                0.0,
            )
        draw.add_rect(p1, p2, HOVER_U32, 0.0, thickness=2.0)
        imgui.set_tooltip(
            f"ch {int(settings.channel_ids[hit])} | "
            f"shank {int(settings.shank_ids[hit])} | "
            f"x={settings.x[hit]:.0f}, y={settings.y[hit]:.0f}"
        )
    imgui.end_child()
