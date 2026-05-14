from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from . import probe
from viewer.stream import Ephys
from viewer.ui import draw_cursor


class EphysSettings:
    def __init__(
        self,
        stream: Ephys,
        *,
        gain: float = 1.0,
        spacing: float = 1.0,
        width: float = 1.0,
        envelope_threshold: float = 2.0,
    ):
        self.gain = gain
        self.spacing = spacing
        self.width = width
        self.envelope_threshold = envelope_threshold
        self.probe = probe.ProbeSettings(stream.geometry)
        self._last_y_limits: tuple[float, float] | None = None

    def y_limits(self, n_channels: int) -> tuple[float, float]:
        if n_channels == 0:
            return -1.0, 1.0
        row_max = (n_channels - 1) * self.spacing
        pad = max(abs(self.gain), 0.5 * self.spacing, 0.25)
        return -pad, row_max + pad

    def draw_settings(self, name: str, stream: Ephys):
        imgui.text("Line width")
        imgui.set_next_item_width(-1)
        _, self.width = imgui.slider_float(
            f"##width_{name}", self.width, 0.5, 4.0, "%.1f"
        )

        imgui.text("Gain")
        imgui.set_next_item_width(-1)
        _, self.gain = imgui.slider_float(
            f"##gain_{name}", self.gain, 0.05, 50.0, "%.2f"
        )

        imgui.text("Channel spacing")
        imgui.set_next_item_width(-1)
        _, self.spacing = imgui.slider_float(
            f"##spacing_{name}", self.spacing, 0.1, 200.0, "%.1f"
        )

        imgui.separator()
        if imgui.button(f"All##probe_all_{name}"):
            self.probe.visible[:] = True
        imgui.same_line()
        if imgui.button(f"None##probe_none_{name}"):
            self.probe.visible[:] = False

        for shank in self.probe.shanks:
            imgui.same_line()
            if imgui.button(f"S{int(shank)}##probe_shank_{name}_{int(shank)}"):
                self.probe.toggle_shank(int(shank))

        visible_count = int(np.count_nonzero(self.probe.visible))
        imgui.text_disabled(f"{visible_count} / {self.probe.n_channels} channels")
        probe.draw_widget(name, self.probe)

    def draw_plot(
        self,
        stream: Ephys,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
    ):
        channel_indices = np.flatnonzero(self.probe.visible)
        visible_count = int(len(channel_indices))
        y_limits = self.y_limits(visible_count)

        if implot.begin_plot(f"{stream.name}", flags=implot.Flags_.no_legend):
            implot.setup_axes("Time (s)", stream.units, 0, 0)
            _setup_y_limits(self, y_limits)
            implot.setup_axis_zoom_constraints(implot.ImAxis_.y1, 1e-6, 1e12)
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

            if visible_count == 0:
                implot.plot_text("No channels selected", t, 0.0)
            elif not chunks:
                implot.plot_text("Loading...", t, 0.0)
            else:
                width_px = max(1.0, implot.get_plot_size().x)
                for item in stream.iter_visible_channels(
                    chunks,
                    view_t0.value,
                    view_t1.value,
                    width_px,
                    channel_indices,
                    envelope_threshold=self.envelope_threshold,
                ):
                    if item["mode"] == "raw":
                        _plot_raw(stream, item, self)
                    else:
                        _plot_envelope(stream, item, self)

            draw_cursor(t)
            implot.end_plot()


def _setup_y_limits(settings: EphysSettings, y_limits: tuple[float, float]):
    cond = imgui.Cond_.once
    if settings._last_y_limits != y_limits:
        cond = imgui.Cond_.always
        settings._last_y_limits = y_limits
    implot.setup_axis_limits(implot.ImAxis_.y1, y_limits[0], y_limits[1], cond)


def _iter_plot_channels(channel_indices: np.ndarray, hovered_idx: int):
    hovered = int(hovered_idx)
    hovered_item = None

    for col, ch_idx in enumerate(channel_indices):
        ch = int(ch_idx)
        if ch == hovered:
            hovered_item = (col, ch, True)
        else:
            yield col, ch, False

    if hovered_item is not None:
        yield hovered_item


def _trace_line_weight(settings: EphysSettings, highlighted: bool) -> float:
    if highlighted:
        return max(1.0, settings.width + 1.5)
    return settings.width


def _trace_brighten(highlighted: bool) -> float:
    return probe.HOVER_BRIGHTEN if highlighted else 0.0


def _plot_raw(
    stream: Ephys,
    item: dict,
    settings: EphysSettings,
):
    xstart = item["sample_start"] / stream.fs
    xscale = item["dt"]
    data = item["data"]
    channel_indices = item["channel_indices"]

    for col, ch_idx, highlighted in _iter_plot_channels(
        channel_indices,
        settings.probe.hovered_idx,
    ):
        row = col
        ys = np.ascontiguousarray(data[:, col])
        ys = (
            (ys * stream.scale + stream.offset) * settings.gain
            + row * settings.spacing
        )
        spec = implot.Spec(line_weight=_trace_line_weight(settings, highlighted))
        spec.line_color = settings.probe.channel_color_vec(
            ch_idx,
            brighten=_trace_brighten(highlighted),
        )
        label = settings.probe.channel_label(ch_idx)
        implot.plot_line(
            f"{label}##raw_{stream.name}_{ch_idx}",
            ys,
            xscale=xscale,
            xstart=xstart,
            spec=spec,
        )


def _plot_envelope(
    stream: Ephys,
    item: dict,
    settings: EphysSettings,
):
    xs = np.ascontiguousarray(item["t"])
    draw_list = implot.get_plot_draw_list()
    channel_indices = item["channel_indices"]

    implot.push_plot_clip_rect()
    try:
        for col, ch_idx, highlighted in _iter_plot_channels(
            channel_indices,
            settings.probe.hovered_idx,
        ):
            row = col
            y_min = item["y_min"][:, col] * stream.scale + stream.offset
            y_max = item["y_max"][:, col] * stream.scale + stream.offset
            y1 = y_min * settings.gain + row * settings.spacing
            y2 = y_max * settings.gain + row * settings.spacing
            lo = np.minimum(y1, y2)
            hi = np.maximum(y1, y2)
            color = settings.probe.channel_color_u32(
                ch_idx,
                alpha=1.0 if highlighted else 0.9,
                brighten=_trace_brighten(highlighted),
            )
            line_weight = _trace_line_weight(settings, highlighted)

            for x, y_lo, y_hi in zip(xs, lo, hi):
                p_lo = implot.plot_to_pixels(float(x), float(y_lo))
                p_hi = implot.plot_to_pixels(float(x), float(y_hi))
                draw_list.add_line(
                    imgui.ImVec2(p_lo.x, p_lo.y),
                    imgui.ImVec2(p_hi.x, p_hi.y),
                    color,
                    line_weight,
                )
    finally:
        implot.pop_plot_clip_rect()
