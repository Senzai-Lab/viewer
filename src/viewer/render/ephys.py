from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from . import probe
from viewer.stream import Ephys


class EphysSettings:
    def __init__(
        self,
        geometry: dict | Ephys,
        *,
        width: float = 1.0,
        gain: float = 1.0,
        envelope_threshold: float = 2.0,
        plot_shank_gap: float = 4.0,
    ):
        self.width = width
        self.gain = gain
        self.envelope_threshold = envelope_threshold
        if isinstance(geometry, Ephys):
            geometry = geometry.geometry
        self.probe = probe.ProbeSettings(
            geometry,
            plot_shank_gap=plot_shank_gap,
        )
        self._last_y_limits: tuple[float, float] | None = None

    def y_limits(self, channel_indices: np.ndarray) -> tuple[float, float]:
        if len(channel_indices) == 0:
            return -1.0, 1.0

        ys = self.probe.plot_y[channel_indices]
        pad = 1.0
        return float(ys.min() - pad), float(ys.max() + pad)

    def draw_settings(self, stream: Ephys, cache):
        name = stream.name
        imgui.text("Line width")
        imgui.set_next_item_width(-1)
        _, self.width = imgui.drag_float(
            f"##width_{name}", self.width, 0.05, 0.5, 4.0, "%.1f"
        )
        imgui.text("Signal gain")
        imgui.set_next_item_width(-1)
        _, self.gain = imgui.drag_float(
            f"##gain_{name}", self.gain, 0.01, 0.0, 0.0, "%.4f"
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
        overlays=(),
    ):
        channel_indices = np.asarray(np.flatnonzero(self.probe.visible), dtype=np.intp)
        visible_count = int(len(channel_indices))
        y_limits = self.y_limits(channel_indices)

        if implot.begin_plot(f"{stream.name}", flags=implot.Flags_.no_legend):
            implot.setup_axes("Time (s)", "Channel offset", 0, 0)
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
                        _plot_raw(stream, item, self, channel_indices)
                    else:
                        _plot_envelope(item, self, channel_indices)

            for overlay in overlays:
                overlay.draw_overlay()
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
    channel_indices: np.ndarray,
):
    xstart = item["t_start"]
    xscale = item["dt"]
    data = item["data"]

    for _, ch_idx, highlighted in _iter_plot_channels(
        channel_indices,
        settings.probe.hovered_idx,
    ):
        baseline = settings.probe.plot_y[ch_idx]
        ys = np.ascontiguousarray(data[:, ch_idx] * settings.gain + baseline)
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
    item: dict,
    settings: EphysSettings,
    channel_indices: np.ndarray,
):
    xs = np.ascontiguousarray(item["t"])
    draw_list = implot.get_plot_draw_list()

    implot.push_plot_clip_rect()
    try:
        for col, ch_idx, highlighted in _iter_plot_channels(
            channel_indices,
            settings.probe.hovered_idx,
        ):
            baseline = settings.probe.plot_y[ch_idx]
            lo = np.ascontiguousarray(
                item["y_min"][:, col] * settings.gain + baseline
            )
            hi = np.ascontiguousarray(
                item["y_max"][:, col] * settings.gain + baseline
            )
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
