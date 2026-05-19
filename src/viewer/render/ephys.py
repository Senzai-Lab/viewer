from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from . import probe as probe_render
from viewer.stream import Ephys
from viewer.ui import setup_time_axis


class EphysView:
    def __init__(
        self,
        *,
        probe: probe_render.ProbeView,
        width: float = 1.0,
        gain: float = 1.0,
        envelope_threshold: float = 2.0,
    ):
        self.width = width
        self.gain = gain
        self.envelope_threshold = envelope_threshold
        self.probe = probe
        self._last_y_limits: tuple[float, float] | None = None

    def y_limits(
        self,
        probe_view: probe_render.ProbeView,
        channel_indices: np.ndarray,
    ) -> tuple[float, float]:
        if len(channel_indices) == 0:
            return -1.0, 1.0

        ys = probe_view.plot_y[channel_indices]
        pad = 1.0
        return float(ys.min() - pad), float(ys.max() + pad)

    def draw_settings(self, stream: Ephys, cache):
        name = stream.name
        probe_view = self.probe
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
            probe_view.visible[:] = True
        imgui.same_line()
        if imgui.button(f"None##probe_none_{name}"):
            probe_view.visible[:] = False

        for shank in probe_view.shanks:
            imgui.same_line()
            if imgui.button(f"S{int(shank)}##probe_shank_{name}_{int(shank)}"):
                probe_view.toggle_shank(int(shank))

        visible_count = int(np.count_nonzero(probe_view.visible))
        imgui.text_disabled(f"{visible_count} / {probe_view.n_channels} channels")
        probe_render.draw_widget(name, probe_view)

    def draw_plot(
        self,
        stream: Ephys,
        chunks: list[dict],
        t: float,
        view_t0: implot.BoxedValue,
        view_t1: implot.BoxedValue,
        overlays=(),
        *,
        time_axis: str = "clock",
    ):
        probe_view = self.probe
        channel_indices = np.asarray(
            np.flatnonzero(probe_view.visible),
            dtype=np.intp,
        )
        visible_count = int(len(channel_indices))
        y_limits = self.y_limits(probe_view, channel_indices)

        if implot.begin_plot(f"{stream.name}", flags=implot.Flags_.no_legend):
            setup_time_axis("Channel offset", time_axis=time_axis)
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
                        _plot_raw(stream, item, self, probe_view, channel_indices)
                    else:
                        _plot_envelope(item, self, probe_view, channel_indices)

            for overlay in overlays:
                overlay.draw_overlay()
            implot.end_plot()


def _setup_y_limits(settings: EphysView, y_limits: tuple[float, float]):
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


def _trace_line_weight(settings: EphysView, highlighted: bool) -> float:
    if highlighted:
        return max(1.0, settings.width + 1.5)
    return settings.width


def _trace_brighten(highlighted: bool) -> float:
    return probe_render.HOVER_BRIGHTEN if highlighted else 0.0


def _plot_raw(
    stream: Ephys,
    item: dict,
    settings: EphysView,
    probe_view: probe_render.ProbeView,
    channel_indices: np.ndarray,
):
    xstart = item["t_start"]
    xscale = item["dt"]
    data = item["data"]

    for _, ch_idx, highlighted in _iter_plot_channels(
        channel_indices,
        probe_view.hovered_idx,
    ):
        baseline = probe_view.plot_y[ch_idx]
        ys = np.ascontiguousarray(data[:, ch_idx] * settings.gain + baseline)
        spec = implot.Spec(line_weight=_trace_line_weight(settings, highlighted))
        spec.line_color = probe_view.channel_color_vec(
            ch_idx,
            brighten=_trace_brighten(highlighted),
        )
        label = probe_view.channel_label(ch_idx)
        implot.plot_line(
            f"{label}##raw_{stream.name}_{ch_idx}",
            ys,
            xscale=xscale,
            xstart=xstart,
            spec=spec,
        )


def _plot_envelope(
    item: dict,
    settings: EphysView,
    probe_view: probe_render.ProbeView,
    channel_indices: np.ndarray,
):
    xs = np.ascontiguousarray(item["t"])
    draw_list = implot.get_plot_draw_list()

    implot.push_plot_clip_rect()
    try:
        for col, ch_idx, highlighted in _iter_plot_channels(
            channel_indices,
            probe_view.hovered_idx,
        ):
            baseline = probe_view.plot_y[ch_idx]
            lo = np.ascontiguousarray(
                item["y_min"][:, col] * settings.gain + baseline
            )
            hi = np.ascontiguousarray(
                item["y_max"][:, col] * settings.gain + baseline
            )
            color = probe_view.channel_color_u32(
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
