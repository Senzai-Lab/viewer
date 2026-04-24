from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot
from viewer import ChunkManager, DenseSignalAdapter, PlaybackController


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
            DenseSignalAdapter(
                "signal-a",
                stream_a,
                sample_rate_hz,
                target_chunk_bytes=128 * 1024,
            ),
            DenseSignalAdapter(
                "signal-b",
                stream_b,
                sample_rate_hz,
                target_chunk_bytes=128 * 1024,
            ),
        ]
        self.adapters_by_id = {adapter.stream_id: adapter for adapter in self.adapters}
        self.stream_ids = [adapter.stream_id for adapter in self.adapters]
        self.controller = PlaybackController(
            min(adapter.t_min for adapter in self.adapters),
            max(adapter.t_max for adapter in self.adapters),
            initial_time=5.0,
            view_span=8.0,
            playback_speed=1.0,
        )
        self.cache = ChunkManager(self.adapters, max_workers=2)
        self.render_visible: dict[str, dict[str, np.ndarray] | None] = {
            stream_id: None for stream_id in self.stream_ids
        }

    def close(self) -> None:
        self.cache.close()


def draw_transport(state: AppState) -> None:
    controller = state.controller

    if imgui.button("Pause" if controller.is_playing else "Play"):
        if controller.is_playing:
            controller.pause()
        else:
            controller.play()

    imgui.same_line()
    if imgui.button("-5 s"):
        controller.pause()
        controller.jump_by(-5.0)

    imgui.same_line()
    if imgui.button("+5 s"):
        controller.pause()
        controller.jump_by(5.0)

    changed, new_t = imgui.slider_float(
        "time",
        float(controller.cursor_t),
        float(controller.t_min),
        float(controller.t_max),
    )
    if changed:
        controller.pause()
        controller.jump_to(new_t)

    changed, new_span = imgui.slider_float(
        "view span",
        float(controller.view_span),
        0.25,
        30.0,
    )
    if changed:
        controller.set_view_span(new_span)

    imgui.text(
        f"cursor={controller.cursor_t:.1f} s  span={controller.view_span:.3f} s  fps={hello_imgui.frame_rate():.1f}"
    )


def draw_dense_stream_plot(
    stream_id: str,
    request,
    visible: dict[str, np.ndarray] | None,
) -> None:
    if implot.begin_plot(f"{stream_id}##plot", imgui.ImVec2(-1, 220)):
        implot.setup_axes("time (s)", "value")
        implot.setup_axis_limits(
            implot.ImAxis_.x1,
            request.view.start,
            request.view.end,
            imgui.Cond_.always,
        )
        implot.setup_axis_limits(
            implot.ImAxis_.y1,
            -1.5,
            1.5,
            imgui.Cond_.always,
        )
        if visible is not None:
            xs = np.ascontiguousarray(visible["times"], dtype=np.float64)
            values = np.ascontiguousarray(visible["values"], dtype=np.float64)
            for channel_index, channel_values in enumerate(values):
                ys = np.ascontiguousarray(channel_values, dtype=np.float64)
                implot.plot_line(f"{stream_id}/ch{channel_index}", xs, ys)
        else:
            implot.plot_text(f"loading##{stream_id}", request.cursor_t, 0.0)

        implot.end_plot()


def build_app(state: AppState):
    def app() -> None:
        controller = state.controller

        controller.tick(imgui.get_io().delta_time)

        request = controller.make_request()
        stream_states = state.cache.poll(request)

        for stream_id in state.stream_ids:
            stream_state = stream_states[stream_id]

            # Promote only fully covered visible payloads into the renderer.
            if stream_state.complete:
                visible = state.adapters_by_id[stream_id].build_view(
                    stream_state.ready_chunks,
                    view=request.view,
                )
                if visible is not None and visible["times"].size > 1:
                    state.render_visible[stream_id] = visible
            elif request.jumped:
                # On a random jump, drop the previous payload for this stream.
                state.render_visible[stream_id] = None

        imgui.begin("Dense streams")
        draw_transport(state)

        for stream_id in state.stream_ids:
            stream_state = stream_states[stream_id]
            imgui.separator()
            imgui.text(
                f"{stream_id}: coverage={stream_state.coverage:.2f}  ready={len(stream_state.ready_chunks)}  missing={stream_state.missing_count}"
            )
            draw_dense_stream_plot(stream_id, request, state.render_visible[stream_id])

        imgui.end()

    return app


def main() -> None:
    state = AppState()
    try:
        immapp.run(
            build_app(state),
            window_size=(1200, 700),
            with_implot=True,
            fps_idle=0.0,
        )
    finally:
        state.close()


if __name__ == "__main__":
    main()