import time

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot

from viewer import (
    ChunkManager,
    DenseSignalAdapter,
    PlaybackController,
    Selection,
    StreamView,
    ns_to_seconds,
    seconds_to_ns,
)


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
        self.stream_views = (
            StreamView("signal-a", Selection(channels=(0, 1))),
            StreamView("signal-b", Selection(channels=(0,))),
        )
        self.controller = PlaybackController(
            min(adapter.time_range.start_ns for adapter in self.adapters),
            max(adapter.time_range.stop_ns for adapter in self.adapters),
            initial_time_ns=seconds_to_ns(5.0),
            view_span_ns=seconds_to_ns(8.0),
            playback_speed=1.0,
        )
        self.session = ChunkManager(
            self.adapters,
            max_workers=2,
            memory_budget_bytes=64 * 1024 * 1024,
        )

    def close(self) -> None:
        self.session.close()


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
        controller.jump_by(seconds_to_ns(-5.0))

    imgui.same_line()
    if imgui.button("+5 s"):
        controller.pause()
        controller.jump_by(seconds_to_ns(5.0))

    changed, new_t = imgui.slider_float(
        "time",
        controller.cursor_s,
        controller.t_min_s,
        controller.t_max_s,
    )
    if changed:
        controller.pause()
        controller.jump_to(seconds_to_ns(new_t))

    changed, new_span = imgui.slider_float(
        "view span",
        controller.view_span_s,
        0.25,
        30.0,
    )
    if changed:
        controller.set_view_span(seconds_to_ns(new_span))

    imgui.text(
        f"cursor={controller.cursor_s:.3f} s  span={controller.view_span_s:.3f} s  fps={hello_imgui.frame_rate():.1f}"
    )


def draw_dense_stream_plot(
    stream_id: str,
    request,
    payload: dict[str, np.ndarray] | None,
) -> None:
    if implot.begin_plot(f"{stream_id}##plot", imgui.ImVec2(-1, 220)):
        implot.setup_axes("time (s)", "value")
        implot.setup_axis_limits(
            implot.ImAxis_.x1,
            ns_to_seconds(request.time.start_ns),
            ns_to_seconds(request.time.stop_ns),
            imgui.Cond_.always,
        )
        implot.setup_axis_limits(
            implot.ImAxis_.y1,
            -1.5,
            1.5,
            imgui.Cond_.always,
        )
        if payload is not None and payload["time_ns"].size:
            xs = np.ascontiguousarray(payload["time_ns"], dtype=np.float64) / 1_000_000_000.0
            values = np.ascontiguousarray(payload["values"], dtype=np.float64)
            channel_indices = np.ascontiguousarray(payload["channel_indices"], dtype=np.int64)
            for channel_index, channel_values in enumerate(values):
                ys = np.ascontiguousarray(channel_values, dtype=np.float64)
                implot.plot_line(f"{stream_id}/ch{int(channel_indices[channel_index])}", xs, ys)
        else:
            implot.plot_text(f"loading##{stream_id}", ns_to_seconds(request.cursor_ns), 0.0)

        implot.end_plot()


def build_app(state: AppState):
    def app() -> None:
        controller = state.controller

        controller.tick(time.monotonic())

        imgui.begin("Dense streams")
        draw_transport(state)

        plot_width = max(1, int(imgui.get_content_region_avail().x))
        viewport = controller.viewport(
            width_px=plot_width,
            streams=state.stream_views,
        )
        frames = state.session.update(viewport)

        for stream_view in state.stream_views:
            stream_frame = frames[stream_view.stream_id]
            imgui.separator()
            imgui.text(
                f"{stream_view.stream_id}: coverage={stream_frame.coverage:.2f}  ready={stream_frame.ready}"
            )
            draw_dense_stream_plot(stream_view.stream_id, viewport, stream_frame.data)

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