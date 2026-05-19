from typing import Any

from imgui_bundle import hello_imgui, imgui, immapp, implot

from viewer.cache import Cache
from viewer.controller import TimeController
from viewer.stream import Stream
from viewer.ui import TIME_AXIS_CLOCK, draw_stream_debug, gui_transport, setup_style


class AppState:
    def __init__(
            self,
            streams: list[tuple[Stream, Any]],
            *,
            event_bars=None,
            overlays=(),
            span: float = 5.0,
            max_workers: int = 2,
    ):
        if not streams:
            raise ValueError("show needs at least one stream")

        self.cache = Cache(workers=max_workers)
        self.views = {}
        self.visible = {}
        self.event_bars = event_bars
        self.overlays = overlays
        self.time_axis = TIME_AXIS_CLOCK

        for stream, view in streams:
            self.cache.add(stream)
            self.visible[stream.name] = True
            self.views[stream.name] = view

        t_min = min(stream.t_min for stream, _ in streams)
        t_max = max(stream.t_max for stream, _ in streams)
        self.controller = TimeController(t_min=t_min, t_max=t_max, span=span)

    def reset(self):
        self.controller.reset()
        self.cache.drop()

    def close(self):
        self.cache.close()


def gui_plot(state: AppState):
    ctrl = state.controller
    cache = state.cache

    dt = imgui.get_io().delta_time
    t = ctrl.tick(dt)
    cache.poll()

    view_t0 = implot.BoxedValue(ctrl.view_t0)
    view_t1 = implot.BoxedValue(ctrl.view_t1)

    visible_streams = [
        (name, stream)
        for name, stream in cache.streams.items()
        if state.visible[name]
    ]
    if not visible_streams and state.event_bars is None:
        return

    if visible_streams:
        rows = len(visible_streams)
        size = imgui.ImVec2(-1, imgui.get_content_region_avail().y)
        flags = (implot.SubplotFlags_.no_title)

        if implot.begin_subplots("##streams", rows, 1, size, flags):
            if state.event_bars is not None:
                state.event_bars.draw(
                    "Event Bars##top_event_bars",
                    t=t,
                    view_t0=view_t0,
                    view_t1=view_t1,
                    height=120.0,
                    time_axis=state.time_axis,
                )
            for name, stream in visible_streams:
                chunks = cache.chunks(stream, t)
                view = state.views[name]
                view.draw_plot(
                    stream,
                    chunks,
                    t,
                    view_t0,
                    view_t1,
                    state.overlays,
                    time_axis=state.time_axis,
                )

            implot.end_subplots()

    ctrl.update_view(view_t0.value, view_t1.value)

    for name, stream in visible_streams:
        cache.request(stream, ctrl.t_cursor)


def gui_settings(state: AppState):
    for name, stream in state.cache.streams.items():
        if imgui.collapsing_header(f"{name}##settings"):
            _, state.visible[name] = imgui.checkbox(f"Visible##{name}", state.visible[name])
            draw_stream_debug(state.cache, stream, state.controller.t_cursor)
            view = state.views[name]
            if imgui.tree_node_ex(f"Settings##stream_settings_{name}"):
                view.draw_settings(stream, state.cache)
                imgui.tree_pop()

            pipe = getattr(stream, "transforms", None)
            if pipe is not None and imgui.tree_node_ex(f"Pipe##pipe_{name}"):
                draw_settings = getattr(pipe, "draw_settings", None)
                if draw_settings is None:
                    imgui.text_disabled("No pipe settings")
                else:
                    changed = draw_settings(stream)
                    if changed:
                        stream.setup_pipe()
                        state.cache.drop(stream)
                imgui.tree_pop()



def show(
    streams: list[tuple[Stream, Any]],
    *,
    title: str = 'Viewer',
    window_size: tuple[int, int] = (1480, 900),
    event_bars=None,
    overlays=(),
    span: float = 2.0,
    max_workers: int = 3,
):
    state = AppState(
        streams,
        event_bars=event_bars,
        overlays=overlays,
        span=span,
        max_workers=max_workers,
    )

    params = immapp.RunnerParams()
    params.app_window_params.window_title = title
    params.app_window_params.window_geometry.size = window_size
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    params.imgui_window_params.enable_viewports = True
    params.fps_idling = hello_imgui.FpsIdling(fps_idle=0)
    
    # --- docking layout ---
    split_right = hello_imgui.DockingSplit()
    split_right.initial_dock = "MainDockSpace"
    split_right.new_dock = "SettingsDock"
    split_right.direction = imgui.Dir_.right
    split_right.ratio = 0.2

    split_bottom = hello_imgui.DockingSplit()
    split_bottom.initial_dock = "MainDockSpace"
    split_bottom.new_dock = "TransportDock"
    split_bottom.direction = imgui.Dir_.down
    split_bottom.ratio = 0.08
    params.docking_params.main_dock_space_node_flags = imgui.DockNodeFlags_.auto_hide_tab_bar

    # --- dockable windows ---
    dock_window_flags = imgui.WindowFlags_.no_title_bar

    win_plot = hello_imgui.DockableWindow()
    win_plot.label = "Streams"
    win_plot.dock_space_name = "MainDockSpace"
    win_plot.gui_function = lambda: gui_plot(state)
    win_plot.can_be_closed = False
    win_plot.imgui_window_flags = dock_window_flags

    win_transport = hello_imgui.DockableWindow()
    win_transport.label = "Transport"
    win_transport.dock_space_name = "TransportDock"
    win_transport.gui_function = lambda: gui_transport(state)
    win_transport.can_be_closed = False
    win_transport.imgui_window_flags = dock_window_flags | imgui.WindowFlags_.no_move

    win_settings = hello_imgui.DockableWindow()
    win_settings.label = "Settings"
    win_settings.dock_space_name = "SettingsDock"
    win_settings.gui_function = lambda: gui_settings(state)
    win_settings.can_be_closed = False
    # win_settings.imgui_window_flags = dock_window_flags

    params.docking_params.docking_splits = [split_right, split_bottom]
    params.docking_params.dockable_windows = [win_plot, win_transport, win_settings]

    addons = immapp.AddOnsParams()
    addons.with_implot = True

    params.callbacks.setup_imgui_style = setup_style

    try:
        immapp.run(runner_params=params, add_ons_params=addons)
    finally:
        state.close()
