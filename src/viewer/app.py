from imgui_bundle import hello_imgui, imgui, immapp, implot

from viewer.cache import ChunkCache
from viewer.controller import TimeController
from viewer.render import DEFAULT_RENDERERS
from viewer.stream import Stream
from viewer.ui import gui_transport


class AppState:
    def __init__(
            self,
            streams: list[Stream],
            *,
            renderers=None,
            span: float = 5.0,
            max_workers: int = 2,
    ):
        if not streams:
            raise ValueError("run_viewer needs at least one stream")

        self.cache = ChunkCache(max_workers=max_workers)
        self.renderers = dict(DEFAULT_RENDERERS)
        if renderers:
            self.renderers.update(renderers)
        self.specs = {}

        for stream in streams:
            if stream.kind not in self.renderers:
                raise ValueError(f"No renderer registered for stream kind: {stream.kind}")

            self.cache.add(stream)
            self.specs[stream.name] = self.renderers[stream.kind].make_spec(stream)

        t_min = min(s.t_min for s in streams)
        t_max = max(s.t_max for s in streams)
        self.controller = TimeController(t_min=t_min, t_max=t_max, span=span)

    def reset(self):
        self.controller.reset()
        self.cache.reset()

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
        if state.specs[name].visible
    ]
    if not visible_streams:
        return

    rows = len(visible_streams)
    size = imgui.ImVec2(-1, imgui.get_content_region_avail().y)
    flags = (
        implot.SubplotFlags_.link_all_x
        | implot.SubplotFlags_.no_menus
        | implot.SubplotFlags_.no_title
    )

    if implot.begin_subplots("##streams", rows, 1, size, flags):
        for name, stream in visible_streams:
            chunks = cache.get_chunks(name, t)
            renderer = state.renderers[stream.kind]
            renderer.draw_plot(stream, chunks, state.specs[name], t, view_t0, view_t1)

        implot.end_subplots()
        ctrl.update_view(view_t0.value, view_t1.value)

    for name, stream in visible_streams:
        cache.request(name, ctrl.t_cursor)


def gui_settings(state: AppState):
    for name, stream in state.cache.streams.items():
        if imgui.collapsing_header(f"{name}##settings", imgui.TreeNodeFlags_.default_open):
            renderer = state.renderers[stream.kind]
            renderer.draw_settings(name, state.specs[name])


def run_viewer(
    streams: list[Stream],
    *,
    title: str = 'Viewer',
    window_size: tuple[int, int] = (1280, 720),
    renderers=None,
    span: float = 5.0,
    max_workers: int = 2,
):
    state = AppState(
        streams,
        renderers=renderers,
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

    # --- dockable windows ---
    win_plot = hello_imgui.DockableWindow()
    win_plot.label = "Streams"
    win_plot.dock_space_name = "MainDockSpace"
    win_plot.gui_function = lambda: gui_plot(state)
    win_plot.can_be_closed = False

    win_transport = hello_imgui.DockableWindow()
    win_transport.label = "Transport"
    win_transport.dock_space_name = "TransportDock"
    win_transport.gui_function = lambda: gui_transport(state)
    win_transport.can_be_closed = False

    win_settings = hello_imgui.DockableWindow()
    win_settings.label = "Settings"
    win_settings.dock_space_name = "SettingsDock"
    win_settings.gui_function = lambda: gui_settings(state)
    win_settings.can_be_closed = False

    params.docking_params.docking_splits = [split_right, split_bottom]
    params.docking_params.dockable_windows = [win_plot, win_transport, win_settings]

    addons = immapp.AddOnsParams()
    addons.with_implot = True

    params.callbacks.setup_imgui_style = lambda: imgui.style_colors_classic()

    try:
        immapp.run(runner_params=params, add_ons_params=addons)
    finally:
        state.close()
