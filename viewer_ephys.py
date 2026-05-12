
from viewer.cache import ChunkCache
from viewer.stream import Stream, TimeSeries
from viewer.controller import TimeController
from viewer.ui import draw_cursor, draw_timeseries_spec, gui_transport, TimeSeriesSpec
from imgui_bundle import immapp, implot, imgui, hello_imgui

import zarr
import numpy as np

DR_SPEC_FN = {
    'timeseries': draw_timeseries_spec
}

def draw_timeseries(
        stream: Stream,
        chunks: list[dict],
        sspec: TimeSeriesSpec,
        t: float,
        view_t0: implot.BoxedValue | None = None,
        view_t1: implot.BoxedValue | None = None
        ):
    
    if implot.begin_plot(f"{stream.name}"):
        y_flags = implot.AxisFlags_.auto_fit if chunks else 0
        implot.setup_axes("Time (s)", "Value", 0, y_flags)
        implot.setup_axis_zoom_constraints(implot.ImAxis_.y1, 1e-6, 1e12)
        
        if view_t0 and view_t1:
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)
        
        if not chunks:
            implot.plot_text("Loading...", t, 0.0)
        
        # Downsample based on samples per pixel column
        plot_width_px = implot.get_plot_size().x
        viewport_samples = stream.fs * (view_t1.value - view_t0.value)
        stride = max(1, int(viewport_samples / max(plot_width_px, 1)))
        t_scale = stride / stream.fs

        for chunk in chunks:
            t_start = chunk['t_start']
            for ch in range(chunk['data'].shape[1]):
                ys = chunk['data'][:, ch]
                if stride > 2:
                    ys = np.ascontiguousarray(ys[::stride])
                ys = ys * sspec.gain + ch * sspec.ch_offset
                implot.plot_line(f"ch{ch}",
                                 ys,
                                 xscale=t_scale,
                                 xstart=t_start,
                                 spec=sspec.to_implot_spec())
        
        draw_cursor(t)
        implot.end_plot()


DR_PLOT_FN = {
    'timeseries': draw_timeseries
}

class AppState:
    def __init__(self, streams: list):
        self.cache = ChunkCache(max_workers=2)
        for s in streams:
            self.cache.add(s)
        
        t_min = min(float(s.ts[0]) for s in streams)
        t_max = max(float(s.ts[-1]) for s in streams)

        self.controller = TimeController(t_min=t_min, t_max=t_max, span=5.0)
        self.specs = {s.name: TimeSeriesSpec() for s in streams}
    
    def reset(self):
        self.controller.reset()
        self.cache.reset()

def gui_plot(state: AppState):
    ctrl = state.controller
    cache = state.cache

    dt = min(imgui.get_io().delta_time, 0.001)
    t = ctrl.tick(dt)
    cache.poll() # collect loaded chunks from threads

    view_t0 = implot.BoxedValue(ctrl.view_t0)
    view_t1 = implot.BoxedValue(ctrl.view_t1)

    visible_streams = [(n, s) for n, s in cache.streams.items() if state.specs[n].visible]
    
    rows = len(visible_streams)
    if rows == 0:
        return
    size = imgui.ImVec2(-1, imgui.get_content_region_avail().y)
    flags = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_menus | implot.SubplotFlags_.no_title
    if implot.begin_subplots('##streams', rows, 1, size, flags):
        for name, stream in visible_streams:
            sspec = state.specs[name]
            chunks = cache.get_range(name, ctrl.view_t0, ctrl.view_t1)
            
            DR_PLOT_FN[stream.kind](stream, chunks, sspec, t, view_t0, view_t1)

            ctrl.update_view(view_t0.value, view_t1.value)
            cache.prefetch(name, t)
        implot.end_subplots()


def gui_settings(state: AppState):
    for name, stream in state.cache.streams.items():
        if imgui.collapsing_header(f"{name}##settings", imgui.TreeNodeFlags_.default_open):
            DR_SPEC_FN[stream.kind](name, state.specs[name])


if __name__ == '__main__':
    root = zarr.open('/Users/iii9781/viewer/scripts/exp1.zarr', mode='r')

    ephys_grp = root['ephys']
    ephys = TimeSeries(
        name='ephys',
        values=ephys_grp['values'],
        ts=ephys_grp['ts'],
        fs=ephys_grp.attrs['fs'],
    )

    pupil_grp = root['behavior/pupil']
    pupil = TimeSeries(
        name='pupil',
        values=pupil_grp['values'],
        ts=pupil_grp['ts'],
        fs=pupil_grp.attrs['fs'],
    )

    state = AppState(streams=[ephys, pupil])

    params = immapp.RunnerParams()
    params.app_window_params.window_title = "Viewer"
    params.app_window_params.window_geometry.size = (1280, 720)
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

    immapp.run(runner_params=params, add_ons_params=addons)