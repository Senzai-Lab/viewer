from viewer.cache import ChunkCache
from viewer.stream import Stream, Units
from viewer.controller import TimeController
from viewer.ui import draw_cursor, gui_transport
from imgui_bundle import immapp, implot, imgui, hello_imgui
from cmap import Colormap
from dataclasses import dataclass, field

import zarr
import numpy as np


# --- Spec ---

@dataclass
class UnitsSpec:
    visible: bool = True
    tick_height: float = 5.0
    thickness: float = 1.0
    unit_offset: float = 1.0
    cmap_name: str = "cmocean:phase"
    color_key: str = "rate"          # metadata key for colormap, or "" for unit index
    sort_key: str = ""               # metadata key for row ordering, or "" for unit id

    # populated at init from stream metadata
    unit_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    metadata: dict = field(default_factory=dict, repr=False)

    # cached sort order: unit_id -> row position
    _sorted_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    _uid_to_row: dict = field(default_factory=dict, repr=False)
    _sort_cache_key: str = field(default="", repr=False)

    # cached colors (recomputed when color_key/cmap changes)
    _colors: list = field(default_factory=list, repr=False)
    _color_cache_key: tuple = field(default=(), repr=False)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def get_sorted_ids(self) -> np.ndarray:
        if self._sort_cache_key == self.sort_key and len(self._sorted_ids) == self.n_units:
            return self._sorted_ids

        if self.sort_key and self.sort_key in self.metadata:
            meta_dict = self.metadata[self.sort_key]
            vals = np.array([float(meta_dict.get(str(uid), 0.0)) for uid in self.unit_ids])
            order = np.argsort(vals)
            self._sorted_ids = self.unit_ids[order]
        else:
            self._sorted_ids = self.unit_ids.copy()

        self._uid_to_row = {int(uid): i for i, uid in enumerate(self._sorted_ids)}
        self._sort_cache_key = self.sort_key
        # invalidate colors since order changed
        self._color_cache_key = ()
        return self._sorted_ids

    def get_uid_to_row(self) -> dict:
        self.get_sorted_ids()
        return self._uid_to_row

    def get_colors(self) -> list:
        sorted_ids = self.get_sorted_ids()
        key = (self.color_key, self.cmap_name, self.n_units, self._sort_cache_key)
        if self._color_cache_key == key:
            return self._colors

        n = self.n_units
        if n == 0:
            self._colors = []
            self._color_cache_key = key
            return self._colors

        if self.color_key and self.color_key in self.metadata:
            meta_dict = self.metadata[self.color_key]
            vals = np.array([float(meta_dict.get(str(uid), 0.0)) for uid in sorted_ids])
            vmin, vmax = vals.min(), vals.max()
            norm = (vals - vmin) / max(vmax - vmin, 1e-9)
        else:
            norm = np.linspace(0, 1, n, dtype=np.float32)

        cm = Colormap(self.cmap_name)
        rgba = cm(norm.astype(np.float32))
        self._colors = [
            imgui.color_convert_float4_to_u32(
                imgui.ImVec4(float(r), float(g), float(b), float(a))
            )
            for r, g, b, a in rgba
        ]
        self._color_cache_key = key
        return self._colors


def draw_units_spec(name: str, spec: UnitsSpec):
    _, spec.visible = imgui.checkbox(f"Visible##{name}", spec.visible)

    imgui.set_next_item_width(-1)
    _, spec.tick_height = imgui.slider_float(
        f"##tick_{name}", spec.tick_height, 1.0, 20.0, "Tick height: %.1f px"
    )
    imgui.set_next_item_width(-1)
    _, spec.thickness = imgui.slider_float(
        f"##thick_{name}", spec.thickness, 0.5, 4.0, "Thickness: %.1f"
    )
    imgui.set_next_item_width(-1)
    _, spec.unit_offset = imgui.slider_float(
        f"##offset_{name}", spec.unit_offset, 0.1, 5.0, "Unit offset: %.1f"
    )

    # colormap selector
    imgui.set_next_item_width(-1)
    changed, new_cmap = imgui.input_text(f"##cmap_{name}", spec.cmap_name)
    if imgui.is_item_hovered(imgui.HoveredFlags_.stationary):
        imgui.set_tooltip("Colormap name")
    if changed:
        spec.cmap_name = new_cmap

    # sort / color key combos
    meta_keys = ["(index)"] + [k for k in spec.metadata.keys()]

    # sort by
    sort_current = spec.sort_key if spec.sort_key else "(index)"
    sort_idx = meta_keys.index(sort_current) if sort_current in meta_keys else 0
    imgui.set_next_item_width(-1)
    changed, new_idx = imgui.combo(f"Sort by##{name}", sort_idx, meta_keys)
    if changed:
        spec.sort_key = "" if meta_keys[new_idx] == "(index)" else meta_keys[new_idx]

    # color by
    color_current = spec.color_key if spec.color_key else "(index)"
    color_idx = meta_keys.index(color_current) if color_current in meta_keys else 0
    imgui.set_next_item_width(-1)
    changed, new_idx = imgui.combo(f"Color by##{name}", color_idx, meta_keys)
    if changed:
        spec.color_key = "" if meta_keys[new_idx] == "(index)" else meta_keys[new_idx]


# --- Draw ---

def draw_raster(
        stream,
        chunks: list[dict],
        spec: UnitsSpec,
        t: float,
        view_t0=None,
        view_t1=None,
):
    n_units = spec.n_units
    colors = spec.get_colors()

    if implot.begin_plot(f"{stream.name}"):
        implot.setup_axes("Time (s)", "Unit", 0, 0)
        implot.setup_axis_limits(implot.ImAxis_.y1, -2, n_units, imgui.Cond_.always)

        if view_t0 and view_t1:
            implot.setup_axis_links(implot.ImAxis_.x1, view_t0, view_t1)

        if not chunks:
            implot.plot_text("Loading...", t, n_units / 2)

        limits = implot.get_plot_limits()
        x_min, x_max = limits.x.min, limits.x.max
        plot_width_px = max(implot.get_plot_size().x, 1.0)
        bin_width = (x_max - x_min) / plot_width_px

        draw_list = implot.get_plot_draw_list()
        implot.push_plot_clip_rect()
        try:
            for chunk in chunks:
                times = chunk['ts']
                unit_ids = chunk['data']
                if times.size == 0:
                    continue

                # clip to visible time range
                i0 = int(np.searchsorted(times, x_min, side='left'))
                i1 = int(np.searchsorted(times, x_max, side='right'))
                if i0 >= i1:
                    continue
                vis_t = times[i0:i1]
                vis_u = unit_ids[i0:i1]

                # pixel-column bins for downsampling
                bins = ((vis_t - x_min) / bin_width).astype(np.int32)
                np.clip(bins, 0, int(plot_width_px) - 1, out=bins)

                # draw one tick per (unit, pixel column)
                sorted_ids = spec.get_sorted_ids()
                for uid in sorted_ids:
                    mask = vis_u == uid
                    if not mask.any():
                        continue
                    occupied = np.unique(bins[mask])
                    row_idx = spec._uid_to_row[int(uid)]
                    row = float(row_idx) * spec.unit_offset
                    color = colors[row_idx]
                    half = spec.tick_height

                    for b in occupied:
                        t_bin = x_min + (b + 0.5) * bin_width
                        p = implot.plot_to_pixels(t_bin, row)
                        draw_list.add_line(
                            imgui.ImVec2(p.x, p.y - half),
                            imgui.ImVec2(p.x, p.y + half),
                            color,
                            spec.thickness,
                        )
        finally:
            implot.pop_plot_clip_rect()

        draw_cursor(t)
        implot.end_plot()


# --- Dispatch tables ---

DR_PLOT_FN = {
    'units': draw_raster,
}

DR_SPEC_FN = {
    'units': draw_units_spec,
}


# --- App ---

def make_units_spec(stream: Units) -> UnitsSpec:
    """Build a UnitsSpec populated with metadata from the stream."""
    metadata = stream.metadata
    # derive unit_ids from metadata keys (use 'rate' dict keys as canonical set)
    for key in ('rate', 'peak', 'order'):
        if key in metadata:
            unit_ids = np.array(sorted(int(k) for k in metadata[key].keys()), dtype=np.int64)
            break
    else:
        # fallback: read unique values from first chunk worth of data
        unit_ids = np.unique(np.asarray(stream.values[:10000]))

    return UnitsSpec(unit_ids=unit_ids, metadata=metadata)


class AppState:
    def __init__(self, streams: list):
        self.cache = ChunkCache(max_workers=2)
        for s in streams:
            self.cache.add(s)

        t_min = min(s.t_min for s in streams)
        t_max = max(s.t_max for s in streams)

        self.controller = TimeController(t_min=t_min, t_max=t_max, span=5.0)
        self.specs = {}
        for s in streams:
            if s.kind == 'units':
                self.specs[s.name] = make_units_spec(s)

    def reset(self):
        self.controller.reset()
        self.cache.reset()


def gui_plot(state: AppState):
    ctrl = state.controller
    cache = state.cache

    dt = min(imgui.get_io().delta_time, 0.001)
    t = ctrl.tick(dt)
    cache.poll()

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

    units_grp = root['units']
    units = Units(
        name='units',
        ts=units_grp['spike_times'],
        values=units_grp['spike_units'],
        metadata=dict(units_grp.attrs),
    )

    state = AppState(streams=[units])

    params = immapp.RunnerParams()
    params.app_window_params.window_title = "Raster Viewer"
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
    win_plot.label = "Raster"
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

