from viewer.controller import PlaybackController
from viewer.core import NS_PER_S, StreamView, seconds_to_ns


def make_controller(**kwargs) -> PlaybackController:
    return PlaybackController(0, 100 * NS_PER_S, view_span_ns=10 * NS_PER_S, **kwargs)


def test_initial_cursor_clamped_to_min():
    c = make_controller()
    assert c.cursor_ns == 0


def test_jump_to_clamps():
    c = make_controller()
    c.jump_to(-5 * NS_PER_S)
    assert c.cursor_ns == 0
    c.jump_to(500 * NS_PER_S)
    assert c.cursor_ns == 100 * NS_PER_S


def test_jump_by_advances_cursor():
    c = make_controller(initial_time_ns=10 * NS_PER_S)
    c.jump_by(2 * NS_PER_S)
    assert c.cursor_ns == 12 * NS_PER_S


def test_hard_jump_sets_jumped_flag():
    c = make_controller(initial_time_ns=20 * NS_PER_S, hard_jump_ratio=0.5)
    # view_span is 10s; threshold is 5s; jump 6s = hard jump
    c.jump_by(6 * NS_PER_S)
    req = c.viewport(width_px=100)
    assert req.jumped is True
    # jumped flag is one-shot
    req2 = c.viewport(width_px=100)
    assert req2.jumped is False


def test_small_jump_is_not_hard():
    c = make_controller(initial_time_ns=20 * NS_PER_S, hard_jump_ratio=0.5)
    c.jump_by(1 * NS_PER_S)
    assert c.viewport(width_px=100).jumped is False


def test_visible_range_centers_on_cursor():
    c = make_controller(initial_time_ns=50 * NS_PER_S)
    tr = c.visible_range
    assert tr.start_ns == 45 * NS_PER_S
    assert tr.stop_ns == 55 * NS_PER_S


def test_tick_advances_when_playing():
    c = make_controller(initial_time_ns=10 * NS_PER_S)
    c.play()
    c.tick(0.0)  # initialize last tick
    new_cursor = c.tick(1.0)  # 1s later
    assert new_cursor == 11 * NS_PER_S


def test_tick_does_not_advance_when_paused():
    c = make_controller(initial_time_ns=10 * NS_PER_S)
    c.tick(0.0)
    c.tick(1.0)
    assert c.cursor_ns == 10 * NS_PER_S


def test_viewport_carries_state():
    c = make_controller(initial_time_ns=30 * NS_PER_S)
    c.play()
    streams = (StreamView("a"),)
    req = c.viewport(width_px=400, streams=streams)
    assert req.width_px == 400
    assert req.streams == streams
    assert req.cursor_ns == 30 * NS_PER_S
    assert req.playing is True
    assert req.direction == 1


def test_set_view_span_clamps():
    c = make_controller()
    c.set_view_span(0)
    assert c.view_span_ns == 1
    c.set_view_span(10**12)
    assert c.view_span_ns == 100 * NS_PER_S


def test_seconds_accessors():
    c = make_controller(initial_time_ns=seconds_to_ns(2.5))
    assert c.cursor_s == 2.5
    assert c.t_min_s == 0.0
    assert c.t_max_s == 100.0
    assert c.view_span_s == 10.0
