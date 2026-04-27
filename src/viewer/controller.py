from __future__ import annotations

from .core import (
    NS_PER_S,
    StreamView,
    TimeRange,
    ViewRequest,
    ns_to_seconds,
    seconds_to_ns,
)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class PlaybackController:
    def __init__(
        self,
        t_min_ns: int,
        t_max_ns: int,
        *,
        initial_time_ns: int | None = None,
        view_span_ns: int = 10 * NS_PER_S,
        playback_speed: float = 1.0,
        hard_jump_ratio: float = 0.5,
    ):
        if t_max_ns <= t_min_ns:
            raise ValueError("t_max_ns must be > t_min_ns")

        self.t_min_ns = int(t_min_ns)
        self.t_max_ns = int(t_max_ns)
        self.cursor_ns = _clamp(
            int(initial_time_ns if initial_time_ns is not None else t_min_ns),
            self.t_min_ns,
            self.t_max_ns,
        )
        self.view_span_ns = _clamp(int(view_span_ns), 1, self.t_max_ns - self.t_min_ns)
        self.playback_speed = float(playback_speed)
        self.hard_jump_ratio = max(0.0, float(hard_jump_ratio))
        self.is_playing = False

        self._jumped = False
        self._last_tick_s: float | None = None

    # -- seconds-facing accessors (UI convenience) --------------------------

    @property
    def t_min_s(self) -> float:
        return ns_to_seconds(self.t_min_ns)

    @property
    def t_max_s(self) -> float:
        return ns_to_seconds(self.t_max_ns)

    @property
    def cursor_s(self) -> float:
        return ns_to_seconds(self.cursor_ns)

    @property
    def view_span_s(self) -> float:
        return ns_to_seconds(self.view_span_ns)

    # -- visible window -----------------------------------------------------

    @property
    def visible_range(self) -> TimeRange:
        half = self.view_span_ns // 2
        start = self.cursor_ns - half
        stop = start + self.view_span_ns

        if start < self.t_min_ns:
            start = self.t_min_ns
            stop = min(self.t_max_ns, start + self.view_span_ns)
        if stop > self.t_max_ns:
            stop = self.t_max_ns
            start = max(self.t_min_ns, stop - self.view_span_ns)
        return TimeRange(int(start), int(stop))

    # -- mutation -----------------------------------------------------------

    def jump_to(self, timestamp_ns: int) -> None:
        target = _clamp(int(timestamp_ns), self.t_min_ns, self.t_max_ns)
        if abs(target - self.cursor_ns) > int(self.view_span_ns * self.hard_jump_ratio):
            self._jumped = True
        self.cursor_ns = target

    def jump_by(self, delta_ns: int) -> None:
        self.jump_to(self.cursor_ns + int(delta_ns))

    def set_view_span(self, span_ns: int) -> None:
        self.view_span_ns = _clamp(int(span_ns), 1, self.t_max_ns - self.t_min_ns)

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    # -- per-frame loop -----------------------------------------------------

    def tick(self, monotonic_now_s: float) -> int:
        """Advance the cursor based on wall-clock delta. Returns new cursor_ns."""

        now = float(monotonic_now_s)
        if self._last_tick_s is not None and self.is_playing:
            delta_ns = seconds_to_ns((now - self._last_tick_s) * self.playback_speed)
            self.cursor_ns = _clamp(
                self.cursor_ns + delta_ns, self.t_min_ns, self.t_max_ns
            )
            if self.cursor_ns in (self.t_min_ns, self.t_max_ns):
                self.is_playing = False
        self._last_tick_s = now
        return self.cursor_ns

    def viewport(
        self,
        *,
        width_px: int,
        streams: tuple[StreamView, ...] = (),
    ) -> ViewRequest:
        request = ViewRequest(
            time=self.visible_range,
            width_px=int(width_px),
            streams=tuple(streams),
            cursor_ns=self.cursor_ns,
            direction=1 if self.playback_speed >= 0 else -1,
            playing=self.is_playing,
            jumped=self._jumped,
        )
        self._jumped = False
        return request
