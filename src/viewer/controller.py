import numpy as np
from .utils import TimeRange, ViewRequest

class PlaybackController:
    def __init__(
        self,
        t_min: float,
        t_max: float,
        *,
        initial_time: float | None = None,
        view_span: float = 10.0,
        playback_speed: float = 1.0,
        hard_jump_ratio: float = 0.5,
    ):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.cursor_t = self._clamp(self.t_min if initial_time is None else initial_time)
        self.view_span = float(np.clip(view_span, 1e-6, max(self.t_max - self.t_min, 1e-6)))
        self.playback_speed = float(playback_speed)
        self.hard_jump_ratio = max(0.0, float(hard_jump_ratio))
        self.is_playing = False
        self._jumped = False

    def _clamp(self, timestamp: float) -> float:
        return float(np.clip(timestamp, self.t_min, self.t_max))

    @property
    def visible_range(self) -> TimeRange:
        half = 0.5 * self.view_span
        start = self.cursor_t - half
        end = self.cursor_t + half

        if start < self.t_min:
            start = self.t_min
            end = min(self.t_max, start + self.view_span)
        if end > self.t_max:
            end = self.t_max
            start = max(self.t_min, end - self.view_span)

        return TimeRange(float(start), float(end))

    def jump_to(self, timestamp: float) -> None:
        next_time = self._clamp(timestamp)
        delta_t = abs(next_time - self.cursor_t)
        self.cursor_t = next_time
        self._jumped = delta_t > self.view_span * self.hard_jump_ratio

    def jump_by(self, delta_s: float) -> None:
        self.jump_to(self.cursor_t + float(delta_s))

    def set_view_span(self, span_s: float) -> None:
        max_span = max(self.t_max - self.t_min, 1e-6)
        self.view_span = float(np.clip(span_s, 1e-6, max_span))

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def tick(self, dt_s: float) -> None:
        if not self.is_playing:
            return

        next_time = self.cursor_t + float(dt_s) * self.playback_speed
        self.cursor_t = self._clamp(next_time)

        if self.cursor_t >= self.t_max or self.cursor_t <= self.t_min:
            self.is_playing = False

    def make_request(self) -> ViewRequest:
        request = ViewRequest(
            view=self.visible_range,
            cursor_t=self.cursor_t,
            direction=1 if self.playback_speed >= 0 else -1,
            jumped=self._jumped,
            playing=self.is_playing,
        )
        self._jumped = False
        return request