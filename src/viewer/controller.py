class TimeController:
    def __init__(
            self,
            t_min: float,
            t_max: float,
            span: float = 5.0,
            playback_speed: float = 1.0,
    ):
        self.t_cursor: float = t_min
        self.t_min = t_min
        self.t_max = t_max
        self.is_playing: bool = False
        self.playback_speed = playback_speed

        # viewport
        self.view_t0 = t_min
        self.view_t1 = t_min + span
        self.span = span
        self.forward = True

    def toggle(self):
        self.is_playing = not self.is_playing
        self._center_view()

    def jump_to(self, t: float):
        self.t_cursor = self._clamp(t)
        self._center_view()
    
    def jump_by(self, direction: float):
        self.jump_to(self.t_cursor + self.span * direction)

    def tick(self, dt: float) -> float:
        """Advance cursor by wall-clock delta"""
        if self.is_playing:
            dt *= self.playback_speed
            self.t_cursor += dt

            if self.t_cursor <= self.t_min or self.t_cursor >= self.t_max:
                self.is_playing = False
            else:
                self.view_t0 += dt
                self.view_t1 += dt
    
        return self.t_cursor

    def update_view(self, t0: float, t1: float):
        old_center = (self.view_t0 + self.view_t1) * 0.5
        old_span = self.view_t1 - self.view_t0

        new_center = (t0 + t1) * 0.5
        new_span = t1 - t0

        pan_delta = new_center - old_center
        panned = abs(pan_delta) > 1e-6
        zoomed = abs(new_span - old_span) > 1e-6

        if self.is_playing:
            self.forward = self.playback_speed >= 0
        elif panned:
            self.forward = pan_delta > 0
        
        if panned and not zoomed:
            self.t_cursor = self._clamp(new_center)

        self.view_t0 = t0
        self.view_t1 = t1
        self.span = new_span

    def _center_view(self):
        self.view_t0 = self.t_cursor - self.span * 0.5
        self.view_t1 = self.t_cursor + self.span * 0.5

    def _clamp(self, t: float) -> float:
        return max(self.t_min, min(self.t_max, t))
    
    def reset(self):
        self.is_playing = False
        self.span = 5.0
        self.playback_speed = 1.0
        self.forward = True
        self.jump_to(self.t_min)
