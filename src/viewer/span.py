from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    t0: float
    t1: float

    @property
    def dt(self) -> float:
        return self.t1 - self.t0

    def pad(self, seconds: float) -> "Span":
        return Span(self.t0 - seconds, self.t1 + seconds)

    def contains(self, t: float) -> bool:
        return self.t0 <= t <= self.t1
