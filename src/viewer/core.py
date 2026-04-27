from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


NS_PER_S = 1_000_000_000
ChunkKey: TypeAlias = str


def seconds_to_ns(seconds: float) -> int:
    return int(round(float(seconds) * NS_PER_S))


def ns_to_seconds(timestamp_ns: int) -> float:
    return float(timestamp_ns) / NS_PER_S


@dataclass(frozen=True)
class TimeRange:
    """Half-open time span ``[start_ns, stop_ns)``."""

    start_ns: int
    stop_ns: int

    @property
    def duration_ns(self) -> int:
        return self.stop_ns - self.start_ns


@dataclass(frozen=True)
class Selection:
    """Per-stream selection. Adapters interpret only the fields they care about."""

    channels: tuple[int, ...] | None = None
    units: tuple[int, ...] | None = None
    joints: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StreamView:
    stream_id: str
    selection: Selection = field(default_factory=Selection)


@dataclass(frozen=True)
class ViewRequest:
    """A viewport request: what the renderer wants for the next frame."""

    time: TimeRange
    width_px: int
    streams: tuple[StreamView, ...] = ()
    cursor_ns: int = 0
    direction: int = 1
    playing: bool = False
    jumped: bool = False


@dataclass(frozen=True)
class LoadedChunk:
    """An adapter-owned chunk payload. ``data`` is opaque to the cache."""

    key: ChunkKey
    time: TimeRange
    data: Any = field(default=None, compare=False)
    nbytes: int = field(default=0, compare=False)


@dataclass(frozen=True)
class StreamFrame:
    """Per-stream slice of a Frame returned to the renderer."""

    stream: StreamView
    coverage: float
    data: Any = None

    @property
    def ready(self) -> bool:
        return self.coverage >= 1.0