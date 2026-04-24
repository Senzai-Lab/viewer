from dataclasses import dataclass, field
from typing import Any, TypeAlias


ChunkKey: TypeAlias = str


def split_chunk_key(key: ChunkKey) -> tuple[str, str]:
    stream_id, separator, chunk_suffix = key.rpartition("/")
    if not separator or not stream_id or not chunk_suffix:
        raise ValueError(f"invalid chunk key {key!r}")
    return stream_id, chunk_suffix

@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def span(self) -> float:
        return float(self.end - self.start)

@dataclass(frozen=True)
class ViewRequest:
    view: TimeRange
    cursor_t: float
    direction: int = 1
    jumped: bool = False
    playing: bool = False


@dataclass(frozen=True)
class ChunkConfig:
    target_bytes: int

    def __post_init__(self) -> None:
        if self.target_bytes <= 0:
            raise ValueError("target_bytes must be > 0")


@dataclass(frozen=True)
class LoadedChunk:
    key: ChunkKey
    stream_id: str
    t0: float
    t1: float
    nbytes: int = field(default=0, compare=False)
    data: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class CachedStreamState:
    visible_keys: list[ChunkKey]
    ready_chunks: list[LoadedChunk]

    @property
    def ready_keys(self) -> list[ChunkKey]:
        return [chunk.key for chunk in self.ready_chunks]

    @property
    def coverage(self) -> float:
        if not self.visible_keys:
            return 1.0
        return len(self.ready_chunks) / len(self.visible_keys)

    @property
    def complete(self) -> bool:
        return len(self.ready_chunks) == len(self.visible_keys)

    @property
    def missing_count(self) -> int:
        return len(self.visible_keys) - len(self.ready_chunks)

def format_duration(seconds: float) -> str:
    """Convert seconds to human-readable format."""
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.2f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"

def format_size(size_bytes: float) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"