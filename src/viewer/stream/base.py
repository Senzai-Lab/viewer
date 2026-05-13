from __future__ import annotations

from typing import Any, Protocol


class Stream(Protocol):
    name: str
    kind: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int

    values: Any
    ts: Any

    def chunk_at(self, t: float) -> int: ...
    def load_chunk(self, chunk_idx: int) -> dict: ...
    def iter_visible(self, chunks, t0: float, t1: float, width_px: float): ...
