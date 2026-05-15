from __future__ import annotations

from typing import Protocol


class Stream(Protocol):
    name: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int

    def chunk_at(self, t: float) -> int: ...
    def load_chunk(self, chunk_idx: int) -> dict: ...