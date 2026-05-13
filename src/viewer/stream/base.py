from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zarr import Array


class Stream(Protocol):
    name: str
    kind: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int

    values: Array
    ts: Array

    def chunk_at(self, t: float) -> int: ...
    def load_chunk(self, chunk_idx: int) -> dict: ...
    def view(self, chunks, t0: float, t1: float, width_px: float): ...
