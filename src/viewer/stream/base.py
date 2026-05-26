from __future__ import annotations

from typing import Any, Protocol

from viewer.span import Span

class BaseStream:
    transforms = None

    @property
    def span(self) -> Span:
        return Span(self.t_min, self.t_max)

    def chunks_in(self, span: Span) -> range:
        start = self.at(span.t0)
        stop = self.at(span.t1)
        return range(start, stop + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, key: int) -> dict:
        if key < 0:
            key += self.n_chunks
        if key < 0 or key >= self.n_chunks:
            raise IndexError(key)
        return self.read(key)


class Stream(Protocol):
    name: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int
    transforms: Any | None

    def at(self, t: float) -> int: ...
    def chunks_in(self, span: Span) -> range: ...
    def read(self, chunk_idx: int) -> dict: ...
    def __getitem__(self, key: int) -> dict: ...
