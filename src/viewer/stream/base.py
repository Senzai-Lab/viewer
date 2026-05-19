from __future__ import annotations

from typing import Protocol

from viewer.span import Span


def chunks_in_span(stream: Stream, span: Span) -> range:
    start = stream.chunk_at(span.t0)
    stop = stream.chunk_at(span.t1)
    return range(start, stop + 1)


def read_key(stream: Stream, key: int | slice) -> dict | list[dict]:
    if isinstance(key, slice):
        return [stream.read(i) for i in range(*key.indices(stream.n_chunks))]

    if key < 0:
        key += stream.n_chunks
    if key < 0 or key >= stream.n_chunks:
        raise IndexError(key)
    return stream.read(key)


class Stream(Protocol):
    name: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int

    def chunk_at(self, t: float) -> int: ...
    def chunks_in(self, span: Span) -> range: ...
    def read(self, chunk_idx: int) -> dict: ...
    def __getitem__(self, key: int | slice) -> dict | list[dict]: ...
