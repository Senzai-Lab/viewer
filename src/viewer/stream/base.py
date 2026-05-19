from __future__ import annotations

from typing import Any, Protocol

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


class BaseStream:
    transforms = None

    @property
    def span(self) -> Span:
        return Span(self.t_min, self.t_max)

    def chunks_in(self, span: Span) -> range:
        return chunks_in_span(self, span)

    def at(self, t: float) -> dict:
        return self.read(self.chunk_at(t))

    def in_span(self, span: Span) -> list[dict]:
        return [self.read(i) for i in self.chunks_in(span)]

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return read_key(self, key)


class Stream(Protocol):
    name: str

    t_min: float
    t_max: float
    n_chunks: int
    chunk_nbytes: int
    transforms: Any | None

    def chunk_at(self, t: float) -> int: ...
    def chunks_in(self, span: Span) -> range: ...
    def read(self, chunk_idx: int) -> dict: ...
    def __getitem__(self, key: int | slice) -> dict | list[dict]: ...
