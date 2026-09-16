from __future__ import annotations

from itertools import zip_longest
from typing import Any, Iterable, Iterator


def strict_zip(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("strict_zip arguments have different lengths")
        yield values
