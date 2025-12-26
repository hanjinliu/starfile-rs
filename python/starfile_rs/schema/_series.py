from __future__ import annotations

from typing import Generic, TypeVar

_T = TypeVar("_T")


class SeriesBase(Generic[_T]):
    """Only used for type annotations."""
