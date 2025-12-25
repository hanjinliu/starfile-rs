from __future__ import annotations

from typing import Any, Generic, TypeVar

_T = TypeVar("_T")


class SeriesBase(Generic[_T]):
    def __get__(self, instance: Any | None, owner):
        if owner is None:
            return self
        return instance
