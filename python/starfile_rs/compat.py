"""Compatibility layer for `starfile`."""

from functools import partial
from typing import (
    Any,
    TYPE_CHECKING,
    Callable,
    Generic,
    Literal,
    MutableMapping,
    TypeVar,
    overload,
)
from starfile_rs.io import StarReader
from starfile_rs.core import as_star, StarDict, LoopDataBlock, SingleDataBlock

if TYPE_CHECKING:
    import os
    import pandas as pd
    import polars as pl

__all__ = ["read", "write"]


@overload
def read(
    filename: "os.PathLike",
    read_n_blocks: int | None = None,
    always_dict: Literal[False] = False,
    parse_as_string: list[str] = [],
    *,
    df: Literal["pandas", "polars"] = "pandas",
) -> "CachedDict[pd.DataFrame] | pd.DataFrame | dict[str, Any]": ...
@overload
def read(
    filename: "os.PathLike",
    read_n_blocks: int | None = None,
    always_dict: Literal[False] = False,
    parse_as_string: list[str] = [],
    *,
    df: Literal["pandas", "polars"] = "polars",
) -> "CachedDict[pl.DataFrame] | pl.DataFrame | dict[str, Any]": ...
@overload
def read(
    filename: "os.PathLike",
    read_n_blocks: int | None = None,
    always_dict: Literal[True] = False,
    parse_as_string: list[str] = [],
    *,
    df: Literal["pandas", "polars"] = "pandas",
) -> "CachedDict[pd.DataFrame]": ...
@overload
def read(
    filename: "os.PathLike",
    read_n_blocks: int | None = None,
    always_dict: Literal[True] = False,
    parse_as_string: list[str] = [],
    *,
    df: Literal["pandas", "polars"] = "polars",
) -> "CachedDict[pl.DataFrame]": ...


def read(
    filename: "os.PathLike",
    read_n_blocks: int | None = None,
    always_dict: bool = False,
    parse_as_string: list[str] = [],
    *,
    df: Literal["pandas", "polars"] = "pandas",
):
    """Read data from a STAR file.

    Single data blocks are read as dictionaries. Loop blocks are read as dataframes.
    When multiple data blocks are present a dictionary of datablocks is
    returned. When a single datablock is present only the block is returned by default.
    To force returning a dectionary even when only one datablock is present set
    `always_dict=True`.

    The dictionary returned by this function is cached. DataFrame parsing occurs only
    when a block is accessed for the first time.

    Parameters
    ----------
    filename: PathLike
        File from which to read data.
    read_n_blocks: int, optional
        Limit reading the file to the first n data blocks.
    always_dict: bool
        Always return a dictionary, even when only a single data block is present.
    parse_as_string: list[str]
        A list of keys or column names which will not be coerced to numeric values.
    df: "pandas" or "polars", default "pandas"
        The dataframe library to use when parsing loop blocks.
    """
    # prepare parser function
    if df == "pandas":
        _parser = partial(_parse_pandas, string_columns=parse_as_string)
    elif df == "polars":
        _parser = partial(_parse_polars, string_columns=parse_as_string)
    else:
        raise ValueError(f"Unsupported df type: {df}")

    # prepare blocks
    blocks = []
    for ith, block in enumerate(StarReader.from_filepath(filename).iter_blocks()):
        blocks.append(block)
        if read_n_blocks is not None and ith + 1 >= read_n_blocks:
            break

    # construct StarDict
    star = StarDict.from_blocks(blocks)
    if len(star) == 1 and not always_dict:
        if isinstance(first_block := star.first(), LoopDataBlock):
            return _parser(first_block)
        elif isinstance(first_block, SingleDataBlock):
            return first_block.to_dict(string_columns=parse_as_string)
        else:  # pragma: no cover
            raise RuntimeError("Unreachable code path.")
    return CachedDict(star, _parser)


def write(
    star_dict: Any,
    filename: "os.PathLike",
):
    """Write a STAR file from a StarDict-like object."""

    if isinstance(star_dict, CachedDict):
        # this is faster because unused blocks are not re-serialized
        star = star_dict._star_dict
    else:
        star = as_star(star_dict)
    star.write(filename)


def _parse_pandas(block: LoopDataBlock, string_columns) -> "pd.DataFrame":
    return block.to_pandas(string_columns=string_columns)


def _parse_polars(block: LoopDataBlock, string_columns) -> "pl.DataFrame":
    return block.to_polars(string_columns=string_columns)


_DF = TypeVar("_DF", "pd.DataFrame", "pl.DataFrame")


class CachedDict(MutableMapping[str, "dict[str, Any] | _DF"], Generic[_DF]):
    """Dictionary-like wrapper around StarDict that caches parsed blocks."""

    def __init__(
        self,
        star_dict: StarDict,
        parser: Callable[[LoopDataBlock], _DF],
    ):
        self._star_dict = star_dict
        self._parser = parser
        self._cache: dict[str, "dict[str, Any] | _DF"] = {}

    def _ipython_key_completions_(self):
        return self._star_dict._ipython_key_completions_()

    def _repr_html_(self):
        return self._star_dict._repr_html_()

    def __repr__(self) -> str:
        d = self._star_dict._block_repr_dict()
        return f"<{self.__class__.__name__} of blocks={d!r}>"

    def __getitem__(self, key: str) -> "dict[str, Any] | _DF":
        if key not in self._cache:
            block = self._star_dict[key]
            if single := block.try_single():
                self._cache[key] = single.to_dict()
            else:
                self._cache[key] = self._parser(block)
        return self._cache[key]

    def __contains__(self, key):
        # MutableMapping tries to access __getitem__ first to check for membership,
        # which we want to avoid here.
        return key in self._star_dict

    def __setitem__(self, key: str, value: "dict[str, Any] | _DF") -> None:
        self._star_dict[key] = value
        self._cache[key] = value

    def __delitem__(self, key: str) -> None:
        del self._star_dict[key]
        if key in self._cache:
            del self._cache[key]

    def __iter__(self):
        return iter(self._star_dict)

    def __len__(self) -> int:
        return len(self._star_dict)
