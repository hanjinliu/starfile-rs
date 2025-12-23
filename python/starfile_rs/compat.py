"""Compatibility layer for `starfile`."""

import os
from typing import Any
from starfile_rs.io import StarReader
from starfile_rs.core import as_star

__all__ = ["read", "write"]


def read(
    filename: os.PathLike,
    read_n_blocks: int | None = None,
    always_dict: bool = False,
    parse_as_string: list[str] = [],
):
    """Read a STAR file and return its contents as a StarDict object."""
    out = {}
    for ith, block in enumerate(StarReader.from_filepath(filename).iter_blocks()):
        if single := block.try_single():
            out[block.name] = single.to_dict(string_columns=parse_as_string)
        else:
            out[block.name] = block.to_pandas(string_columns=parse_as_string)
        if read_n_blocks is not None and ith + 1 >= read_n_blocks:
            break
    if len(out) == 1 and not always_dict:
        return next(iter(out.values()))
    return out


def write(
    star_dict: Any,
    filename: os.PathLike,
):
    """Write a STAR file from a StarDict-like object."""

    as_star(star_dict).write(filename)
