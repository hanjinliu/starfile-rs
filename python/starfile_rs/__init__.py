from starfile_rs.core import (
    read_star,
    read_star_block,
    empty_star,
    as_star,
    read_star_text,
)
from starfile_rs.components import SingleDataBlock, LoopDataBlock, DataBlock

__all__ = [
    "read_star",
    "read_star_text",
    "read_star_block",
    "as_star",
    "empty_star",
    "DataBlock",
    "SingleDataBlock",
    "LoopDataBlock",
]

try:
    from starfile_rs._starfile_rs_rust import __version__
except Exception:
    __version__ = "unknown"
