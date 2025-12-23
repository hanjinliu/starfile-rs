from pathlib import Path
from typing import Iterator
from starfile_rs import _starfile_rs_rust as _rs
from starfile_rs.components import DataBlock, LoopDataBlock, SingleDataBlock


class StarReader:
    def __init__(self, rust_obj: _rs.StarReader | _rs.StarTextReader) -> None:
        self._rust_obj = rust_obj

    @classmethod
    def from_filepath(cls, path: str | Path) -> "StarReader":
        return cls(_rs.StarReader(str(path)))

    @classmethod
    def from_text(cls, content: str) -> "StarReader":
        return cls(_rs.StarTextReader(content))

    def __enter__(self) -> "StarReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._rust_obj.close()

    def iter_blocks(self) -> "Iterator[DataBlock]":
        while True:
            block = self._rust_obj.next_block()
            if block is None:
                break
            if block.block_type().is_loop():
                yield LoopDataBlock(block)
            else:
                yield SingleDataBlock(block)
