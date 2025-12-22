from pathlib import Path
from typing import Iterator
from starfile_rs import _starfile_rs_rust as _rs
from starfile_rs.components import DataBlock, LoopDataBlock, SingleDataBlock


class StarReader:
    def __init__(self, path: str | Path) -> None:
        self._rust_obj = _rs.StarReader(str(path))

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

    def get_block(self, name: str) -> "DataBlock | None":
        with self:
            for block in self.iter_blocks():
                if block.name == name:
                    return block
        return None
