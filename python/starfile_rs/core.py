from typing import Iterable, Iterator, Mapping, TYPE_CHECKING
from starfile_rs.io import StarReader
from starfile_rs.components import DataBlock

if TYPE_CHECKING:
    import os


def read_star(path: "os.PathLike") -> "StarDict":
    """Read a STAR file and return its contents as a StarFileData object."""
    return StarDict.from_star(path)


def read_star_block(path: "os.PathLike", block_name: str) -> "DataBlock":
    """Read a specific data block from a STAR file."""
    with StarReader(path) as reader:
        for block in reader.iter_blocks():
            if block.name == block_name:
                return block
    raise KeyError(f"Data block with name {block_name!r} not found in file {path!r}.")


class StarDict(Mapping[str, "DataBlock"]):
    """A `dict`-like object representing the contents of a STAR file."""

    def __init__(self, blocks: dict[str, "DataBlock"], names: list[str]) -> None:
        self._blocks = blocks
        self._names = names

    @classmethod
    def from_star(cls, path: str) -> "StarDict":
        """Construct a StarDict from a STAR file."""
        with StarReader(path) as reader:
            blocks = cls.from_blocks(reader.iter_blocks())
        return cls(blocks, list(blocks.keys()))

    @classmethod
    def from_blocks(cls, blocks: Iterable["DataBlock"]) -> "StarDict":
        """Construct a StarDict from a list of DataBlock objects."""
        block_dict = {block.name: block for block in blocks}
        names = [block.name for block in blocks]
        return cls(block_dict, names)

    def nth(self, index: int) -> "DataBlock":
        """Return the n-th data block in the STAR file."""
        return self[self._names[index]]

    def first(self) -> "DataBlock":
        """Return the first data block in the STAR file."""
        return self.nth(0)

    def try_nth(self, index: int) -> "DataBlock | None":
        """Try to return the n-th data block in the STAR file, return None if out of range."""
        try:
            name = self._names[index]
        except IndexError:
            return None
        return self[name]

    def try_first(self) -> "DataBlock | None":
        """Try to return the first data block in the STAR file, return None if empty."""
        return self.try_nth(0)

    def __getitem__(self, key: str) -> "DataBlock":
        return self._blocks[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        d = {}
        for name, block in self._blocks.items():
            d[name] = block.__class__.__name__
        return f"<{self.__class__.__name__} of blocks={d!r}>"
