from typing import Iterable, Iterator, Mapping, SupportsIndex, TYPE_CHECKING
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

    def first(self) -> "DataBlock":
        """Return the first data block in the STAR file."""
        if first := self.try_first():
            return first
        raise ValueError("The STAR file is empty.")

    def try_first(self) -> "DataBlock | None":
        """Try to return the first data block in the STAR file, return None if empty."""
        if self._blocks:
            return self[0]
        else:
            return None

    def __getitem__(self, key: str | SupportsIndex) -> "DataBlock":
        if hasattr(key, "__index__"):
            key = self._names[key]
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
