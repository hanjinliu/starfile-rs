from importlib import import_module
from pathlib import Path
from io import TextIOBase
from typing import (
    Any,
    Iterable,
    Iterator,
    TYPE_CHECKING,
    Literal,
    Mapping,
    MutableMapping,
    overload,
)
from starfile_rs.io import StarReader
from starfile_rs.components import DataBlock, SingleDataBlock, LoopDataBlock

if TYPE_CHECKING:
    from typing import TypeGuard
    import os
    import numpy as np
    import pandas as pd
    import polars as pl


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


def empty_star() -> "StarDict":
    """Create an empty STAR file representation."""
    return StarDict({}, [])


@overload
def as_star(obj: Any) -> "StarDict": ...
@overload
def as_star(obj: Literal[None], **kwargs) -> "StarDict": ...


def as_star(obj=None, /, **kwargs) -> "StarDict":
    """Convenient function to convert an object to a StarDict.

    Allowed input types include:
    - `StarDict`: returns the same object
    - dict of scalars: creates a `StarDict` containing a single data block
    - a block-like object: creates a `StarDict` with a single loop block
    - dict of block-like data: most complete construction of `StarDict`
    - sequence of block-like data: creates a `StarDict` with numbered block names

    Each data block can also be provided as keyword arguments. If both a positional
    object and keyword arguments are provided, a `TypeError` is raised.

    For safer construction of `StarDict` objects, use `empty_star()` and set blocks
    explicitly with `with_single_block()` and `with_loop_block()`.

    Examples
    --------
    ```python
    import polars as pl
    import pandas as pd
    from starfile_rs import as_star

    as_star({"key1": 1, "key2": 2.0, "key3": "value"})
    as_star(pl.DataFrame({"col1": [1, 2], "col2": [3.0, 4.0}))
    as_star(
        optics={"mag": 84000, "cs": 2.7},
        particles=pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    )
    """
    if obj is None:
        if not kwargs:
            raise TypeError("Either an object or keyword arguments must be provided.")
        return as_star(kwargs)
    if kwargs:
        raise TypeError("Cannot provide both positional object and keyword arguments.")
    if isinstance(obj, StarDict):
        return obj
    out = empty_star()
    if isinstance(obj, Mapping):
        if any(_is_scalar(v) for v in obj.values()):
            out.with_single_block("", obj)
        else:
            for key, value in obj.items():
                if isinstance(value, Mapping):
                    out.with_single_block(key, value)
                else:
                    out.with_loop_block(key, value)
    elif isinstance(obj, (list, tuple)):
        return as_star({str(idx): df for idx, df in enumerate(obj)})
    else:
        out.with_loop_block("", obj)
    return out


class StarDict(MutableMapping[str, "DataBlock"]):
    """A `dict`-like object representing the contents of a STAR file."""

    def __init__(self, blocks: dict[str, "DataBlock"], names: list[str]) -> None:
        self._blocks = blocks
        self._names = names

    def _ipython_key_completions_(self) -> list[str]:
        return self._names

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

    def __setitem__(self, key, value) -> None:
        raise AttributeError(
            "Cannot set item to StarDict. Use `with_single_block()` or `with_block()` "
            "to explicitly create a new StarDict with the desired block type."
        )

    def __delitem__(self, key: str) -> None:
        self._names.remove(key)
        self._blocks.pop(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        d = {}
        for name, block in self._blocks.items():
            d[name] = block.__class__.__name__
        return f"<{self.__class__.__name__} of blocks={d!r}>"

    # mutable methods
    def with_block(
        self,
        block: "SingleDataBlock",
        inplace: bool = True,
    ) -> "StarDict":
        if not isinstance(block, DataBlock):
            raise TypeError("block must be an instance of DataBlock.")
        new_block = self._blocks | {block.name: block}
        new_names = list(new_block.keys())
        if inplace:
            self._blocks = new_block
            self._names = new_names
            return self
        else:
            return StarDict(new_block, new_names)

    def with_single_block(
        self,
        name: str,
        data: dict[str, Any] | Iterable[tuple[str, Any]],
        inplace: bool = True,
    ) -> "StarDict":
        """Set a single data block in the STAR file.

        Examples
        --------
        ```python
        from starfile_rs import read_star
        star = read_star("path/to/file.star")
        star_edited = star.with_single_block(
            "new_block",
            {"key1": 1, "key2": 2.0, "key3": "value"}
        )

        """
        block = SingleDataBlock.from_iterable(name, data)
        return self.with_block(block, inplace=inplace)

    def with_loop_block(
        self,
        name: str,
        data: "pd.DataFrame | pl.DataFrame | np.ndarray",
        inplace: bool = True,
    ) -> "StarDict":
        """Set a loop data block in the STAR file."""
        if _is_pandas_dataframe(data):
            block = LoopDataBlock.from_pandas(name, data)
        elif _is_polars_dataframe(data):
            block = LoopDataBlock.from_polars(name, data)
        elif _is_numpy_array(data):
            block = LoopDataBlock.from_numpy(name, data)
        else:
            block = LoopDataBlock.from_obj(name, data)
        return self.with_block(block, inplace=inplace)

    def write(self, file: str | Path | TextIOBase) -> None:
        """Serialize the STAR file contents to a string."""
        if isinstance(file, (str, Path)):
            with open(file, "w") as f:
                f.write(self.to_string())
        else:
            file.write(self.to_string())

    def to_string(self, comment: str | None = None) -> str:
        """Convert the STAR file contents to a string."""
        strings = []
        for name, block in self._blocks.items():
            strings.append(f"\ndata_{name}\n\n")
            strings.append(block.to_string())
            strings.append("\n")
        if comment:
            strings.insert(0, f"# {comment}\n")
        return "".join(strings)


def _is_pandas_dataframe(obj: Any) -> "TypeGuard[pd.DataFrame]":
    return _is_instance(obj, "pandas", "DataFrame")


def _is_polars_dataframe(obj: Any) -> "TypeGuard[pl.DataFrame]":
    return _is_instance(obj, "polars", "DataFrame")


def _is_numpy_array(obj: Any) -> "TypeGuard[np.ndarray]":
    return _is_instance(obj, "numpy", "ndarray")


def _is_instance(obj, mod: str, cls_name: str):
    if not isinstance(obj_mod := getattr(obj, "__module__", None), str):
        return False
    if obj_mod.split(".")[0] != mod:
        return False
    if obj.__class__.__name__ != cls_name:
        return False
    imported_mod = import_module(mod)
    cls = getattr(imported_mod, cls_name)
    return isinstance(obj, cls)


def _is_scalar(obj: Any) -> bool:
    return (
        isinstance(obj, (str, int, float, bool))
        or obj is None
        or hasattr(obj, "__index__")
        or hasattr(obj, "__float__")
    )
