from io import StringIO
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING, Mapping
from starfile_rs import _starfile_rs_rust as _rs

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

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

class StarFileData(Mapping[str, "DataBlock"]):
    def __init__(self, blocks: dict[str, "DataBlock"], names: list[str]) -> None:
        self._blocks = blocks
        self._names = names
    
    @classmethod
    def from_star(cls, path: str) -> "StarFileData":
        with StarReader(path) as reader:
            blocks = {block.name: block for block in reader.iter_blocks()}
        return cls(blocks, list(blocks.keys()))
    
    def first(self) -> "DataBlock":
        """Return the first data block in the STAR file."""
        return next(iter(self._blocks.values()))
    
    def block_ith(self, i: int) -> "DataBlock":
        """Return the i-th data block in the STAR file."""
        name = self._names[i]
        return self._blocks[name]
    
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

def read(path: str) -> StarFileData:
    """Read a STAR file and return its contents as a StarFileData object."""
    return StarFileData.from_star(path)

def read_block(path: str, block_name: str) -> "DataBlock":
    """Read a specific data block from a STAR file."""
    with StarReader(path) as reader:
        for block in reader.iter_blocks():
            if block.name == block_name:
                return block
    raise KeyError(f"Data block with name {block_name!r} not found in file {path!r}.")

class DataBlock:
    def __init__(self, obj: _rs.DataBlock, /) -> None:
        self._rust_obj = obj

    @property
    def name(self) -> str:
        """Name of the data block."""
        return self._rust_obj.name()
    
    @property
    def columns(self) -> list[str]:
        """Column names of the data block."""
        return self._rust_obj.column_names()
    
    def as_single(self) -> "SingleDataBlock":
        """Convert this data block to a single data block.
        
        Raises ValueError if conversion is not possible. To safely attempt conversion, 
        use `try_as_single()` instead.
        """
        if out := self.try_as_single():
            return out
        raise ValueError("Cannot convert to single data block.")
    
    def try_as_single(self) -> "SingleDataBlock | None":
        """Try to convert to a single data block, return None otherwise."""
        if isinstance(self, SingleDataBlock):
            return self
        elif isinstance(self, LoopDataBlock):
            if self._rust_obj.loop_nrows() != 1:
                return None
            return SingleDataBlock(self._rust_obj.as_single())
        else:
            return None
    
    def as_loop(self) -> "LoopDataBlock":
        """Convert this data block to a loop data block.
        
        Raises ValueError if conversion is not possible. To safely attempt conversion, 
        use `try_as_loop()` instead.
        """
        if out := self.try_as_loop():
            return out
        raise ValueError("Cannot convert to loop data block.")
    
    def try_as_loop(self) -> "LoopDataBlock | None":
        """Try to convert to a loop data block, return None otherwise."""
        if isinstance(self, LoopDataBlock):
            return self
        elif isinstance(self, SingleDataBlock):
            return LoopDataBlock(self._rust_obj.as_loop())
        else:
            return None

class SingleDataBlock(DataBlock, Mapping[str, Any]):
    def __getitem__(self, key: str) -> str:
        """Get the value of a single data item by its key."""
        value_str = self._rust_obj.single_to_dict()[key]
        return _parse_python_scalar(value_str)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} of name={self.name!r}, items={self.to_dict()!r}>"
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the single data block."""
        return iter(self._rust_obj.single_to_dict())
    
    def __len__(self) -> int:
        """Return the number of items in the single data block."""
        return len(self.columns)

    def to_dict(self) -> dict[str, Any]:
        """Convert single data block to a dictionary of strings."""
        dict_str = self._rust_obj.single_to_dict()
        return {k: _parse_python_scalar(v) for k, v in dict_str.items()}

def _parse_python_scalar(value: str) -> Any:
    """Parse a string value to a Python scalar."""
    if value in _NAN_STRINGS:
        return None
    try:
        if "." in value or "e" in value or "E" in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

class LoopDataBlock(DataBlock):
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} of name={self.name!r}, nrows={self._rust_obj.loop_nrows()}>"

    def __len__(self) -> int:
        """Return the number of rows in the loop data block."""
        return self._rust_obj.loop_nrows()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the loop data block as (nrows, ncolumns)."""
        return (len(self), len(self.columns))

    def to_pandas(self) -> "pd.DataFrame":
        """Convert the data block to a pandas DataFrame."""
        import pandas as pd

        return pd.read_csv(
            self._as_buf(),
            delimiter=r"\s+",
            names=self.columns,
            header=None,
            comment="#",
            keep_default_na=False,
            na_values=_NAN_STRINGS,
            engine="c",
        )
    
    def to_polars(self) -> "pl.DataFrame":
        import polars as pl
        
        # TODO: white space handling
    
    def iter_pandas_chunks(self, chunksize: int = 100) -> "Iterator[pd.DataFrame]":
        """Convert the data block to an iterator of pandas DataFrame chunks."""
        import pandas as pd

        yield from pd.read_csv(
            self._as_buf(),
            delimiter=r"\s+",
            names=self.columns,
            header=None,
            comment="#",
            keep_default_na=False,
            na_values=_NAN_STRINGS,
            engine="c",
            chunksize=chunksize,
        )
    
    @classmethod
    def from_pandas(cls, name: str, df: "pd.DataFrame") -> "LoopDataBlock":
        """Create a LoopDataBlock from a pandas DataFrame."""
        buf = StringIO()
        df.to_csv(
            buf,
            sep=" ",
            header=False,
            index=False,
            na_rep="<NA>",
            float_format="%.6g",
        )
        buf.seek(0)
        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=df.columns.tolist(),
            content=buf.read(),
            nrows=len(df),
        )
        return cls(rust_block)

    def _as_buf(self) -> StringIO:
        return StringIO(self._rust_obj.loop_content())

_NAN_STRINGS = ["nan", "NaN", "<NA>"]
