from io import StringIO
from typing import Any, Iterator, TYPE_CHECKING, Mapping
from starfile_rs import _starfile_rs_rust as _rs

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


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
        """Try to convert to a loop data block, return None otherwise."""
        if isinstance(self, LoopDataBlock):
            return self
        elif isinstance(self, SingleDataBlock):
            return LoopDataBlock(self._rust_obj.as_loop())
        else:  # pragma: no cover
            raise RuntimeError("Unreachable code path.")


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

    def to_pandas(self) -> "pd.DataFrame":
        """Convert the single data block to a pandas DataFrame."""
        return self.as_loop().to_pandas()

    def to_polars(self) -> "pl.DataFrame":
        """Convert the single data block to a polars DataFrame."""
        return self.as_loop().to_polars()


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

        sep = " "
        return pl.read_csv(
            self._as_buf(sep),
            separator=sep,
            has_header=False,
            comment_prefix="#",
            null_values=_NAN_STRINGS,
            new_columns=self.columns,
        )

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

    @classmethod
    def empty(cls, name: str, columns: list[str] | None = None) -> "LoopDataBlock":
        """Create an empty LoopDataBlock with the given name and columns."""
        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=columns or [],
            content="",
            nrows=0,
        )
        return cls(rust_block)

    def _as_buf(self, new_sep: str | None = None) -> StringIO:
        if new_sep is not None:
            value = self._rust_obj.loop_content_with_sep(new_sep)
        else:
            value = self._rust_obj.loop_content()
        return StringIO(value)


_NAN_STRINGS = ["nan", "NaN", "<NA>"]
