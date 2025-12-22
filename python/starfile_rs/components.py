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

    def trust_single(self, allow_conversion: bool = True) -> "SingleDataBlock":
        """Convert this data block to a single data block.

        Raises ValueError if conversion is not possible. To safely attempt conversion,
        use `try_single()` instead.

        Parameters
        ----------
        allow_conversion : bool, default True
            If True, single-row loop data blocks will be converted to single data blocks
            by this method. Set this to False if you only want to accept actual single
            data blocks.
        """
        if out := self.try_single(allow_conversion):
            return out
        raise ValueError(f"Data block {self.name!r} is not a single data block.")

    def try_single(self, allow_conversion: bool = True) -> "SingleDataBlock | None":
        """Try to convert to a single data block, return None otherwise.

        Parameters
        ----------
        allow_conversion : bool, default True
            If True, single-row loop data blocks will be converted to single data blocks
            by this method. Set this to False if you only want to accept actual single
            data blocks.
        """
        if isinstance(self, SingleDataBlock):
            return self
        elif isinstance(self, LoopDataBlock):
            if self._rust_obj.loop_nrows() != 1 or not allow_conversion:
                return None
            return SingleDataBlock(self._rust_obj.as_single())
        else:
            return None

    def trust_loop(self, allow_conversion: bool = True) -> "LoopDataBlock":
        """Convert this data block to a loop data block.

        Raises ValueError if conversion is not possible. To safely attempt conversion,
        use `try_loop()` instead.

        Parameters
        ----------
        allow_conversion : bool, default True
            If True, single-row loop data blocks will be converted to single data blocks
            by this method. Set this to False if you only want to accept actual single
            data blocks.
        """
        if out := self.try_loop(allow_conversion):
            return out
        raise ValueError(f"Data block {self.name} is not a loop data block.")

    def try_loop(self, allow_conversion: bool = True) -> "LoopDataBlock":
        """Convert to a loop data block.

        This conversion is always safe, as single data blocks can always be represented
        as loop data blocks with one row.
        """
        if isinstance(self, LoopDataBlock):
            return self
        elif isinstance(self, SingleDataBlock) and allow_conversion:
            return LoopDataBlock(self._rust_obj.as_loop())
        else:
            raise ValueError(f"Data block {self.name!r} is not a loop data block.")

    def to_pandas(
        self,
        string_columns: list[str] = [],
    ) -> "pd.DataFrame":
        """Convert the data block to a pandas DataFrame."""
        return self.trust_loop(True).to_pandas(string_columns=string_columns)

    def to_polars(
        self,
        string_columns: list[str] = [],
    ) -> "pl.DataFrame":
        """Convert the data block to a polars DataFrame."""
        return self.trust_loop(True).to_polars(string_columns=string_columns)


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

    def to_dict(
        self,
        string_columns: list[str] = [],
    ) -> dict[str, Any]:
        """Convert single data block to a dictionary of python objects."""
        dict_str = self._rust_obj.single_to_dict()

        return {
            k: _parse_python_scalar(v) if k not in string_columns else v
            for k, v in dict_str.items()
        }


def _parse_python_scalar(value: str) -> Any:
    """Parse a string value to a Python scalar."""
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

    def to_pandas(
        self,
        string_columns: list[str] = [],
    ) -> "pd.DataFrame":
        """Convert the data block to a pandas DataFrame."""
        import pandas as pd

        if string_columns:
            dtype = {col: str for col in string_columns}
        else:
            dtype = None
        # NOTE: converting multiple whitespaces to a single space for pandas read_csv
        # performs better
        sep = " "
        df = pd.read_csv(
            self._as_buf(sep),
            dtype=dtype,
            delimiter=sep,
            names=self.columns,
            header=None,
            comment="#",
            keep_default_na=False,
            na_values=_NAN_STRINGS,
            engine="c",
        )
        return df

    def to_polars(
        self,
        string_columns: list[str] = [],
    ) -> "pl.DataFrame":
        """Convert the data block to a polars DataFrame."""
        import polars as pl

        if string_columns:
            schema_overrides = {col: pl.String for col in string_columns}
        else:
            schema_overrides = None
        sep = " "
        return pl.read_csv(
            self._as_buf(sep),
            separator=sep,
            has_header=False,
            comment_prefix="#",
            null_values=_NAN_STRINGS,
            new_columns=self.columns,
            schema_overrides=schema_overrides,
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

    def _as_buf(self, new_sep: str) -> StringIO:
        value = self._rust_obj.loop_content_with_sep(new_sep).replace("'", '"')
        return StringIO(value)


_NAN_STRINGS = ["nan", "NaN", "<NA>", ""]
