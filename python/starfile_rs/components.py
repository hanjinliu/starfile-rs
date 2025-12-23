import csv
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, TYPE_CHECKING, Literal, Mapping
from starfile_rs import _starfile_rs_rust as _rs

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    from typing import Self


class DataBlock(ABC):
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

    @columns.setter
    def columns(self, names: list[str]) -> None:
        """Set the column names of the data block."""
        return self._rust_obj.set_column_names(names)

    def _ipython_key_completions_(self) -> list[str]:
        return self.columns

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

    def try_loop(self, allow_conversion: bool = True) -> "LoopDataBlock | None":
        """Convert to a loop data block.

        This conversion is always safe, as single data blocks can always be represented
        as loop data blocks with one row.
        """
        if isinstance(self, LoopDataBlock):
            return self
        elif isinstance(self, SingleDataBlock) and allow_conversion:
            return LoopDataBlock(self._rust_obj.as_loop())
        else:
            return None

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

    def to_numpy(
        self,
        structure_by: Literal[None, "pandas", "polars"] = None,
    ) -> "np.ndarray":
        """Convert the data block to a numpy ndarray."""
        return self.trust_loop(True).to_numpy(structure_by=structure_by)

    @abstractmethod
    def clone(self) -> "Self":
        """Create a clone of the DataBlock."""

    @abstractmethod
    def to_string(self) -> str:
        """Convert the data block to a string."""


class SingleDataBlock(DataBlock, Mapping[str, Any]):
    def __getitem__(self, key: str) -> str:
        """Get the value of a single data item by its key."""
        for k, value_str in self._rust_obj.single_to_list():
            if k == key:
                return _parse_python_scalar(value_str)
        raise KeyError(key)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} of name={self.name!r}, items={self.to_dict()!r}>"

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the single data block."""
        return iter(self.columns)

    def __len__(self) -> int:
        """Return the number of items in the single data block."""
        return len(self.columns)

    @classmethod
    def from_iterable(
        cls,
        name: str,
        data: dict[str, Any] | Iterable[tuple[str, Any]],
    ) -> "SingleDataBlock":
        """Create a SingleDataBlock from a dict-like python objects."""
        if isinstance(data, Mapping):
            it = data.items()
        else:
            it = data
        str_data = [(k, str(v)) for k, v in it]
        rust_block = _rs.DataBlock.construct_single_block(
            name=name,
            scalars=str_data,
        )
        return cls(rust_block)

    def to_dict(
        self,
        string_columns: list[str] = [],
    ) -> dict[str, Any]:
        """Convert single data block to a dictionary of python objects."""
        return {
            k: _parse_python_scalar(v) if k not in string_columns else v
            for k, v in self._rust_obj.single_to_list()
        }

    def to_list(self, string_columns: list[str] = []) -> list[tuple[str, Any]]:
        """Convert single data block to a list of key-value pairs."""
        return [
            (k, _parse_python_scalar(v) if k not in string_columns else v)
            for k, v in self._rust_obj.single_to_list()
        ]

    def clone(self) -> "SingleDataBlock":
        """Create a clone of the SingleDataBlock."""
        new_block_rs = _rs.DataBlock.construct_single_block(
            name=self.name,
            scalars=list(self._rust_obj.single_to_list()),
        )
        return SingleDataBlock(new_block_rs)

    def to_string(self) -> str:
        """Convert the single data block to a string."""
        return "\n".join(
            f"_{n} {_python_obj_to_str(v)}" for n, v in self._rust_obj.single_to_list()
        )


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

    def to_numpy(
        self,
        structure_by: Literal[None, "pandas", "polars"] = None,
    ) -> "np.ndarray":
        """Convert the data block to a numpy ndarray.

        If `structure_by` is given, a structured array will be created using the
        specified library to determine the data types of each column. Otherwise, a
        numeric array will be created by loading the data with `numpy.loadtxt()`.
        """
        import numpy as np

        if structure_by is not None:
            if structure_by == "pandas":
                df = self.to_pandas()
                # make structured array
                arr = np.empty(
                    len(df), dtype=[(col, df[col].dtype.type) for col in df.columns]
                )
                for col in df.columns:
                    arr[col] = df[col].to_numpy()
            elif structure_by == "polars":
                arr = self.to_polars().to_numpy(structured=True)
            else:
                raise ValueError(
                    "structure_by must be one of None, 'pandas', or 'polars'."
                )
        else:
            sep = " "
            buf = self._as_buf(sep)
            arr = np.loadtxt(buf, delimiter=sep, ndmin=2, quotechar='"')
        return arr

    @classmethod
    def from_pandas(
        cls,
        name: str,
        df: "pd.DataFrame",
        *,
        separator: str = "\t",
        float_precision: int = 6,
    ) -> "LoopDataBlock":
        """Create a LoopDataBlock from a pandas DataFrame."""
        # pandas to_csv does not quote empty string. This causes incorrect parsing
        # when reading output star files by RELION.
        df_replaced = df.where(df != "", '""')
        out = df_replaced.to_csv(
            sep=separator,
            header=False,
            index=False,
            na_rep="<NA>",
            quoting=csv.QUOTE_NONE,
            quotechar='"',
            float_format=f"%.{float_precision}g",
            lineterminator="\n",
        )

        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=df.columns.tolist(),
            content=out,
            nrows=len(df),
        )
        return cls(rust_block)

    @classmethod
    def from_polars(
        cls,
        name: str,
        df: "pl.DataFrame",
        *,
        separator: str = "\t",
        float_precision: int = 6,
    ) -> "LoopDataBlock":
        """Create a LoopDataBlock from a polars DataFrame."""
        out = df.write_csv(
            separator=separator,
            include_header=False,
            null_value="<NA>",
            float_precision=float_precision,
        )
        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=df.columns,
            content=out,
            nrows=len(df),
        )
        return cls(rust_block)

    @classmethod
    def from_numpy(
        cls,
        name: str,
        array: "np.ndarray",
        *,
        columns: list[str] | None = None,
        separator: str = "\t",
    ) -> "LoopDataBlock":
        """Create a LoopDataBlock from a numpy ndarray."""
        import numpy as np

        if array.ndim == 1 and array.dtype.names is not None:
            if columns is None:
                columns = list(array.dtype.names)
            nrows = array.shape[0]
        elif array.ndim == 2:
            nrows, ncols = array.shape
            if columns is None:
                columns = [f"column_{i}" for i in range(ncols)]
            elif len(columns) != ncols:
                raise ValueError(
                    "Length of columns must match number of columns in the array."
                )
        else:
            raise ValueError("Numpy array must be 2-dimensional.")

        buf = StringIO()
        np.savetxt(
            buf,
            array,
            fmt="%s",
            delimiter=separator,
        )
        buf.seek(0)
        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=columns,
            content=buf.read(),
            nrows=nrows,
        )
        return cls(rust_block)

    @classmethod
    def from_obj(
        cls,
        name: str,
        data: Any,
        separator: str = "\t",
    ) -> "LoopDataBlock":
        buf = StringIO()
        if isinstance(data, Mapping):
            columns = list(data.keys())
            it = zip(*data.values())
        else:
            columns = [f"column_{i}" for i in range(len(data[0]))]
            it = data
        nrows = 0
        w = csv.writer(buf, delimiter=separator)
        for row in it:
            w.writerow([_python_obj_to_str(val) for val in row])
            nrows += 1
        rust_block = _rs.DataBlock.construct_loop_block(
            name=name,
            columns=columns,
            content=buf.getvalue(),
            nrows=nrows,
        )
        return cls(rust_block)

    def clone(self) -> "LoopDataBlock":
        """Create a clone of the LoopDataBlock."""
        new_block_rs = _rs.DataBlock.construct_loop_block(
            name=self.name,
            content=self._rust_obj.loop_content(),
            columns=self.columns,
            nrows=len(self),
        )
        return LoopDataBlock(new_block_rs)

    def to_string(self, column_numbering: bool = True) -> str:
        """Convert the loop data block to a string."""
        if column_numbering:
            column_str = "\n".join(
                f"_{col} #{ith + 1}" for ith, col in enumerate(self.columns)
            )
        else:
            column_str = "\n".join(f"_{col}" for col in self.columns)
        content = self._rust_obj.loop_content()
        return f"loop_\n{column_str}\n{content}"

    def _as_buf(self, new_sep: str) -> StringIO:
        value = self._rust_obj.loop_content_with_sep(new_sep).replace("'", '"')
        return StringIO(value)


_NAN_STRINGS = ["nan", "NaN", "<NA>"]


def _python_obj_to_str(value: Any) -> str:
    """Convert a Python scalar to a string representation for STAR files."""
    if value == "":
        return '""'
    return str(value)
