from importlib import import_module
import csv
from io import BytesIO, StringIO
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, TYPE_CHECKING, Literal, Mapping
from starfile_rs import _starfile_rs_rust as _rs
from starfile_rs import _utils

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    from typing import Self, TypeGuard


class DataBlock(ABC):
    def __init__(self, obj: _rs.DataBlock, /) -> None:
        self._rust_obj = obj

    def __bool__(self) -> bool:
        """Always True to use try_single() and try_loop() in a if clause."""
        return True

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

    @staticmethod
    def _try_parse_single_and_then_loop(name: str, value) -> "DataBlock":
        try:
            block = SingleDataBlock._from_any(name, value)
        except Exception:
            block = LoopDataBlock._from_any(name, value)
        return block

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
        else:  # pragma: no cover
            raise RuntimeError("Unreachable code reached in DataBlock.try_single()")

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
        if (out := self.try_loop(allow_conversion)) is not None:
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

    def _repr_html_(self) -> str:
        body = self._rust_obj.to_html(cell_style="padding: 4px;", max_lines=200)
        title = f"<h4>{self.name}</h4>"
        return title + body

    @abstractmethod
    def clone(self) -> "Self":
        """Create a clone of the DataBlock."""

    @abstractmethod
    def to_string(self, *, block_title: bool = True) -> str:
        """Convert the data block to a string."""


class SingleDataBlock(DataBlock, Mapping[str, Any]):
    """A single data structure of a STAR file.

    Single data blocks appear as follows in STAR files:
    ```
    data_example

    _item1 value1
    _item2 value2
    ```

    This object stores the byte content of the single data block and provides methods to
    convert it to dict-like structures.
    """

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

    @classmethod
    def _from_any(
        cls,
        name: str,
        data: Any,
    ) -> "SingleDataBlock":
        if _is_pandas_dataframe(data):
            block = LoopDataBlock.from_pandas(
                name, data, quote_unsafe=False
            ).trust_single()
        elif _is_polars_dataframe(data):
            block = LoopDataBlock.from_polars(
                name, data, quote_unsafe=False
            ).trust_single()
        elif _is_numpy_array(data):
            block = LoopDataBlock.from_numpy(name, data).trust_single()
        else:
            block = SingleDataBlock.from_iterable(name, data)
        return block

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

    def to_string(
        self,
        *,
        block_title: bool = True,
    ) -> str:
        """Convert the single data block to a string.

        Parameters
        ----------
        block_title : bool, default True
            If True, include the 'data_XX' title at the beginning.
        """
        out = "\n".join(
            f"_{n} {_utils.python_obj_to_str(v)}"
            for n, v in self._rust_obj.single_to_list()
        )
        if block_title:
            return f"data_{self.name}\n\n{out}"
        return out


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
    """A loop data structure of a STAR file.

    Loop data blocks appear as follows in STAR files:

    ```
    data_example

    loop_
    _column1
    _column2
    xxx 0.0
    yyy 1.0
    ```

    This object stores the byte content of the loop data block and provides methods to
    convert it to DataFrame-like structures. Note that parsing to DataFrame is more
    costly than reading files into a data block.
    """

    def __init__(
        self,
        obj: _rs.DataBlock,
        /,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        super().__init__(obj)
        self._row_start = start
        self._row_end = obj.loop_nrows() if end is None else end

    def __len__(self) -> int:
        """Return the number of rows in the loop data block."""
        return self._row_end - self._row_start

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name!r} with {len(self)} rows and {len(self.columns)} columns>"

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the loop data block as (nrows, ncolumns)."""
        return (len(self), len(self.columns))

    def slice(self, offset: int, length: int) -> "LoopDataBlock":
        """Return a sliced view of the loop data block.

        This is a cheap operation that does not involve data copying. If you want to use
        a small part of a large loop data block, consider using this method before
        calling `to_pandas()`, `to_polars()`, or other conversion methods.

        Parameters
        ----------
        offset : int
            The starting row index of the slice.
        length : int
            The number of rows in the slice.
        """
        start = self._row_start + offset
        end = min(start + length, self._row_end)
        if start < 0 or end > self._row_end or start >= end:
            raise IndexError("Slice indices are out of range.")
        new_block = LoopDataBlock(self._rust_obj, start=start, end=end)
        return new_block

    def to_pandas(
        self,
        string_columns: list[str] = [],
    ) -> "pd.DataFrame":
        """Convert the data block to a pandas DataFrame."""
        if string_columns:
            dtype = {col: str for col in string_columns}
        else:
            dtype = None
        return self._to_pandas_impl(dtype=dtype, names=self.columns)

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
        return self._to_polars_impl(
            schema_overrides=schema_overrides, new_columns=self.columns
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
            if len(self) == 0:
                return np.empty((0, len(self.columns)))
            buf = self._as_buf(_SPACE)
            arr = np.loadtxt(buf, delimiter=_SPACE, ndmin=2, quotechar='"')
        return arr

    @classmethod
    def from_pandas(
        cls,
        name: str,
        df: "pd.DataFrame",
        *,
        separator: str = "\t",
        float_precision: int = 6,
        quote_unsafe: bool = False,
    ) -> "LoopDataBlock":
        """Create a LoopDataBlock from a pandas DataFrame."""
        # pandas to_csv does not quote empty string. This causes incorrect parsing
        # when reading output star files by RELION.
        if not quote_unsafe:
            new_columns: "list[pd.Series]" = []
            for column_name in df.columns:
                if (col := df[column_name]).dtype.kind not in "biuf":
                    col = col.astype(str)
                    cond = col.str.contains(_SPACE) | (col == "")
                    new_col = col.where(~cond, '"' + col + '"')
                    new_columns.append(new_col)
            if new_columns:
                df = df.copy()
                for col in new_columns:
                    df[col.name] = col
        out = df.to_csv(
            sep=separator,
            header=False,
            index=False,
            na_rep="<NA>",
            quoting=csv.QUOTE_NONE,
            quotechar='"',
            float_format=f"%.{float_precision}g",
            lineterminator="\n",
        )

        rust_block = _rs.DataBlock.construct_loop_block_from_bytes(
            name=name,
            columns=df.columns.tolist(),
            content=out.encode(),
            num_rows=len(df),
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
        quote_unsafe: bool = False,
    ) -> "LoopDataBlock":
        """Create a LoopDataBlock from a polars DataFrame."""
        import polars as pl

        if not quote_unsafe:
            expressions: "list[pl.Expr]" = []
            for column_name in df.columns:
                if df[column_name].dtype != pl.String:
                    continue
                _col = pl.col(column_name)
                cond = (
                    pl.when(_col.str.contains(_SPACE) | _col.eq(""))
                    .then('"' + _col + '"')
                    .otherwise(_col)
                )
                expressions.append(cond.alias(column_name))
            if expressions:
                df = df.with_columns(expressions)
        out = df.write_csv(
            separator=separator,
            include_header=False,
            null_value="<NA>",
            float_precision=float_precision,
            quote_style="never",
        )
        rust_block = _rs.DataBlock.construct_loop_block_from_bytes(
            name=name,
            columns=df.columns,
            content=out.encode(),
            num_rows=len(df),
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
        # TODO: check quoting
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
        rust_block = _rs.DataBlock.construct_loop_block_from_bytes(
            name=name,
            columns=columns,
            content=buf.read().encode(),
            num_rows=nrows,
        )
        return cls(rust_block)

    @classmethod
    def from_obj(
        cls,
        name: str,
        data: Any,
        separator: str = "\t",
    ) -> "LoopDataBlock":
        # TODO: check quoting
        buf = StringIO()
        if isinstance(data, Mapping):
            columns = list(data.keys())
            if all(_utils.is_scalar(v) for v in data.values()):
                data = {k: [v] for k, v in data.items()}
            else:
                lengths = set()
                for val in data.values():
                    if not _utils.is_sequence(val):
                        raise ValueError(
                            "All values in the mapping must be sequences of equal length."
                        )
                    lengths.add(len(val))
                if len(lengths) > 1:
                    raise ValueError(
                        "All values in the mapping must be sequences of equal length."
                    )
            it = zip(*data.values())
        else:
            columns = [f"column_{i}" for i in range(len(data[0]))]
            it = data
        nrows = 0
        w = csv.writer(buf, delimiter=separator)
        for row in it:
            w.writerow([_utils.python_obj_to_str(val) for val in row])
            nrows += 1
        rust_block = _rs.DataBlock.construct_loop_block_from_bytes(
            name=name,
            columns=columns,
            content=buf.getvalue().encode(),
            num_rows=nrows,
        )
        return cls(rust_block)

    @classmethod
    def _from_any(
        cls,
        name: str,
        data: Any,
        separator: str = "\t",
        float_precision: int = 6,
        quote_unsafe: bool = False,
    ) -> "LoopDataBlock":
        """An intelligent constructor from various data types."""
        if _is_pandas_dataframe(data):
            block = LoopDataBlock.from_pandas(
                name,
                data,
                quote_unsafe=quote_unsafe,
                separator=separator,
                float_precision=float_precision,
            )
        elif _is_polars_dataframe(data):
            block = LoopDataBlock.from_polars(
                name,
                data,
                quote_unsafe=quote_unsafe,
                separator=separator,
                float_precision=float_precision,
            )
        elif _is_numpy_array(data):
            block = LoopDataBlock.from_numpy(name, data)
        else:
            block = LoopDataBlock.from_obj(name, data)
        return block

    def clone(self) -> "LoopDataBlock":
        """Create a clone of the LoopDataBlock."""
        new_block_rs = _rs.DataBlock.construct_loop_block(
            name=self.name,
            columns=self.columns,
            content=self._rust_obj.loop_content(self._row_start, self._row_end),
            offsets=self._rust_obj.loop_offsets(),
        )
        return LoopDataBlock(new_block_rs)

    def to_string(
        self,
        *,
        column_numbering: bool = True,
        block_title: bool = True,
    ) -> str:
        """Convert the loop data block to a string.

        Parameters
        ----------
        column_numbering : bool, default True
            If True, include commented column numbering (e.g., _colname #1).
        block_title : bool, default True
            If True, include the 'data_XX' title at the beginning.
        """
        if column_numbering:
            column_str = "\n".join(
                f"_{col} #{ith + 1}" for ith, col in enumerate(self.columns)
            )
        else:
            column_str = "\n".join(f"_{col}" for col in self.columns)
        content = self._rust_obj.loop_content(self._row_start, self._row_end).decode()
        if block_title:
            return f"data_{self.name}\n\nloop_\n{column_str}\n{content}"
        return f"loop_\n{column_str}\n{content}"

    def _as_buf(
        self,
        new_sep: str,
    ) -> BytesIO:
        sep_u8 = new_sep.encode()[0]
        value = self._rust_obj.loop_content_with_sep(
            sep_u8,
            self._row_start,
            self._row_end,
        )
        return BytesIO(value)

    def _to_pandas_impl(self, **kwargs) -> "pd.DataFrame":
        """Convert the data block to a pandas DataFrame."""
        import pandas as pd

        # NOTE: converting multiple whitespaces to a single space for pandas read_csv
        # performs better
        return pd.read_csv(
            self._as_buf(_SPACE),
            delimiter=_SPACE,
            header=None,
            comment="#",
            keep_default_na=False,
            na_values=_NAN_STRINGS,
            engine="c",
            **kwargs,
        )

    def _to_polars_impl(self, **kwargs) -> "pl.DataFrame":
        """Convert the data block to a polars DataFrame."""
        import polars as pl

        # polars does not support reading empty data.
        if len(self) == 0:
            return pl.DataFrame(
                {col: pl.Series([], dtype=pl.Unknown) for col in self.columns}
            )

        return pl.read_csv(
            self._as_buf(_SPACE),
            separator=_SPACE,
            has_header=False,
            comment_prefix="#",
            null_values=_NAN_STRINGS,
            **kwargs,
        )


_NAN_STRINGS = ["nan", "NaN", "<NA>"]
_SPACE = " "


def _is_pandas_dataframe(obj: Any) -> "TypeGuard[pd.DataFrame]":
    return _is_instance(obj, "pandas", "DataFrame")


def _is_polars_dataframe(obj: Any) -> "TypeGuard[pl.DataFrame]":
    return _is_instance(obj, "polars", "DataFrame")


def _is_numpy_array(obj: Any) -> "TypeGuard[np.ndarray]":
    return _is_instance(obj, "numpy", "ndarray")


def _is_instance(obj, mod: str, cls_name: str):
    if not isinstance(obj_mod := getattr(type(obj), "__module__", None), str):
        return False
    if obj_mod.split(".")[0] != mod:
        return False
    if obj.__class__.__name__ != cls_name:
        return False
    imported_mod = import_module(mod)
    cls = getattr(imported_mod, cls_name)
    return isinstance(obj, cls)
