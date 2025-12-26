from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    SupportsIndex,
    TypeVar,
    overload,
)
import pandas as pd

from starfile_rs.components import LoopDataBlock
from starfile_rs.schema._fields import LoopField, Field
from starfile_rs.schema._models import (
    SingleDataModel,
    LoopDataModel as LoopDataModelBase,
    StarModel,
)
from starfile_rs.schema._series import SeriesBase

__all__ = ["StarModel", "Field", "SingleDataModel", "LoopDataModel", "Series"]

_T = TypeVar("_T")

if TYPE_CHECKING:

    class PandasSeries(pd.Series, Generic[_T]):
        @overload
        def __getitem__(self, key: SupportsIndex) -> _T: ...
        @overload
        def __getitem__(self, key: Any) -> pd.Series[_T]: ...

        def __iter__(self) -> Iterator[_T]: ...
        def __next__(self) -> _T: ...


class Series(SeriesBase[_T]):
    def __get__(self, instance: Any | None, owner) -> PandasSeries[_T]:
        return self


class LoopDataModel(LoopDataModelBase[pd.DataFrame]):
    _series_class = pd.Series
    _dataframe_class = pd.DataFrame

    @classmethod
    def _get_dataframe(
        cls, block: LoopDataBlock, fields: list[LoopField]
    ) -> pd.DataFrame:
        dtype = {f.column_name: _arg_to_dtype(f._get_annotation_arg()) for f in fields}
        for f in fields:
            if f.column_name not in block.columns:
                if f._default is f._empty:
                    raise ValueError(
                        f"Required column '{f.column_name}' not found in data block "
                        f"'{block.name}'."
                    )
                else:
                    dtype.pop(f.column_name)

        names = list(dtype.keys())
        usecols = [block.columns.index(name) for name in names]
        return block.trust_loop()._to_pandas_impl(
            usecols=usecols, names=names, dtype=dtype
        )


def _arg_to_dtype(arg):
    if arg is str:
        return "string"
    return arg
