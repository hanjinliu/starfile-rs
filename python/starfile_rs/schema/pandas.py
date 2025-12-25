from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, SupportsIndex, TypeVar, overload
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

    class pd_Series(pd.Series, Generic[_T]):
        @overload
        def __getitem__(self, key: SupportsIndex) -> _T: ...
        @overload
        def __getitem__(self, key: slice) -> pd.Series[_T]: ...


class Series(SeriesBase[_T]):
    def __get__(self, instance: Any | None, owner) -> pd_Series[_T]:
        return self


class LoopDataModel(LoopDataModelBase[pd.DataFrame]):
    _series_class = pd.Series
    _dataframe_class = pd.DataFrame

    @classmethod
    def _get_dataframe(
        cls, block: LoopDataBlock, fields: list[LoopField]
    ) -> pd.DataFrame:
        dtype = {f.column_name: _arg_to_dtype(f._get_annotation_arg()) for f in fields}
        names = list(dtype.keys())
        usecols = [block.columns.index(name) for name in names]
        return block.trust_loop()._to_pandas_impl(
            usecols=usecols, names=names, dtype=dtype
        )


def _arg_to_dtype(arg):
    if arg is str:
        return "string"
    return arg
