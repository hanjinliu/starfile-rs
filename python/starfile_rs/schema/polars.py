from __future__ import annotations

from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, overload
import polars as pl

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

    class PolarsSeries(pl.Series, Generic[_T]):
        @overload
        def __getitem__(self, key: SupportsIndex) -> _T: ...
        @overload
        def __getitem__(self, key: slice) -> pl.Series[_T]: ...


class Series(SeriesBase[_T]):
    def __get__(self, instance, owner) -> PolarsSeries[_T]:
        return self


class LoopDataModel(LoopDataModelBase[pl.DataFrame]):
    _series_class = pl.Series
    _dataframe_class = pl.DataFrame

    @classmethod
    def _get_dataframe(
        cls, block: LoopDataBlock, fields: list[LoopField]
    ) -> pl.DataFrame:
        schema = {col: pl.String for col in block.columns}
        new_columns: list[int] = []
        for f in fields:
            schema[f.column_name] = _type_to_schema(f._get_annotation_arg())
            new_columns.append(f.column_name)
        columns = [block.columns.index(name) for name in new_columns]
        return block.trust_loop()._to_polars_impl(
            columns=columns,
            new_columns=new_columns,
            schema=schema,
            infer_schema=False,
        )


def _type_to_schema(typ):
    if typ is int:
        return pl.Int64
    elif typ is float:
        return pl.Float64
    elif typ is str:
        return pl.String
    elif typ is bool:
        return pl.Boolean
    else:
        return typ  # just trust polars to handle it
