from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args, get_origin, overload

from starfile_rs import DataBlock

if TYPE_CHECKING:
    from typing import Self
    from starfile_rs.schema._models import (
        BaseBlockModel,
        SingleDataModel,
        LoopDataModel,
        StarModel,
    )
    from starfile_rs.schema._series import SeriesBase


class Field:
    """Descriptor for star file schema fields.

    This object will automatically be converted to BlockField, SingleField, or LoopField
    when used in schema subclasses.
    """

    def __init__(self, name: str | None = None):
        self._star_name = name
        self._field_name: str | None = None
        self._annotation: Any | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.attribute_name!r}, annotation={self._annotation!r})"

    def normalize_value(self, value: Any) -> Any:
        return self._validate_value(self._annotation, value)

    def _validate_value(self, annotation: Any, value: Any) -> Any:
        raise ValueError(f"Field {self.attribute_name!r} has no validation implemented")

    @classmethod
    def _from_field(
        cls,
        field: Field,
        annotation,
    ) -> Self:
        self = cls(field._star_name)
        if annotation is None:
            raise ValueError(
                f"Field {field.attribute_name!r} requires a type annotation"
            )
        self._annotation = annotation
        self._field_name = field._field_name
        return self

    @property
    def attribute_name(self) -> str:
        return self._field_name

    @property
    def annotation(self) -> Any:
        return self._annotation

    def __set_name__(self, owner: type[LoopDataModel], name: str) -> None:
        self._field_name = name
        if self._star_name is None:
            self._star_name = name


class BlockField(Field):
    @overload
    def __get__(
        self,
        instance: Literal[None],
        owner: type[StarModel],
    ) -> BlockField: ...
    @overload
    def __get__(
        self,
        instance: StarModel,
        owner: type[StarModel] | None = None,
    ) -> BaseBlockModel: ...

    def __get__(
        self,
        instance: StarModel | None,
        owner: type[StarModel] | None = None,
    ) -> BlockField | BaseBlockModel:
        if instance is None:
            return self
        model = instance._block_models[self.block_name]
        return model

    @property
    def block_name(self) -> str:
        """The actual block name in the star file."""
        if self._star_name is None:
            raise ValueError("Field name is not set")
        return self._star_name

    def _validate_value(self, annotation: type[BaseBlockModel], value: DataBlock):
        return annotation.validate_block(value)


class _BlockComponentField(Field):
    @property
    def column_name(self) -> str:
        """The actual column name in the data block of the star file."""
        if self._star_name is None:
            raise ValueError("Field name is not set")
        return self._star_name


class LoopField(_BlockComponentField):
    def _validate_value(self, annotation: SeriesBase[_T], value: Any) -> SeriesBase[_T]:
        return value

    @overload
    def __get__(
        self,
        instance: Literal[None],
        owner: type[LoopDataModel],
    ) -> LoopField: ...
    @overload
    def __get__(
        self,
        instance: LoopDataModel,
        owner: type[LoopDataModel],
    ) -> Any: ...

    def __get__(self, instance: LoopDataModel | None, owner):
        if owner is None:
            return self
        return self._get_from_model(instance)

    def _get_from_model(self, instance: LoopDataModel) -> SeriesBase[_T]:
        return instance.dataframe[self.column_name]

    def _get_annotation_arg(self) -> type[_T]:
        _, arg = split_series_annotation(self._annotation)
        return arg

    def _get_annotation_arg_name(self) -> str:
        arg = self._get_annotation_arg()
        return getattr(arg, "__name__", str(arg))


class SingleField(_BlockComponentField):
    def _validate_value(self, annotation: Any, value: Any) -> Any:
        return annotation(value)

    @overload
    def __get__(
        self,
        instance: Literal[None],
        owner: type[SingleDataModel],
    ) -> SingleField: ...
    @overload
    def __get__(
        self,
        instance: SingleDataModel,
        owner: type[SingleDataModel] | None = None,
    ) -> Any: ...

    def __get__(
        self,
        instance: SingleDataModel | None,
        owner: type[SingleDataModel] | None = None,
    ):
        if instance is None:
            return self
        return self.normalize_value(instance._block.trust_single()[self.column_name])


_T = TypeVar("_T")


def split_series_annotation(
    annotation: type[SeriesBase[_T]],
) -> tuple[type[SeriesBase], type[_T]]:
    origin = get_origin(annotation)
    if origin is None:
        raise TypeError(f"Expected Series[T] annotation, got {annotation}")
    args = get_args(annotation)
    if len(args) != 1:
        raise TypeError(f"Expected Series[T] with one type argument, got {annotation}")
    return origin, args[0]
