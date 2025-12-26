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

_empty = object()


class Field:
    """Descriptor for star file schema fields.

    This object will automatically be converted to BlockField, SingleField, or LoopField
    when used in schema subclasses.

    Parameters
    ----------
    name : str, optional
        The actual name of the field in the star file. If None, the attribute name will
        be used.
    frozen : bool, default False
        If True, the field cannot be modified after being set.
    """

    _empty = _empty

    def __init__(
        self,
        name: str | None = None,
        *,
        default: Any = _empty,
        frozen: bool = False,
    ):
        self._star_name = name
        self._field_name: str | None = None
        self._annotation: Any | None = None
        self._default = default
        self._frozen = frozen

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._field_name!r}, annotation={self.annotation!r})"

    def normalize_value(self, value: Any) -> Any:
        return self._validate_value(self._star_name, self.annotation, value)

    def _validate_value(self, name: str, annotation: Any, value: Any) -> Any:
        """Validate and possibly convert the value according to the annotation."""
        # this method should be overridden in subclasses. Note that Field itself should
        # not be an ABC, because it will be instantiated temporarily before being
        # converted to a BlockField, SingleField, or LoopField.
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _from_field(
        cls,
        field: Field,
        annotation,
    ) -> Self:
        self = cls(field._star_name, frozen=field._frozen, default=field._default)
        if annotation is None:
            raise ValueError(f"Field {field._field_name!r} requires a type annotation")
        self._annotation = annotation
        self._field_name = field._field_name
        return self

    @property
    def annotation(self) -> Any:
        return self._annotation

    def __set_name__(self, owner: type[LoopDataModel], name: str) -> None:
        if name not in owner.__annotations__:
            raise TypeError(
                f"Field '{name}' in {owner.__name__} must have a type annotation"
            )
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
        if self.block_name in instance._block_models:
            return instance._block_models[self.block_name]
        if self._default is not self._empty:
            return self._default
        raise AttributeError(
            f"Block '{self.block_name}' has not been set in instance of {owner.__name__}"
        )

    def __set__(
        self,
        instance: StarModel,
        value: DataBlock,
    ) -> None:
        if self._frozen:
            raise AttributeError(
                f"Field '{self._field_name}' is frozen and cannot be modified."
            )
        model = instance._block_models[self.block_name]
        block = model.validate_block(value)
        instance._block_models[self.block_name] = block

    @property
    def block_name(self) -> str:
        """The actual block name in the star file."""
        if self._star_name is None:
            raise ValueError("Field name is not set")
        return self._star_name

    def _validate_value(
        self, name: str, annotation: type[BaseBlockModel], value: DataBlock
    ):
        if not hasattr(annotation, "__starfile_fields__"):
            raise TypeError(
                "BlockField annotation must be a subclass of BaseBlockModel, "
                f"got {annotation}"
            )
        return annotation.validate_block(value)


class _BlockComponentField(Field):
    @property
    def column_name(self) -> str:
        """The actual column name in the data block of the star file."""
        if self._star_name is None:
            raise ValueError("Field name is not set")
        return self._star_name


class LoopField(_BlockComponentField):
    def _validate_value(
        self, name: str, annotation: SeriesBase[_T], value: Any
    ) -> SeriesBase[_T]:
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
        if self.column_name not in instance._block.columns:
            if self._default is not self._empty:
                return self._default
            raise AttributeError(
                f"Column '{self.column_name}' has not been set in instance of {owner.__name__}"
            )
        return instance.dataframe[self.column_name]

    def _get_annotation_arg(self) -> type[_T]:
        _, arg = split_series_annotation(self._annotation)
        return arg

    def _get_annotation_arg_name(self) -> str:
        arg = self._get_annotation_arg()
        return getattr(arg, "__name__", str(arg))


class SingleField(_BlockComponentField):
    def _validate_value(self, name: str, annotation: Any, value: Any) -> Any:
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
        block = instance._block.trust_single()
        if self.column_name not in block.columns:
            if self._default is not self._empty:
                return self._default
            raise AttributeError(
                f"Column '{self.column_name}' has not been set in instance of {owner.__name__}"
            )
        return self.normalize_value(block[self.column_name])


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
