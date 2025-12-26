from __future__ import annotations

import enum
from io import TextIOBase
from pathlib import Path
from types import MappingProxyType
from functools import cached_property
from typing import (
    Any,
    Generic,
    Literal,
    Mapping,
    TYPE_CHECKING,
    TypeVar,
    get_origin,
    get_type_hints,
)
from starfile_rs.components import DataBlock, LoopDataBlock, SingleDataBlock
from starfile_rs.core import iter_star_blocks, StarDict, read_star_text
from starfile_rs.schema._fields import (
    Field,
    BlockField,
    SingleField,
    LoopField,
    _BlockComponentField,
)
from starfile_rs.schema._exception import BlockValidationError, ValidationError

if TYPE_CHECKING:
    from typing import Self, dataclass_transform
else:

    def dataclass_transform(*args, **kwargs):
        return lambda c: c


class Extra(enum.Enum):
    ALLOW = "allow"
    FORBID = "forbid"
    IGNORE = "ignore"


ExtraType = Literal["allow", "forbid", "ignore"]
STARFILE_CONSTRUCT_KEY = "__starfile_construct__"


class _SchemaBase:
    __starfile_fields__: MappingProxyType[str, Field]

    def __repr__(self) -> str:
        field_reprs = []
        for name in self.__starfile_fields__.keys():
            value = getattr(self, name)
            field_reprs.append(f"{name}={value!r}")
        fields_str = ", ".join(field_reprs)
        return f"{type(self).__name__}({fields_str})"


@dataclass_transform(field_specifiers=(Field,))
class StarModel(_SchemaBase):
    """Base class for STAR file schema models."""

    __starfile_fields__: MappingProxyType[str, BlockField]
    __starfile_extra__: Extra
    _block_models: dict[str, BaseBlockModel]

    def __init__(self, /, **kwargs: Any):
        if STARFILE_CONSTRUCT_KEY in kwargs:
            self._block_models = kwargs[STARFILE_CONSTRUCT_KEY]
        else:
            name_map = {
                name: f.block_name for name, f in self.__starfile_fields__.items()
            }
            kwargs_renamed = {name_map.get(k, k): v for k, v in kwargs.items()}
            other = type(self).validate_dict(kwargs_renamed)
            self._block_models = other._block_models

    def __init_subclass__(cls, extra: ExtraType = "ignore"):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            if name in StarModel.__annotations__:
                continue
            if not issubclass(annot, BaseBlockModel):
                raise TypeError(
                    f"StarModel field '{name}' must be a subclass of SingleDataModel "
                    f"or LoopDataModel, got {annot}"
                )
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                new_field = BlockField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)

        cls.__starfile_fields__ = MappingProxyType(schema_fields)
        cls.__starfile_extra__ = Extra(extra)

    @classmethod
    def validate_dict(cls, star: Mapping[str, Any]) -> Self:
        """Validate a dict of DataBlocks against this StarModel schema."""
        missing: list[str] = []
        star_input: dict[str, BaseBlockModel] = {}
        annots = get_type_hints(cls)
        star = dict(star)
        star_keys = list(star.keys())
        for name, field in cls.__starfile_fields__.items():
            if (block_name := field.block_name) not in star:
                if field._default is field._empty:
                    missing.append(block_name)
                continue

            # field.normalize_value will eventually call BaseBlockModel.validate_block
            block = star.pop(block_name)
            annot = annots[name]
            if issubclass(annot, SingleDataModel):
                star_input[block_name] = field.normalize_value(block)
            elif issubclass(annot, LoopDataModel):
                star_input[block_name] = field.normalize_value(block)
            else:  # pragma: no cover
                # All the annotations should have been checked in __init_subclass__
                raise RuntimeError("Unreachable code reached.")
        if missing:
            _miss = ", ".join(repr(m) for m in missing)
            raise ValidationError(
                f"StarModel {cls.__name__} is missing required fields: "
                f"{_miss}. Incoming block names: {star_keys}"
            )
        if star:
            if cls.__starfile_extra__ is Extra.FORBID:
                unexpected = ", ".join(star.keys())
                raise ValidationError(
                    f"StarModel {cls.__name__} got unexpected blocks: {unexpected}"
                )
            elif cls.__starfile_extra__ is Extra.IGNORE:
                pass
            elif cls.__starfile_extra__ is Extra.ALLOW:
                for name, block in star.items():
                    star_input[name] = AnyBlock.validate_object(block)

        # Sort star_input by the order of input star. This is important to keep the
        # order of blocks when writing back to file.
        star_input_sorted = {}
        for k in star_keys:
            if k in star_input:
                # make sure key and block name match
                model = star_input[k]
                model._block._rust_obj.set_name(k)
                star_input_sorted[k] = model
        return cls(**{STARFILE_CONSTRUCT_KEY: star_input_sorted})

    @classmethod
    def validate_file(cls, path) -> Self:
        """Read a STAR file and validate it against this StarModel schema."""
        required = {f.block_name for f in cls.__starfile_fields__.values()}
        mapping = {}
        all_block_names: list[str] = []
        for block in iter_star_blocks(path):
            all_block_names.append(block.name)
            if (name := block.name) in required:
                mapping[name] = block
                required.remove(name)
            if not required and cls.__starfile_extra__ is Extra.IGNORE:
                break
        # Check for missing required fields here. Although validate_dict will also
        # do this, user can be notified with what blocks are actually in the file.
        for field in cls.__starfile_fields__.values():
            if field._default is not Field._empty:
                required.discard(field.block_name)
        if required:
            _miss = ", ".join(repr(m) for m in required)
            raise ValidationError(
                f"StarModel {cls.__name__} is missing required fields: "
                f"{_miss}. Incoming block names: {all_block_names}"
            )
        return cls.validate_dict(mapping)

    @classmethod
    def validate_text(cls, text: str) -> Self:
        """Read a STAR file string and validate it against this StarModel schema."""
        star_dict = read_star_text(text)
        return cls.validate_dict(star_dict)

    def write(self, path: str | Path | TextIOBase) -> None:
        """Write the StarModel to a STAR file."""
        return self.to_star_dict().write(path)

    def to_star_dict(self) -> StarDict:
        """Convert the StarModel to a StarDict."""
        return StarDict.from_blocks(
            model._block for model in self._block_models.values()
        )

    def to_string(self, comment: str | None = None) -> str:
        """Convert the StarModel to a STAR file string."""
        return self.to_star_dict().to_string(comment=comment)


class BaseBlockModel(_SchemaBase):
    __starfile_fields__: MappingProxyType[str, _BlockComponentField]
    _block: DataBlock

    def __init__(self, /, **kwargs: Any):
        if STARFILE_CONSTRUCT_KEY in kwargs:
            self._block = kwargs[STARFILE_CONSTRUCT_KEY]
        else:
            # Create an unnamed DataBlock from kwargs
            name_map = {
                name: f.column_name for name, f in self.__starfile_fields__.items()
            }
            kwargs_renamed = {name_map.get(k, k): v for k, v in kwargs.items()}
            self._block = type(self)._parse_block("", kwargs_renamed)

    @classmethod
    def validate_block(cls, value: Any) -> Self:
        if not isinstance(value, DataBlock):
            raise TypeError(f"Value {value!r} is not a DataBlock")
        block = cls._parse_block(value.name, value)
        fields = list(cls.__starfile_fields__.values())
        missing: list[str] = []
        for f in fields:
            if f.column_name not in block.columns and f._default is Field._empty:
                missing.append(f.column_name)
        if missing:
            # If this model has no attributes, validation error will not be raised here.
            raise BlockValidationError(
                f"Block {block.name!r} did not pass validation by {cls.__name__!r}: "
                f"missing columns: {missing}"
            )
        return cls(**{STARFILE_CONSTRUCT_KEY: block})

    @classmethod
    def _parse_block(cls, name: str, value: Any) -> DataBlock:
        if not isinstance(value, DataBlock):
            value = DataBlock._try_parse_single_and_then_loop(name, value)
        return value

    @classmethod
    def validate_object(cls, value: Any) -> Self:
        """Create an unnamed DataBlock from an object."""
        block = cls._parse_block("", value)
        return cls.validate_block(block)

    @classmethod
    def validate_file(cls, path) -> Self:
        it = iter_star_blocks(path)
        first_block = next(it)
        if next(it, None) is not None:
            raise BlockValidationError(f"File {path} contains multiple blocks.")
        return cls.validate_block(first_block)

    def to_string(self) -> str:
        """Convert the BlockModel to a STAR file string."""
        return self._block.to_string()


class AnyBlock(BaseBlockModel):
    """Class used for accepting any DataBlock without validation.

    Usually used for extra="allow" situations in StarModel.
    """

    __starfile_fields__ = MappingProxyType({})

    @property
    def block(self) -> DataBlock:
        """Return the underlying DataBlock."""
        return self._block


_DF = TypeVar("_DF")


@dataclass_transform(field_specifiers=(Field,))
class LoopDataModel(BaseBlockModel, Generic[_DF]):
    """Schema model for a loop data block."""

    __starfile_fields__: MappingProxyType[str, LoopField]
    _series_class: type
    _block: LoopDataBlock

    def __init_subclass__(cls):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                if get_origin(annot) is None:
                    raise TypeError(
                        f"LoopDataModel field '{name}' must be annotated with "
                        f"{cls._series_class.__name__}[T], got {annot}"
                    )
                new_field = LoopField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)

        cls.__starfile_fields__ = MappingProxyType(schema_fields)

    def __repr__(self) -> str:
        field_reprs = []
        for name, field in self.__starfile_fields__.items():
            annot = field._get_annotation_arg_name()
            series_repr = f"Series[{annot}]"
            field_reprs.append(f"{name}={series_repr}")
        nrows = self._block.shape[0]
        fields_str = ", ".join(field_reprs)
        return f"{type(self).__name__}(<{nrows} rows> {fields_str})"

    @cached_property
    def dataframe(self) -> _DF:
        """Return the underlying table as a DataFrame."""
        fields = list(self.__starfile_fields__.values())
        return self._get_dataframe(self._block.trust_loop(), fields)

    @property
    def block(self) -> LoopDataBlock:
        """Return the underlying LoopDataBlock."""
        return self._block

    @classmethod
    def _parse_block(cls, name: str, value: Any) -> LoopDataBlock:
        if isinstance(value, cls):
            return value._block
        if not isinstance(value, DataBlock):
            block = cls._parse_object(name, value)
        elif (block := value.try_loop()) is None:
            raise BlockValidationError(
                f"Block {value.name} cannot be interpreted as a LoopDataBlock"
            )
        return block

    @classmethod
    def _get_dataframe(cls, block: LoopDataBlock, fields: list[LoopField]) -> _DF:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _parse_object(cls, name: str, value: Any) -> LoopDataBlock:
        """Method called to parse non-DataBlock objects into LoopDataBlock.

        Construction of a loop data relies on pandas/polars DataFrame constructor, so
        this method should be implemented in subclasses.
        """
        raise NotImplementedError  # pragma: no cover


@dataclass_transform(field_specifiers=(Field,))
class SingleDataModel(BaseBlockModel):
    """Schema model for a single data block."""

    __starfile_fields__: MappingProxyType[str, SingleField]

    def __init_subclass__(cls):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                new_field = SingleField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)

        cls.__starfile_fields__ = MappingProxyType(schema_fields)

    @classmethod
    def _parse_block(cls, name: str, value: Any) -> LoopDataBlock:
        if isinstance(value, cls):
            return value._block
        if not isinstance(value, DataBlock):
            block = SingleDataBlock._from_any(name, value)
        elif (block := value.try_single()) is None:
            raise BlockValidationError(
                f"Block {value.name} cannot be interpreted as a SingleDataBlock"
            )
        return block

    @property
    def block(self) -> SingleDataBlock:
        """Return the underlying SingleDataBlock."""
        return self._block
