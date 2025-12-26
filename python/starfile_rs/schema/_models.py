from __future__ import annotations

import enum
from io import TextIOBase
from pathlib import Path
from types import MappingProxyType
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
    from typing import Self


class Extra(enum.Enum):
    ALLOW = "allow"
    FORBID = "forbid"
    IGNORE = "ignore"


ExtraType = Literal["allow", "forbid", "ignore"]


class _SchemaBase:
    __starfile_fields__: MappingProxyType[str, Field]
    __starfile_extra__: Extra

    def __repr__(self) -> str:
        field_reprs = []
        for name in self.__starfile_fields__.keys():
            value = getattr(self, name)
            field_reprs.append(f"{name}={value!r}")
        fields_str = ", ".join(field_reprs)
        return f"{type(self).__name__}({fields_str})"


class StarModel(_SchemaBase):
    """Base class for STAR file schema models."""

    __starfile_fields__: MappingProxyType[str, BlockField]

    def __init__(self, block_models: dict[str, BaseBlockModel]):
        self._block_models = block_models

    def __init_subclass__(cls, extra: ExtraType = "ignore"):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            if name in _SchemaBase.__annotations__:
                continue
            if not issubclass(annot, BaseBlockModel):
                raise TypeError(
                    f"StarModel field '{name}' must be a subclass of BaseBlockModel"
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
        star_input: dict[str, DataBlock] = {}
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
                if not isinstance(block, DataBlock):
                    block = SingleDataBlock._from_any(block_name, block)
                star_input[block_name] = field.normalize_value(block)
            elif issubclass(annot, LoopDataModel):
                if not isinstance(block, DataBlock):
                    block = LoopDataBlock._from_any(block_name, block)
                star_input[block_name] = field.normalize_value(block)
            else:
                pass
        if missing:
            raise ValidationError(
                f"StarModel {cls.__name__} is missing required fields: {', '.join(missing)}"
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
                    try:
                        block = SingleDataBlock._from_any(name, block)
                    except Exception:
                        block = LoopDataBlock._from_any(name, block)
                    star_input[name] = AnyBlock(block)

        # Sort star_input by the order of input star. This is important to keep the
        # order of blocks when writing back to file.
        star_input = {k: star_input[k] for k in star_keys if k in star_input}
        return cls(star_input)

    @classmethod
    def validate_file(cls, path) -> Self:
        """Read a STAR file and validate it against this StarModel schema."""
        required = {f.block_name for f in cls.__starfile_fields__.values()}
        mapping = {}
        for block in iter_star_blocks(path):
            if (name := block.name) in required:
                mapping[name] = block
                required.remove(name)
            if not required and not cls.__starfile_extra__:
                break
        # NOTE: we do not check if required is empty here, because validate_dict will
        # do that.
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

    def __init__(self, block: DataBlock):
        self._block = block

    @classmethod
    def validate_block(cls, value: Any) -> Self:
        # validate column names
        if not isinstance(value, DataBlock):
            raise TypeError(f"Value {value!r} is not a DataBlock")
        fields = list(cls.__starfile_fields__.values())
        missing: list[str] = []
        for f in fields:
            if f.column_name not in value.columns and f._default is Field._empty:
                missing.append(f.column_name)
        if missing:
            # If this model has no attributes, validation error will not be raised here.
            raise BlockValidationError(
                f"Block {value.name} did not pass validation by {cls.__name__!r}: "
                f"missing columns: {missing}"
            )
        return cls(value)

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


class LoopDataModel(BaseBlockModel, Generic[_DF]):
    """Schema model for a loop data block."""

    __starfile_fields__: MappingProxyType[str, LoopField]
    _series_class: type
    _block: LoopDataBlock

    def __init__(self, block: LoopDataBlock):
        super().__init__(block)
        self._dataframe_cache = None

    def __init_subclass__(cls, extra: ExtraType = "ignore"):
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
        cls.__starfile_extra__ = Extra(extra)

    def __repr__(self) -> str:
        field_reprs = []
        for name, field in self.__starfile_fields__.items():
            annot = field._get_annotation_arg_name()
            series_repr = f"Series[{annot}]"
            field_reprs.append(f"{name}={series_repr}")
        nrows = self._block.shape[0]
        fields_str = ", ".join(field_reprs)
        return f"{type(self).__name__}(<{nrows} rows> {fields_str})"

    @property
    def dataframe(self) -> _DF:
        """Return the underlying table as a DataFrame."""
        if (df := self._dataframe_cache) is None:
            fields = list(self.__starfile_fields__.values())
            df = self._get_dataframe(self._block.trust_loop(), fields)
            self._dataframe_cache = df
        return df

    @property
    def block(self) -> LoopDataBlock:
        """Return the underlying LoopDataBlock."""
        return self._block

    @classmethod
    def validate_block(cls, value: Any) -> Self:
        if not isinstance(value, DataBlock):
            raise TypeError(f"Value {value!r} is not a DataBlock")
        elif (block := value.try_loop()) is None:
            raise BlockValidationError(
                f"Block {value.name} cannot be interpreted as a LoopDataBlock"
            )
        else:
            out = block
        return super().validate_block(out)

    @classmethod
    def _get_dataframe(cls, block: LoopDataBlock, fields: list[LoopField]) -> _DF:
        raise NotImplementedError  # pragma: no cover


class SingleDataModel(BaseBlockModel):
    """Schema model for a single data block."""

    __starfile_fields__: MappingProxyType[str, SingleField]

    def __init_subclass__(cls, extra: ExtraType = "ignore"):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                new_field = SingleField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)

        cls.__starfile_fields__ = MappingProxyType(schema_fields)
        cls.__starfile_extra__ = Extra(extra)

    @classmethod
    def validate_block(cls, value: Any) -> Self:
        if not isinstance(value, DataBlock):
            raise TypeError(f"Value {value!r} is not a DataBlock")
        elif (block := value.try_single()) is None:
            raise BlockValidationError(
                f"Block {value.name} cannot be interpreted as a SingleDataBlock"
            )
        else:
            out = block
        return super().validate_block(out)

    @property
    def block(self) -> SingleDataBlock:
        """Return the underlying SingleDataBlock."""
        return self._block
