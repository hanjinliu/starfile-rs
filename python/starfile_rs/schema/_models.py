from __future__ import annotations

from typing import (
    Any,
    Generic,
    Mapping,
    TYPE_CHECKING,
    TypeVar,
    get_origin,
    get_type_hints,
)
from starfile_rs.components import DataBlock, LoopDataBlock, SingleDataBlock
from starfile_rs.core import iter_star_blocks
from starfile_rs.schema._fields import (
    Field,
    BlockField,
    SingleField,
    LoopField,
    _BlockComponentField,
)
from starfile_rs.schema._exception import ValidationError

if TYPE_CHECKING:
    from typing import Self


class _SchemaBase:
    __starfile_fields__: dict[str, Field]

    def __repr__(self) -> str:
        field_reprs = []
        for name, field in self.__starfile_fields__.items():
            value = getattr(self, name)
            field_reprs.append(f"{name}={value!r}")
        fields_str = ", ".join(field_reprs)
        return f"{type(self).__name__}({fields_str})"


class StarModel(_SchemaBase):
    __starfile_fields__: dict[str, BlockField]

    def __init__(self, block_models: dict[str, BaseBlockModel]):
        self._block_models = block_models

    def get_block(self, name: str) -> DataBlock:
        return self._block_models[name]._block

    def __init_subclass__(cls):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            if name in StarModel.__annotations__:
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

        cls.__starfile_fields__ = schema_fields

    @classmethod
    def validate_dict(cls, star: Mapping[str, DataBlock]) -> Self:
        missing: list[str] = []
        mismatch: list[tuple[str, str, str]] = []
        star_input: dict[str, DataBlock] = {}
        annots = get_type_hints(cls)
        for name, field in cls.__starfile_fields__.items():
            if (block_name := field.block_name) not in star:
                missing.append(block_name)
            else:
                assert isinstance(field, BlockField)
                block = star[block_name]
                if issubclass(annots[name], SingleDataModel):
                    if (b_single := block.try_single()) is None:
                        mismatch.append(
                            (block_name, type(block).__name__, SingleDataBlock.__name__)
                        )
                    else:
                        star_input[block_name] = field.normalize_value(b_single)
                elif issubclass(annots[name], LoopDataModel):
                    if not isinstance(block, LoopDataBlock):
                        mismatch.append(
                            (block_name, type(block).__name__, LoopDataBlock.__name__)
                        )
                    else:
                        star_input[block_name] = field.normalize_value(block)
                else:
                    pass
        if missing:
            raise ValidationError(
                f"StarModel {cls.__name__} is missing required fields: {', '.join(missing)}"
            )
        if mismatch:
            mismatch_str = ", ".join(
                f"{name} (got {got}, expected {expected})"
                for name, got, expected in mismatch
            )
            raise ValidationError(
                f"StarModel {cls.__name__} has type mismatches for fields: {mismatch_str}"
            )
        self = cls(star_input)
        return self

    @classmethod
    def validate_file(cls, path) -> Self:
        required = {f.block_name for f in cls.__starfile_fields__.values()}
        mapping = {}
        for block in iter_star_blocks(path):
            if (name := block.name) in required:
                mapping[name] = block
                required.remove(name)
            if not required:
                break
        if required:
            raise ValidationError(
                f"StarModel {cls.__name__} is missing required fields: {', '.join(required)}"
            )
        return cls.validate_dict(mapping)


class BaseBlockModel(_SchemaBase):
    __starfile_fields__: dict[str, _BlockComponentField]

    def __init__(self, block: DataBlock):
        self._block = block

    @classmethod
    def validate_block(cls, value: DataBlock) -> Any:
        columns = [f.column_name for f in cls.__starfile_fields__.values()]
        if not all(col in value.columns for col in columns):
            missing = [col for col in columns if col not in value.columns]
            raise ValidationError(
                f"Block {value.name} did not pass validation by {cls.__name__!r}: "
                f"missing columns: {missing}"
            )
        return cls(value)


_DF = TypeVar("_DF")


class LoopDataModel(BaseBlockModel, Generic[_DF]):
    """Schema model for a loop data block."""

    __starfile_fields__: dict[str, LoopField]
    _series_class: type
    _block: LoopDataBlock

    def __init__(self, block: LoopDataBlock):
        super().__init__(block)
        self._dataframe_cache = None

    def __init_subclass__(cls):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                if issubclass(cls._series_class, get_origin(annot)):
                    raise TypeError(
                        f"LoopDataModel field '{name}' must be annotated with "
                        f"{cls._series_class.__name__}[T], got {annot}"
                    )
                new_field = LoopField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)
        cls.__starfile_fields__ = schema_fields

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
        if (df := self._dataframe_cache) is None:
            fields = list(self.__starfile_fields__.values())
            df = self._get_dataframe(self._block.trust_loop(), fields)
            self._dataframe_cache = df
        return df

    @classmethod
    def _get_dataframe(cls, block: LoopDataBlock, fields: list[LoopField]) -> _DF:
        raise NotImplementedError("Must be implemented in subclasses")


class SingleDataModel(BaseBlockModel):
    """Schema model for a single data block."""

    __starfile_fields__: dict[str, SingleField]

    def __init_subclass__(cls):
        schema_fields: dict[str, Field] = {}
        for name, annot in get_type_hints(cls).items():
            field = getattr(cls, name, None)
            if isinstance(field, Field):
                new_field = SingleField._from_field(field, annot)
                schema_fields[name] = new_field
                setattr(cls, name, new_field)
        cls.__starfile_fields__ = schema_fields
