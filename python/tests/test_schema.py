import sys
from typing import TYPE_CHECKING
import pandas as pd
import polars as pl
import pytest
from starfile_rs import LoopDataBlock, SingleDataBlock
from starfile_rs.schema import (
    StarModel,
    SingleDataModel,
    Field,
    ValidationError,
    pandas as spd,
    polars as spl,
)
from .constants import test_data_directory, loop_simple

class General(SingleDataModel):
    final_res: float = Field("rlnFinalResolution")
    rlnMaskName: str = Field()  # test default name
    randomise_from: str = Field("rlnRandomiseFrom")  # test force str

@pytest.mark.parametrize(
    "loopDataModel, series, mod_",
    [
        (spd.LoopDataModel, spd.Series, pd),
        (spl.LoopDataModel, spl.Series, pl),
    ]
)
def test_construction(
    loopDataModel,
    series,
    mod_,
):
    if TYPE_CHECKING:
        from starfile_rs.schema.pandas import LoopDataModel, Series
        mod = pd
    else:
        LoopDataModel = loopDataModel
        Series = series
        mod = mod_

    class Fsc(LoopDataModel):
        rlnAngstromResolution: Series[str] = Field()
        fsc_corrected: Series[float] = Field("rlnFourierShellCorrelationCorrected")

    class MyModel(StarModel):
        gen: General = Field("general")
        fsc: Fsc = Field()

    m = MyModel.validate_file(test_data_directory / "basic_block.star")
    repr(m)
    repr(m.gen)
    repr(m.fsc)
    assert m.gen.final_res == pytest.approx(16.363636)
    assert m.gen.rlnMaskName == "mask.mrc"
    assert m.gen.randomise_from == "32.727273"
    assert m.fsc.dataframe.shape == (49, 2)
    assert isinstance(m.fsc.rlnAngstromResolution, mod.Series)
    assert isinstance(m.fsc.rlnAngstromResolution[0], str)
    assert isinstance(m.fsc.fsc_corrected, mod.Series)
    assert isinstance(m.fsc.fsc_corrected[0], float)
    assert isinstance(m.to_string(), str)
    assert isinstance(m.gen.to_string(), str)
    assert isinstance(m.fsc.to_string(), str)
    assert isinstance(m.gen.block, SingleDataBlock)
    assert isinstance(m.fsc.block, LoopDataBlock)

@pytest.mark.parametrize(
    "loopDataModel, series",
    [
        (spd.LoopDataModel, spd.Series),
        (spl.LoopDataModel, spl.Series),
    ]
)
def test_validation_missing_column(
    loopDataModel,
    series,
):
    if TYPE_CHECKING:
        from starfile_rs.schema.pandas import LoopDataModel, Series
    else:
        LoopDataModel = loopDataModel
        Series = series

    class GeneralLoop(LoopDataModel):
        final_res: Series[float] = Field("rlnFinalResolution")
        rlnMaskName: Series[str] = Field()  # test default name

    class Fsc(LoopDataModel):
        rlnAngstromResolution: Series[str] = Field()
        fsc_corrected: Series[float] = Field("rlnFourierShellCorrelationCorrected")

    class FscSingle(SingleDataModel):
        rlnAngstromResolution: str = Field()
        fsc_corrected: float = Field("rlnFourierShellCorrelationCorrected")

    class MyModel_0(StarModel):
        gen: General = Field("generalxxxxx")  # invalid block name
        fsc: Fsc = Field()

    class MyModel_1(StarModel):
        gen: General = Field("general")
        fsc: FscSingle = Field()  # not a single block

    class MyModel_2(StarModel):
        gen: GeneralLoop = Field("general")  # single -> loop is allowed
        fsc: Fsc = Field()

    with pytest.raises(ValidationError):
        MyModel_0.validate_file(test_data_directory / "basic_block.star")
    with pytest.raises(ValidationError):
        MyModel_1.validate_file(test_data_directory / "basic_block.star")
    MyModel_2.validate_file(test_data_directory / "basic_block.star")

def test_wrong_annotation():
    with pytest.raises(TypeError):
        # error raised on definition
        class MyModel(StarModel):
            gen: int = Field("general")  # invalid

def test_missing_annotation():
    # NOTE: error in __set_name__ is redirected to RuntimeError in Python < 3.12
    if sys.version_info < (3, 12):
        exc_type = RuntimeError
    else:
        exc_type = TypeError
    with pytest.raises(exc_type):
        class MyModel(StarModel):
            gen = Field()  # missing

def test_other_class_var_allowed():
    class MyModel(StarModel):
        gen: General = Field("general")
        some_class_var = 42  # allowed

    m = MyModel.validate_file(test_data_directory / "basic_block.star")
    assert m.gen.final_res == pytest.approx(16.363636)
    assert m.some_class_var == 42

def test_field_default():
    class OptionalField(SingleDataModel):
        some_value: float = Field(default=None)

    class MyModel(StarModel):
        gen: General = Field("general")
        optional_field: OptionalField = Field(default=None)

    m = MyModel.validate_file(test_data_directory / "basic_block.star")
    assert m.gen.final_res == pytest.approx(16.363636)
    assert m.optional_field is None

    with pytest.raises(TypeError):
        OptionalField.validate_block({"some_value": 3.14})
    m = MyModel.validate_dict(
        {
            "general": {
                "rlnFinalResolution": 10,
                "rlnMaskName": "mask.mrc",
                "rlnRandomiseFrom": "0",
            },
            "optional_field": {
                "some_value": -1.2
            },
        }
    )
    assert m.optional_field.some_value == -1.2
    m = MyModel.validate_dict(
        {
            "general": {
                "rlnFinalResolution": 10,
                "rlnMaskName": "mask.mrc",
                "rlnRandomiseFrom": "0",
            },
            "optional_field": {
                "other_value": -1.2
            },
        }
    )
    assert m.optional_field.some_value is None

def test_field_default_loop():
    from starfile_rs.schema.pandas import LoopDataModel, Series

    class Block0(LoopDataModel):
        a: Series[int] = Field(name="A")
        b: Series[float] = Field(name="B", default=None)

    class MyModel(StarModel):
        block_0: Block0 = Field()

    m = MyModel.validate_dict(
        {
            "block_0": {
                "A": [1, 2, 3],
            }
        }
    )
    df = m.block_0.dataframe
    assert "A" in df.columns
    assert "B" not in df.columns
    assert m.block_0.b is None
    assert df["A"].tolist() == [1, 2, 3]

def test_pandas_object_type_column(tmpdir):
    from starfile_rs.schema.pandas import LoopDataModel, StarModel, Field, Series

    class Block(LoopDataModel):
        value: Series[object] = Field()

    class MyModel(StarModel):
        block: Block = Field()

    m = MyModel.validate_text(
        "data_block\n"
        "loop_\n"
        "_value\n"
        "'string'\n"
        "42\n"
        "3.14\n"
    )
    save_path = tmpdir / "output.star"
    assert m.block.dataframe.dtypes["value"] == "object"
    # object column is parsed as str
    assert m.block.dataframe["value"].tolist() == ["string", "42", "3.14"]

    m = MyModel.validate_dict({"block": {"value": ["string", 42, 3.14]}})
    m.write(save_path)
    m2 = MyModel.validate_file(save_path)
    assert m2.block.dataframe.dtypes["value"] == "object"
    assert m2.block.dataframe["value"].tolist() == ["string", "42", "3.14"]

def test_repr():
    from starfile_rs.schema.pandas import LoopDataModel, StarModel, Field, Series

    class Fsc(LoopDataModel):
        rlnAngstromResolution: Series[str] = Field()
        fsc_corrected: Series[float] = Field("rlnFourierShellCorrelationCorrected")

    class MyModel(StarModel):
        general: General = Field()
        fsc: Fsc = Field()

    m = MyModel.validate_file(test_data_directory / "basic_block.star")
    assert MyModel.general is m.__starfile_fields__["general"]
    assert MyModel.fsc is m.__starfile_fields__["fsc"]
    repr(MyModel.general)
    repr(MyModel.fsc)

def test_extra():
    from starfile_rs.schema.pandas import StarModel, Field

    incoming_dict = {
        "general": {
            "rlnFinalResolution": 10,
            "rlnMaskName": "mask.mrc",
            "rlnRandomiseFrom": "0",
        },
        "extra_field": {
            "some_value": 42
        },
        "another_extra": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    }

    class MyModel(StarModel, extra="forbid"):
        general: General = Field()

    with pytest.raises(ValidationError):
        MyModel.validate_dict(incoming_dict)

    class MyModel(StarModel, extra="ignore"):
        general: General = Field()

    m = MyModel.validate_dict(incoming_dict)
    assert m.general.final_res == 10
    assert list(m.to_star_dict().keys()) == ["general"]

    class MyModel(StarModel, extra="allow"):
        general: General = Field()

    m = MyModel.validate_dict(incoming_dict)
    assert m.general.final_res == 10
    star = m.to_star_dict()
    assert list(star.keys()) == ["general", "extra_field", "another_extra"]
    assert star["extra_field"].trust_single().to_dict() == {"some_value": 42}
    assert star["another_extra"].columns == ["a", "b"]
    assert star["another_extra"].trust_loop().to_pandas().to_dict(orient="list") == {"a": [1, 2], "b": [3, 4]}

def test_validate_file():
    from starfile_rs.schema.pandas import LoopDataModel, Series

    class OneLoop(LoopDataModel):
        x: Series[float] = Field("rlnCoordinateX")
        y: Series[float] = Field("rlnCoordinateY")
        z: Series[float] = Field("rlnCoordinateZ")

    m = OneLoop.validate_file(loop_simple)
    assert m.x.size > 10

@pytest.mark.parametrize(
    "loopDataModel, series, mod_",
    [
        (spd.LoopDataModel, spd.Series, pd),
        (spl.LoopDataModel, spl.Series, pl),
    ]
)
def test_setting_dataframe(
    loopDataModel,
    series,
    mod_,
):
    if TYPE_CHECKING:
        from starfile_rs.schema.pandas import LoopDataModel, Series
        mod = pd
    else:
        LoopDataModel = loopDataModel
        Series = series
        mod = mod_

    class OneLoop(LoopDataModel):
        x: Series[float] = Field("rlnCoordinateX")
        y: Series[float] = Field("rlnCoordinateY")
        z: Series[float] = Field("rlnCoordinateZ")

    class MyModel(StarModel, extra="allow"):
        general: General = Field()
        one_loop: OneLoop = Field("loop_1")  # type: ignore

    m = MyModel.validate_dict(
        {
            "general": {
                "rlnFinalResolution": 10,
                "rlnMaskName": "mask.mrc",
                "rlnRandomiseFrom": "0",
            },
            "loop_1": {
                "rlnCoordinateX": [1.0, 2.0, 3.0],
                "rlnCoordinateY": [4.0, 5.0, 6.0],
                "rlnCoordinateZ": [7.0, 8.0, 9.0],
            },
        }
    )
    m.general = {"rlnFinalResolution": 12.0, "rlnMaskName": "new_mask.mrc", "rlnRandomiseFrom": "1.0"}
    assert m.general.final_res == 12.0
    assert m.general.rlnMaskName == "new_mask.mrc"
    assert m.general.randomise_from == "1.0"

    # once schema is ready, setting any type of dataframes should be allowed
    m.one_loop = {
        "rlnCoordinateX": [10.0, 20.0],
        "rlnCoordinateY": [30.0, 40.0],
        "rlnCoordinateZ": [50.0, 60.0],
    }
    assert m.one_loop.dataframe.shape == (2, 3)

    m.one_loop = mod.DataFrame(
        {
            "rlnCoordinateX": [100.0, 200.0],
            "rlnCoordinateY": [300.0, 400.0],
            "rlnCoordinateZ": [500.0, 600.0],
        }
    )
    assert m.one_loop.dataframe["rlnCoordinateX"].max() > 150.0
    if mod is pd:
        mod_other = pl
    else:
        mod_other = pd
    m.one_loop = mod_other.DataFrame(
        {
            "rlnCoordinateX": [-100.0, 200.0],
            "rlnCoordinateY": [-300.0, 400.0],
            "rlnCoordinateZ": [-500.0, 600.0],
        }
    )
    assert m.one_loop.dataframe["rlnCoordinateX"].min() < -50.0

@pytest.mark.parametrize(
    "loopDataModel, series",
    [
        (spd.LoopDataModel, spd.Series),
        (spl.LoopDataModel, spl.Series),
    ]
)
def test_dataclass_like_init(
    loopDataModel,
    series,
):
    if TYPE_CHECKING:
        from starfile_rs.schema.pandas import LoopDataModel, Series
    else:
        LoopDataModel = loopDataModel
        Series = series

    class OneLoop(LoopDataModel):
        x: Series[float] = Field("rlnCoordinateX")
        y: Series[float] = Field("rlnCoordinateY")
        z: Series[float] = Field("rlnCoordinateZ")

    class MyModel(StarModel, extra="allow"):
        general: General = Field()
        one_loop: OneLoop = Field("loop_1")

    m = MyModel(
        general=General(
            final_res=15.0,
            rlnMaskName="mask2.mrc",
            randomise_from="2.0",
        ),
        one_loop=OneLoop(
            x=[1.0, 2.0, 3.0],
            y=[4.0, 5.0, 6.0],
            z=[7.0, 8.0, 9.0],
        ),
    )
    assert m.general.final_res == 15.0
    assert m.general.rlnMaskName == "mask2.mrc"
    assert m.general.randomise_from == "2.0"
    assert list(m.one_loop.x) == pytest.approx([1.0, 2.0, 3.0])
    assert list(m.one_loop.y) == pytest.approx([4.0, 5.0, 6.0])
    assert list(m.one_loop.z) == pytest.approx([7.0, 8.0, 9.0])
