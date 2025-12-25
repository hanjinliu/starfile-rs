from typing import TYPE_CHECKING
import pandas as pd
import polars as pl
import pytest
from starfile_rs.schema import (
    StarModel,
    SingleDataModel,
    Field,
    ValidationError,
    pandas as spd,
    polars as spl,
)
from .constants import test_data_directory

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
    assert m.fsc._dataframe_cache is None
    assert m.fsc.dataframe.shape == (49, 2)
    assert m.fsc._dataframe_cache is not None
    assert isinstance(m.fsc.rlnAngstromResolution, mod.Series)
    assert isinstance(m.fsc.rlnAngstromResolution[0], str)
    assert isinstance(m.fsc.fsc_corrected, mod.Series)
    assert isinstance(m.fsc.fsc_corrected[0], float)


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
        fsc: FscSingle = Field()

    class MyModel_2(StarModel):
        gen: GeneralLoop = Field("general")
        fsc: Fsc = Field()

    with pytest.raises(ValidationError):
        MyModel_0.validate_file(test_data_directory / "basic_block.star")
    with pytest.raises(ValidationError):
        MyModel_1.validate_file(test_data_directory / "basic_block.star")
    with pytest.raises(ValidationError):
        MyModel_2.validate_file(test_data_directory / "basic_block.star")

def test_wrong_annotation():
    with pytest.raises(TypeError):
        # error raised on definition
        class MyModel(StarModel):
            gen: int = Field("general")  # invalid

def test_missing_annotation():
    with pytest.raises(TypeError):
        class MyModel(StarModel):
            gen = Field()  # missing

def test_other_class_var_allowed():
    class MyModel(StarModel):
        gen: General = Field("general")
        some_class_var = 42  # allowed

    m = MyModel.validate_file(test_data_directory / "basic_block.star")
    assert m.gen.final_res == pytest.approx(16.363636)
    assert m.some_class_var == 42

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
