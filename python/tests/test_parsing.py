import time

import pandas as pd
import polars as pl
import pytest

from starfile_rs import read_star, SingleDataBlock, LoopDataBlock
from starfile_rs.core import read_star_block
from .constants import (
    loop_simple,
    postprocess,
    pipeline,
    rln31_style,
    optimiser_2d,
    optimiser_3d,
    sampling_2d,
    sampling_3d,
    single_line_middle_of_multiblock,
    single_line_end_of_multiblock,
    non_existant_file,
    two_single_line_loop_blocks,
    two_basic_blocks,
    empty_loop,
    basic_single_quote,
    basic_double_quote,
    loop_single_quote,
    loop_double_quote,
)
from .utils import generate_large_star_file


def test_instantiation():
    """Tests instantiation of the StarFile class"""
    # instantiation with file which exists
    read_star(loop_simple)

    # instantiation with non-existant file should fail
    assert non_existant_file.exists() is False
    with pytest.raises(FileNotFoundError):
        read_star(non_existant_file)


def test_read_loop_block():
    """Check that loop block is parsed correctly, data has the correct shape"""
    star = read_star(loop_simple)

    # Check that only one object is present
    assert len(star) == 1

    # get dataframe and check shape
    df_pd = star.nth(0).to_pandas()
    assert isinstance(df_pd, pd.DataFrame)
    assert df_pd.shape == (16, 12)
    df_pl = star.nth(0).to_polars()
    assert isinstance(df_pl, pl.DataFrame)
    assert df_pl.shape == (16, 12)

    # check columns
    expected_columns = [
        'rlnCoordinateX',
        'rlnCoordinateY',
        'rlnCoordinateZ',
        'rlnMicrographName',
        'rlnMagnification',
        'rlnDetectorPixelSize',
        'rlnCtfMaxResolution',
        'rlnImageName',
        'rlnCtfImage',
        'rlnAngleRot',
        'rlnAngleTilt',
        'rlnAnglePsi',
    ]
    assert list(df_pd.columns) == expected_columns
    assert df_pl.columns == expected_columns


def test_read_multiblock_file():
    """Check that multiblock STAR files such as postprocess RELION files
    parse properly
    """
    star = read_star(postprocess)
    assert len(star) == 3

    assert 'general' in star
    assert len(star['general'].trust_single()) == 6
    expected_columns = [
        'rlnFinalResolution',
        'rlnBfactorUsedForSharpening',
        'rlnUnfilteredMapHalf1',
        'rlnUnfilteredMapHalf2',
        'rlnMaskName',
        'rlnRandomiseFrom',
    ]
    assert star["general"].columns == expected_columns

    assert 'fsc' in star
    assert isinstance(star['fsc'], LoopDataBlock)
    assert star['fsc'].trust_loop().shape == (49, 7)

    assert 'guinier' in star
    assert isinstance(star['guinier'], LoopDataBlock)
    assert star['guinier'].trust_loop().shape == (49, 3)


def test_read_pipeline():
    """Check that a pipeline.star file is parsed correctly"""
    star = read_star(pipeline)

    # Check that data match file contents
    assert isinstance(star['pipeline_general'], SingleDataBlock)
    assert star['pipeline_processes'].trust_loop().to_pandas().shape == (31, 4)
    assert star['pipeline_processes'].trust_loop().to_polars().shape == (31, 4)
    assert star['pipeline_processes'].trust_loop().shape == (31, 4)
    assert star['pipeline_nodes'].trust_loop().shape == (74, 2)
    assert star['pipeline_input_edges'].trust_loop().shape == (48, 2)
    assert star['pipeline_output_edges'].trust_loop().shape == (72, 2)


def test_read_rln31():
    """Check that reading of RELION 3.1 style star files works properly"""
    s = read_star(rln31_style)

    for block in s.values():
        assert isinstance(block, LoopDataBlock)

    assert isinstance(s['block_1'], LoopDataBlock)
    assert isinstance(s['block_2'], LoopDataBlock)
    assert isinstance(s['block_3'], LoopDataBlock)


def test_read_block():
    """
    Check that passing read_n_blocks allows reading of only a specified
    number of data blocks from a star file
    """
    # test 1 block
    s = read_star_block(postprocess, "general")
    assert isinstance(s, SingleDataBlock)

    # test 2 blocks
    s = read_star_block(postprocess, "fsc")
    assert isinstance(s, LoopDataBlock)


def test_single_line_middle_of_multiblock():
    s = read_star(single_line_middle_of_multiblock)
    assert len(s) == 2


def test_single_line_end_of_multiblock():
    s = read_star(single_line_end_of_multiblock)
    assert len(s) == 2

    # iterate over dataframes, checking keys, names and shapes
    for idx, (key, block) in enumerate(s.items()):
        if idx == 0:
            assert key == 'block_1'
            assert block.trust_loop().shape == (2, 5)
        if idx == 1:
            assert key == 'block_2'
            assert block.trust_loop().shape == (1, 5)


def test_read_optimiser_2d():
    star = read_star(optimiser_2d)
    assert len(star) == 1
    assert len(star['optimiser_general']) == 84


def test_read_optimiser_3d():
    star = read_star(optimiser_3d)
    assert len(star) == 1
    assert len(star['optimiser_general']) == 84


def test_read_sampling_2d():
    star = read_star(sampling_2d)
    assert len(star) == 1
    assert len(star['sampling_general']) == 12


def test_read_sampling_3d():
    star = read_star(sampling_3d)
    assert len(star) == 2
    assert len(star['sampling_general']) == 15
    assert star['sampling_directions'].trust_loop().shape == (192, 2)


def test_parsing_speed(tmpdir):
    path = generate_large_star_file(tmpdir)
    start = time.time()
    read_star(path)
    end = time.time()

    # Check that execution takes less than 100 ms
    assert end - start < 0.1


def test_two_single_line_loop_blocks():
    star = read_star(two_single_line_loop_blocks)
    assert len(star) == 2

    assert star['block_0'].columns == [f'val{i}' for i in (1, 2, 3)]
    assert star['block_0'].trust_loop().shape == (1, 3)

    assert star['block_1'].columns == [f'col{i}' for i in (1, 2, 3)]
    assert star['block_1'].trust_loop().shape == (1, 3)


def test_two_basic_blocks():
    star = read_star(two_basic_blocks)
    assert len(star) == 2
    assert 'block_0' in star
    b0 = star['block_0']
    assert b0.trust_single().to_dict() == {
        'val1': 1.0,
        'val2': 2.0,
        'val3': 3.0,
    }
    assert 'block_1' in star
    b1 = star['block_1']
    assert b1.trust_single().to_dict() == {
        'col1': 'A',
        'col2': 'B',
        'col3': 'C',
    }


def test_empty_loop_block():
    """Parsing an empty loop block should return an empty dataframe."""
    parser = read_star(empty_loop)
    assert len(parser) == 1


@pytest.mark.parametrize("filename", [basic_single_quote, basic_double_quote])
def test_quote_basic(filename):
    star = read_star(filename)
    assert len(star) == 1
    assert star.nth(0).trust_single().to_dict()['no_quote_string'] == "noquote"
    assert star.nth(0).trust_single().to_dict()['quote_string'] == "quote string"
    assert star.nth(0).trust_single().to_dict()['whitespace_string'] == " "
    assert star.nth(0).trust_single().to_dict()['empty_string'] == ""


@pytest.mark.parametrize("filename", [loop_single_quote, loop_double_quote])
def test_quote_loop_pandas(filename):
    # NOTE: the nan values is not consistent with starfile.
    import math
    star = read_star(filename)
    assert len(star) == 1
    assert star[''].to_pandas().loc[0, 'no_quote_string'] == "noquote"
    assert star[''].to_pandas().loc[0, 'quote_string'] == "quote string"
    assert star[''].to_pandas().loc[0, 'whitespace_string'] == " "
    # assert star[''].to_pandas().loc[0, 'empty_string'] == ""
    assert math.isnan(star[''].to_pandas().loc[0, 'empty_string'])
    assert star[''].to_pandas().dtypes['number_and_string'] == "object"
    assert star[''].to_pandas().dtypes['number_and_empty'] == 'float64'
    assert star[''].to_pandas().dtypes['number'] == 'float64'
    assert star[''].to_pandas().dtypes['empty_string_and_normal_string'] == "object"

    assert math.isnan(star[''].to_pandas().loc[1, 'number_and_empty'])
    # assert star[''].to_pandas().loc[0, 'empty_string_and_normal_string'] == ''
    assert math.isnan(star[''].to_pandas().loc[0, 'empty_string_and_normal_string'])

@pytest.mark.parametrize("filename", [loop_single_quote, loop_double_quote])
def test_quote_loop_polars(filename):
    star = read_star(filename)
    assert len(star) == 1
    assert star[''].to_polars()['no_quote_string'][0] == "noquote"
    assert star[''].to_polars()['quote_string'][0] == "quote string"
    assert star[''].to_polars()['whitespace_string'][0] == " "
    # pl.Null value becomes None
    assert star[''].to_polars()['empty_string'][0] is None
    assert star[''].to_polars()['number_and_string'].dtype == pl.String
    assert star[''].to_polars()['number_and_empty'].dtype == pl.Float64
    assert star[''].to_polars()['number'].dtype == pl.Float64
    assert star[''].to_polars()['empty_string_and_normal_string'].dtype == pl.String

    assert star[''].to_polars()['number_and_empty'][1] is None
    assert star[''].to_polars()['empty_string_and_normal_string'][0] is None


def test_parse_as_string():
    star = read_star(postprocess)
    string_columns = ['rlnFinalResolution', 'rlnResolution']

    # check 'rlnFinalResolution' is parsed as string in general (basic) block
    block = star['general']
    d = block.trust_single().to_dict(string_columns=string_columns)
    assert type(d['rlnFinalResolution']) is str

    # check 'rlnResolution' is parsed as string in fsc (loop) block
    df_pd = star['fsc'].to_pandas(string_columns=string_columns)
    assert df_pd['rlnResolution'].dtype == object

    # check 'rlnResolution' is parsed as string in fsc (loop) block
    df_pl = star['fsc'].to_polars(string_columns=string_columns)
    assert df_pl['rlnResolution'].dtype == pl.String
