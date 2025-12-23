import textwrap
import pytest
import numpy as np
import pandas as pd
import polars as pl

from starfile_rs import read_star_text, SingleDataBlock, LoopDataBlock
from starfile_rs.core import empty_star

def test_trust_and_try():
    star_content = """
    data_single
    _item1 10
    _item2 "example"

    data_loop
    loop_
    _col1
    _col2
    1 "a"
    2 "b"
    3 "c"
    """
    star = read_star_text(textwrap.dedent(star_content))
    assert isinstance(star.first().trust_single(), SingleDataBlock)
    assert isinstance(star.nth(1).trust_loop(), LoopDataBlock)
    assert isinstance(star.first().trust_loop(), LoopDataBlock)
    assert star.nth(1).try_single() is None
    with pytest.raises(ValueError):
        star.nth(1).trust_single()
    assert star.nth(1).try_single() is None
    with pytest.raises(ValueError):
        star.nth(1).trust_single(allow_conversion=False)
    assert star.first().try_loop(allow_conversion=False) is None
    with pytest.raises(ValueError):
        star.first().trust_loop(allow_conversion=False)

def test_to_dataframe():
    star_content = """
    data_single
    _item1 10
    _item2 "example"

    data_loop
    loop_
    _col1
    _col2
    _col3
    1 "a" -1.0
    2 "b" 0.0
    3 "c" 1.2e-3
    """

    star = read_star_text(textwrap.dedent(star_content))
    assert star.nth(0).to_pandas().columns.to_list() == ['item1', 'item2']
    assert star.nth(1).to_pandas().columns.to_list() == ['col1', 'col2', 'col3']
    assert star.nth(0).to_polars().columns == ['item1', 'item2']
    assert star.nth(1).to_polars().columns == ['col1', 'col2', 'col3']
    assert star.nth(0).to_numpy(structure_by="pandas").dtype.names == ('item1', 'item2')
    assert star.nth(0).to_numpy(structure_by="polars").dtype.names == ('item1', 'item2')
    assert star.nth(1).to_numpy(structure_by="pandas").dtype.names == ('col1', 'col2', 'col3')
    assert star.nth(1).to_numpy(structure_by="polars").dtype.names == ('col1', 'col2', 'col3')

def test_single_block_construction():
    star = empty_star()
    star.with_single_block(
        name="single_0",
        data={"key1": 42, "key2": 3.14, "key3": "value"}
    )
    assert star["single_0"].name == "single_0"
    assert isinstance(single := star.nth(-1), SingleDataBlock)
    assert single.to_dict() == {"key1": 42, "key2": 3.14, "key3": "value"}

    star.with_single_block(
        name="single_1",
        data=[("key1", 1), ("key2", 2.0), ("key3", "value")]
    )
    assert star["single_1"].name == "single_1"
    assert isinstance(single := star.nth(-1), SingleDataBlock)
    assert single.to_list() == [("key1", 1), ("key2", 2.0), ("key3", "value")]

def test_loop_block_construction():
    star = empty_star()

    # pandas
    star.with_loop_block(
        name="loop_0",
        data=pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    )
    assert star["loop_0"].name == "loop_0"
    assert isinstance(loop := star.nth(-1), LoopDataBlock)
    assert loop.columns == ["col1", "col2"]
    assert loop.shape == (3, 2)

    # polars
    star.with_loop_block(
        name="loop_1",
        data=pl.DataFrame({"colA": [0.1, 0.2], "colB": ["x", "y"]})
    )
    assert star["loop_1"].name == "loop_1"
    assert isinstance(loop := star.nth(-1), LoopDataBlock)
    assert loop.columns == ["colA", "colB"]
    assert loop.shape == (2, 2)

    # numpy regular array
    star.with_loop_block(
        name="loop_2",
        data=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    )
    assert star["loop_2"].name == "loop_2"
    assert isinstance(loop := star.nth(-1), LoopDataBlock)
    assert loop.columns == ["column_0", "column_1"]
    assert loop.shape == (3, 2)

    # numpy structured array
    dtype = np.dtype([("field1", np.int32), ("field2", np.float64)])
    data = np.array([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=dtype)
    star.with_loop_block(
        name="loop_3",
        data=data
    )
    assert star["loop_3"].name == "loop_3"
    assert isinstance(loop := star.nth(-1), LoopDataBlock)
    assert loop.columns == ["field1", "field2"]
    assert loop.shape == (3, 2)

def test_rename():
    data = """
    data_A
    _item 1
    _value "test"

    data_B
    loop_
    _col1
    _col2
    10 "x"
    20 "y"
    """

    star = read_star_text(textwrap.dedent(data))
    assert list(star.keys()) == ["A", "B"]
    assert star["A"].name == "A"
    assert star["B"].name == "B"

    star.rename({"A": "renamed_A", "B": "renamed_B"})
    assert list(star.keys()) == ["renamed_A", "renamed_B"]
    assert star["renamed_A"].name == "renamed_A"
    assert star["renamed_B"].name == "renamed_B"

def test_rename_columns():
    data = """
    data_A
    _item 1
    _value "test"

    data_B
    loop_
    _col1
    _col2
    10 "x"
    20 "y"
    """

    star = read_star_text(textwrap.dedent(data))

    single = star.first().trust_single()
    assert single.columns == ["item", "value"]
    assert single.to_polars().columns == ["item", "value"]
    single.columns = ["new_item", "new_value"]
    assert single.columns == ["new_item", "new_value"]
    assert single.to_polars().columns == ["new_item", "new_value"]

    loop = star.nth(1)
    assert loop.columns == ["col1", "col2"]
    assert loop.to_polars().columns == ["col1", "col2"]
    loop.columns = ["new_col1", "new_col2"]
    assert loop.columns == ["new_col1", "new_col2"]
    assert loop.to_polars().columns == ["new_col1", "new_col2"]
