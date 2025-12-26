import textwrap
import pytest
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import polars as pl

from starfile_rs import read_star_text, SingleDataBlock, LoopDataBlock, compat
from starfile_rs.core import as_star, empty_star
from starfile_rs.components import _is_instance
from .constants import basic_single_quote, loop_double_quote, postprocess, rln31_style

def test_repr():
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
    assert "SingleDataBlock" in repr(star)
    assert "LoopDataBlock" in repr(star)
    single_block = star.first()
    loop_block = star.nth(1)

    single_repr = repr(single_block)
    assert "SingleDataBlock" in single_repr
    assert "item1" in single_repr
    assert "item2" in single_repr
    assert "10" in single_repr
    assert "example" in single_repr

    loop_repr = repr(loop_block)
    assert "LoopDataBlock" in loop_repr

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

def test_loop_to_single():
    star_content = """
    data_loop
    loop_
    _col1
    _col2
    1 a
    """
    star = read_star_text(textwrap.dedent(star_content))
    loop_block = star.first()
    assert isinstance(loop_block, LoopDataBlock)
    assert isinstance(loop_block.try_single(), SingleDataBlock)
    assert loop_block.try_single(False) is None
    single_block = loop_block.trust_single()
    with pytest.raises(ValueError):
        loop_block.trust_single(False)
    assert single_block.to_list() == [("col1", 1), ("col2", "a")]
    assert single_block["col1"] == 1
    assert single_block["col2"] == "a"

def test_to_dataframe():
    star_content = """
    data_single
    _item1 10
    _item2 example

    data_loop
    loop_
    _col1
    _col2
    _col3
    1 "a" -1.0
    2 b 0.0
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

    star["single_0"] = {"new_key": "new_value"}
    assert list(star["single_0"]) == ["new_key"]

    with pytest.raises(KeyError):
        star["single_2"]


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

    # obj
    star_new = star.with_loop_block(
        name="loop_4",
        data={"a": [True, False], "b": ["yes", "no"]},
        inplace=False,
    )
    assert "loop_4" not in star
    assert star_new["loop_4"].name == "loop_4"
    assert star_new["loop_4"].trust_loop().shape == (2, 2)

    star_new["loop_5"] = star_new["loop_4"]
    star_new["loop_5"] = star_new["loop_4"].trust_loop().to_pandas()
    del star_new["loop_5"]
    assert "loop_5" not in star_new

    with pytest.raises(TypeError):
        star_new.with_block(0)

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

def test_to_numeric_array():
    data = """
    data_
    loop_
    _X
    _Y
    1.0 2.0
    3.0 4.0
    5.0 6.0
    """
    star = read_star_text(textwrap.dedent(data))
    arr = star.first().to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)
    assert_allclose(arr, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

def test_from_numpy_shape_check():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    LoopDataBlock.from_numpy("name", data)  # Should not raise
    LoopDataBlock.from_numpy("name", data, columns=["A", "B", "C"])  # Should not raise
    with pytest.raises(ValueError):
        LoopDataBlock.from_numpy("name", data, columns=["A", "B"])
    with pytest.raises(ValueError):
        LoopDataBlock.from_numpy("name", data[..., None], columns=["A", "B", "C"])

def test_try_nth():
    star = as_star({"a": pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})})
    star.nth(0)  # Should not raise
    with pytest.raises(IndexError):
        star.nth(1)
    assert star.try_nth(1) is None
    assert star.try_first() is not None

def test_as_star_kwargs():
    star = as_star(
        block_1={"a": 1, "b": "text"},
        block_2=pd.DataFrame({"x": [0.1, 0.2], "y": ["p", "q"]}),
        block_3=pl.DataFrame({"m": [10, 20, 30], "n": [True, False, True]}),
        block_4={"x": 1},
    )
    assert list(star.keys()) == ["block_1", "block_2", "block_3", "block_4"]
    with pytest.raises(TypeError):
        # cannot use both positional and keyword arguments
        star = as_star(
            {"b": {"x": 1}},
            block_1={"a": 1, "b": "text"},
        )
    with pytest.raises(TypeError):
        as_star()
    assert isinstance(as_star(star), type(star))

def test_clone():
    star = as_star(
        block_1={"a": 1, "b": "text"},
        block_2=pd.DataFrame({"x": [0.1, 0.2], "y": ["p", "q"]}),
    )
    assert star.nth(0).clone().columns == ["a", "b"]
    assert star.nth(1).clone().columns == ["x", "y"]

@pytest.mark.parametrize(
    "path",
    [
        basic_single_quote,
        loop_double_quote,
        postprocess,
        rln31_style,
    ]
)
def test_compat(path, tmpdir):
    star = compat.read(path)
    star = compat.read(path, read_n_blocks=1)
    compat.write(star, tmpdir / "out.star")

def test_ipython_methods():
    star = as_star(
        block_1={"a": 1, "b": "text"},
        block_2=pd.DataFrame({"x": [0.1, 0.2], "y": ["p", "q"]}),
    )
    assert star._ipython_key_completions_() == ["block_1", "block_2"]
    assert star.nth(0)._ipython_key_completions_() == ["a", "b"]
    assert star.nth(1)._ipython_key_completions_() == ["x", "y"]

def test_comment():
    star = empty_star()
    comment = "This is a comment line."
    out = star.to_string(comment=comment)
    assert out.startswith("# " + comment)

def test_isinstance_check():
    assert _is_instance(pd.DataFrame({"a": [1]}), "pandas", "DataFrame")
    assert _is_instance(pl.DataFrame({"a": [1]}), "polars", "DataFrame")
    assert _is_instance(np.array([1, 2, 3]), "numpy", "ndarray")

def test_similar_types():
    assert not _is_instance(pd.Series([1, 2, 3]), "pandas", "DataFrame")
    class WeirdClass:
        __module__ = 0  # wrong type

    assert not _is_instance(WeirdClass(), "pandas", "DataFrame")

def test_html():
    star = as_star(
        single_block={"key1": 42, "key2": "some_value"},
        loop_block=pd.DataFrame({"col1": [164, 294], "col2": ["aaaa", "bbbbb"]}),
    )
    html_output = star._repr_html_()
    assert "key1" in html_output
    assert "key2" in html_output
    assert "col1" in html_output
    assert "col2" in html_output
    assert "42" in html_output
    assert "some_value" in html_output
    assert "164" in html_output
    assert "294" in html_output
    assert "aaaa" in html_output
    assert "bbbbb" in html_output

    html_single = star["single_block"].trust_single()._repr_html_()
    html_loop = star["loop_block"].trust_loop()._repr_html_()
    assert "key1" in html_single
    assert "key2" in html_single
    assert "col1" in html_loop
    assert "col2" in html_loop
    assert "42" in html_single
    assert "some_value" in html_single
    assert "164" in html_loop
    assert "94" in html_loop
    assert "aaaa" in html_loop
    assert "bbbbb" in html_loop

def test_very_long_html():
    from starfile_rs import _repr
    star = empty_star()
    for i in range(200):
        star.with_single_block(name=f"block_{i}", data={"key": i})
    _repr.html_block(star, max_blocks=100)  # Should not raise
