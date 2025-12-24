import time
from pathlib import Path
from starfile_rs import read_star, empty_star, as_star
import pandas as pd
import polars as pl
from .constants import loop_simple, postprocess, test_df
from .utils import generate_large_star_file


def test_write_simple_block(tmpdir):
    star = read_star(postprocess)
    output_file = Path(tmpdir, 'basic_block.star')
    star.write(output_file)
    assert output_file.exists()


def test_write_loop(tmpdir):
    star = read_star(loop_simple)
    output_file = Path(tmpdir, 'loop_block.star')
    star.write(output_file)
    assert output_file.exists()


def test_write_multiblock(tmpdir):
    star = read_star(postprocess)
    output_file = Path(tmpdir, 'multiblock.star')
    star.write(output_file)
    assert output_file.exists()


def test_from_single_dataframe(tmpdir):
    output_file = Path(tmpdir, 'from_df.star')

    star = empty_star()
    star.with_loop_block("", test_df).write(output_file)
    assert output_file.exists()

    read_star(output_file)


def test_create_from_dataframes(tmpdir):
    output_file = Path(tmpdir, 'from_list.star')
    star = as_star([test_df, test_df]).write(output_file)
    assert output_file.exists()

    star = read_star(output_file)
    assert len(star) == 2
    assert list(star.keys()) == ['0', '1']

def test_can_write_non_zero_indexed_one_row_dataframe(tmpdir):
    # see PR #13 - https://github.com/alisterburt/starfile/pull/13
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"])
    df.index += 1

    filename = Path(tmpdir, "test.star")
    as_star(df).write(filename)
    with open(filename) as output_file:
        output = output_file.read()

    expected = (
        "_A #1\n"
        "_B #2\n"
        "_C #3\n"
        "1\t2\t3"
    )
    assert (expected in output)


# @pytest.mark.parametrize("quote_character, quote_all_strings, num_quotes",
#                          [('"', False, 6),
#                           ('"', True, 8),
#                           ("'", False, 6),
#                           ("'", True, 8)
#                           ])
# def test_string_quoting_loop_datablock(quote_character, quote_all_strings, num_quotes, tmpdir):
#     df = pd.DataFrame(
#         [[1, "nospace", "String with space", " ", ""]],
#         columns=[
#             "a_number",
#             "string_without_space",
#             "string_space",
#             "just_space",
#             "empty_string"
#         ]
#     )

#     filename = tmpdir / "test.star"
#     StarWriter(df, filename, quote_character=quote_character, quote_all_strings=quote_all_strings).write()

#     # Test for the appropriate number of quotes
#     with open(filename) as f:
#         star_content = f.read()
#         assert star_content.count(quote_character) == num_quotes

#     s = StarParser(filename)
#     assert df.equals(s.data_blocks[""])


def test_writing_speed(tmpdir):
    path = generate_large_star_file(tmpdir)
    star = read_star(path)
    start = time.time()
    star.write(tmpdir / "output.star")
    end = time.time()

    # Check that execution takes less than a second
    assert end - start < 1.0


# @pytest.mark.parametrize("quote_character, quote_all_strings, num_quotes",
#                          [('"', False, 6),
#                           ('"', True, 8),
#                           ("'", False, 6),
#                           ("'", True, 8)
#                           ])
# def test_string_quoting_simple_datablock(quote_character, quote_all_strings, num_quotes, tmp_path):
#     o = {
#         "a_number": 1,
#         "string_without_space": "nospace",
#         "string_space": "String with space",
#         "just_space": " ",
#         "empty_string": ""
#     }

#     filename = tmp_path / "test.star"
#     StarWriter(o, filename, quote_character=quote_character, quote_all_strings=quote_all_strings).write()

#     # Test for the appropriate number of quotes
#     with open(filename) as f:
#         star_content = f.read()
#         assert star_content.count(quote_character) == num_quotes

#     s = StarParser(filename)
#     assert o == s.data_blocks[""]

def test_empty_string_simple():
    # empty strings should always be written as ''
    star = empty_star()
    star["simple"] = {
        "a": "",
        "b": "non-empty",
        "c": ""
    }
    s0 = star.to_string(comment=None)
    assert s0.strip() == (
        'data_simple\n\n'
        '_a ""\n'
        '_b non-empty\n'
        '_c ""'
    )

def test_empty_string_loop():
    # empty strings should always be written as ''
    star = empty_star()
    star["pandas"] = pd.DataFrame({"a": ["", "non-empty"], "b": ["", ""]})
    star["polars"] = pl.DataFrame({"a": ["", "non-empty"], "b": ["", ""]})
    s0 = star.to_string(comment=None)
    assert s0.strip() == (
        'data_pandas\n\n'
        'loop_\n_a #1\n_b #2\n""\t""\nnon-empty\t""\n\n\n'
        'data_polars\n\n'
        'loop_\n_a #1\n_b #2\n""\t""\nnon-empty\t""'
    )

def test_pandas_mixed_types():
    # pandas prefer object type, which causes issues when writing STAR files
    star = empty_star()
    star["pandas"] = pd.DataFrame({"a": [1, "a b", 3.0], "b": [0, 1, 2]})
    s0 = star.to_string(comment=None)
    assert s0.strip() == (
        'data_pandas\n\n'
        'loop_\n_a #1\n_b #2\n'
        '1\t0\n"a b"\t1\n3.0\t2'
    )

def test_string_with_space_simple():
    # strings with spaces should always be quoted
    star = empty_star()
    star["simple"] = {
        "a": "string with space",
        "b": "nospace",
        "c": '"OK "',
    }
    s0 = star.to_string(comment=None)
    assert s0.strip() == (
        'data_simple\n\n'
        '_a "string with space"\n'
        '_b nospace\n'
        '_c "OK "'
    )

def test_string_with_space_loop():
    # strings with spaces should always be quoted
    star = empty_star()
    star["pandas"] = pd.DataFrame({"a": ["string with space", "nospace"], "b": ["with space", "nospace"]})
    star["polars"] = pl.DataFrame({"a": ["string with space", "nospace"], "b": ["with space", "nospace"]})
    s0 = star.to_string(comment=None)
    assert s0.strip() == (
        'data_pandas\n\n'
        'loop_\n_a #1\n_b #2\n"string with space"\t"with space"\nnospace\tnospace\n\n\n'
        'data_polars\n\n'
        'loop_\n_a #1\n_b #2\n"string with space"\t"with space"\nnospace\tnospace'
    )
