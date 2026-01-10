# :star: starfile-rs :crab:

[![BSD-3-Clause License](https://img.shields.io/pypi/l/starfile-rs.svg?color=green)](https://github.com/hanjinliu/starfile-rs/blob/main/LICENSE)
[![Python package index download statistics](https://img.shields.io/pypi/dm/starfile-rs.svg)](https://pypistats.org/packages/starfile-rs)
[![PyPI version](https://badge.fury.io/py/starfile-rs.svg)](https://badge.fury.io/py/starfile-rs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/starfile-rs.svg)](https://pypi.org/project/starfile-rs)
[![codecov](https://codecov.io/gh/hanjinliu/starfile-rs/graph/badge.svg?token=X1F259JYT5)](https://codecov.io/gh/hanjinliu/starfile-rs)


A blazing-fast and type-safe STAR file reader and writer powered by Rust (as a successor of [`starfile`](https://github.com/teamtomo/starfile)).

This package implements interface with [`pandas`](https://github.com/pandas-dev/pandas), [`polars`](https://github.com/pola-rs/polars) and [`numpy`](https://github.com/numpy/numpy) for modern data manipulation, and the [`pydantic`](https://github.com/pydantic/pydantic)-like [schema validation](https://github.com/hanjinliu/starfile-rs?tab=readme-ov-file#schema-validation) for improved IDE autocompletion and safety of attribute access.

For `starfile` users, please try `from starfile_rs.compat import read` to replace `starfile.read` with minimal code changes while enjoying better performance.

## :computer: Installation

```bash
pip install starfile-rs[pandas]  # for pandas support
pip install starfile-rs[polars]  # for polars support
```

or clone this repository and install locally (requires Rust):

```bash
git clone https://github.com/hanjinliu/starfile-rs.git
cd starfile-rs
pip install -e .[pandas]  # for pandas support
pip install -e .[polars]  # for polars support
```

## :sunny: Highlights

### Easy to Use

`StarDict` provides a dict-like interface and each data block can be converted to/from popular data structures.

```python
from starfile_rs import read_star

# read as a dict-like object of data blocks
star = read_star("path/to/file.star")

# convert the first data block to pandas.DataFrame
star.nth(0).to_pandas()

# convert the "particles" data block to polars.DataFrame
star["particles"].to_polars()

# trust the "general" data block is a single data block and convert to dict
star["general"].trust_single().to_dict()

# update or add a new loop data block
import pandas as pd

star["new_loop_data"] = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
```

### Performance

All the data are from [Burt et al.](https://zenodo.org/records/11068319)

- **Example 1**: Random access to a data block located at an unknown position in a 15 MB STAR file (`Polish/job050/motion.star`)

  ![](images/time-random-access.png)

  `starfile-rs` does not parse string to `pandas.DataFrame` until you call `.to_pandas()`, so it is extremely fast for random access in a large STAR file compared to [`starfile`](https://github.com/teamtomo/starfile).

- **Example 2**: Parsing a 12 MB data block (The "particles" block from `Refine3D/bin6/run_it000_data.star`).

  ![](images/time-large-block.png)

  Reading lines and whitespace trimming are performed in Rust. This speeds up the parsing significantly even though the table parsing is similarly done by `pandas`. If you use `polars`, the performance gain is more significant.

### Type Safety

One cannot determine the structure of a STAR file until actually parsing it.`starfile-rs` splits the safe (`try_*`) and unsafe (`trust_*`) methods to avoid extensive isinstance checks.

```python
from starfile_rs import read_star

path = "path/to/file.star"

star = read_star(path)  # Safe, unless the file is broken
star["general"]  # Unsafe, if the block does not exist

if block := star.get("general"):  # Safe
    block.to_pandas()
    block.to_polars()

if block := star.get("general"):
    block.trust_single()  # Unsafe, if the block is not a single data block

if block := star.get("general"):
    if single := block.try_single():  # Safe
        single.to_dict()
```

### Schema Validation

STAR files from specific software such as [RELION](https://relion.readthedocs.io/en/latest/) often have known structures. `starfile-rs` provides **schema validation** to parse STAR files into strongly typed models, making it even much safer to access attributes.

For example, [RELION job.star file schema](examples/schema_relion_job.py) can be defined as follows.

```python
import starfile_rs.schema.pandas as schema

class Job(schema.SingleDataModel):
    type_label: str = schema.Field("rlnJobTypeLabel")
    is_continue: int = schema.Field("rlnJobIsContinue")
    is_tomo: int = schema.Field("rlnJobIsTomo")

class JobOptionsValues(schema.LoopDataModel):
    variable: schema.Series[str] = schema.Field("rlnJobOptionVariable")
    value: schema.Series[object] = schema.Field("rlnJobOptionValue")

class JobStarModel(schema.StarModel):
    job: Job = schema.Field("job")
    options: JobOptionsValues = schema.Field("joboptions_values")
```

Now, you can safely parse and access attributes from a job.star file,

```python
job = JobStarModel.validate_file("path/to/job.star")
print(job.job.type_label)
print(job.options)
```

edit the data,

```python
job.job.is_continue = 1
```

and write back to a file.

```python
job.write("path/to/new_job.star")
```
