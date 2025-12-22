# starfile-rs

A blazing-fast and type-safe STAR file reader/writer powered by Rust.

```python
from starfile_rs import read_star

read_star("path/to/file.star")  # read as a dict of data blocks
```

## Highlights

### Performance

All the data are from [Burt et al.](https://zenodo.org/records/11068319).

- Example 1: Random access to a data block located at an unknown position in a 15 MB STAR file (`Polish/job050/motion.star`)

  ![](images/time-random-access.png)

- Example 2: Random access to a 12 MB data block (The "particles" block from `"Refine3D/bin6/run_it000_data.star"`).

  ![](images/time-large-block.png)

   Parsing from `str` to `pandas.DataFrame` is not what we can improve, so there is only a small (but still, substantial) difference here. However, if you use `polars.DataFrame`, the performance gain is more significant.

### Type Safety

One cannot determine the structure of a STAR file until actually parsing it.
`starfile-rs` splits the unsafe and safe methods to avoid extensive isinstance checks at
runtime.

```python
from starfile_rs import read_star, read_star_blocks

path = "path/to/file.star"

star = read_star(path)  # -> Safe, if you trust the file
star["general"]  # Unsafe, if the block does not exist

if block := star.get("general"):  # Safe
    print(block.to_pandas())

if block := star.get("general"):
    block.as_single()  # Unsafe, if the block is not single data block

if block := star.get("general"):
    if single := block.try_as_single():  # Safe
        print(single.to_dict())
```
