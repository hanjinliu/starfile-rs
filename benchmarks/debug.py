from pathlib import Path
import starfile_rs
import numpy as np
from timeit import default_timer
import pandas as pd  # noqa
import polars as pl  # noqa

ROOT_DIR = Path(r"C:\Users\liuha\scratch\relion-5-sta-results\Relion-5.0")


class TimeRecords:
    def __init__(self):
        self.records = []
        self.start_time = 0

    def __array__(self, dtype=None, copy=True):
        return np.array(self.records, dtype=dtype)

    def __enter__(self):
        self.start_time = default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = default_timer()
        self.records.append(end_time - self.start_time)


def random_access(num=100):
    path = ROOT_DIR / "Polish/job050/motion.star"

    time_parse = TimeRecords()
    time_to_pandas = TimeRecords()
    time_to_polars = TimeRecords()
    for i in range(num):
        with time_parse:
            block = starfile_rs.read(path)["TS_43/6682"]
        with time_to_pandas:
            df_pandas = block.as_loop().to_pandas()
        with time_to_polars:
            df_polars = block.as_loop().to_polars()
    print(type(df_pandas))
    print("shape match:", df_pandas.shape == (41, 3))
    print("values:", df_pandas.mean())
    print("values:", df_pandas.std())
    print(type(df_polars))
    print("shape match:", df_polars.shape == (41, 3))
    print("values:", df_polars.mean())
    print("values:", df_polars.std())

    print(
        f"Parse time: {np.mean(time_parse) * 1000:.1f} ± {np.std(time_parse) * 1000:.1f} ms"
    )
    print(
        f"Pandas time: {np.mean(time_to_pandas) * 1000:.2f} ± {np.std(time_to_pandas) * 1000:.2f} ms"
    )
    print(
        f"Polars time: {np.mean(time_to_polars) * 1000:.2f} ± {np.std(time_to_polars) * 1000:.2f} ms"
    )


if __name__ == "__main__":
    random_access(100)
