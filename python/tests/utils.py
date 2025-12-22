from io import StringIO
import numpy as np
from pathlib import Path

def generate_large_star_file(tmpdir: str) -> str:
    data = np.random.randint(0, 100, size=(100000, 4))
    buf = StringIO()
    np.savetxt(buf, data, encoding="utf-8", fmt="%d")
    buf.seek(0)
    star_path = Path(tmpdir) / "large_test.star"
    star_path.write_text(
        f"data_large\nloop_\n_A\n_B\n_C\n_D\n{buf.read()}",
        encoding="utf-8",
    )
    return star_path
