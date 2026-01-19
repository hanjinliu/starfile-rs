from __future__ import annotations

from typing import Any


def is_scalar(obj: Any) -> bool:
    """Check if the object is a scalar value."""
    return (
        isinstance(obj, (str, int, float, bool))
        or obj is None
        or hasattr(obj, "__index__")
        or hasattr(obj, "__float__")
    )


def is_sequence(obj: Any) -> bool:
    """Check if the object is a sequence."""
    if isinstance(obj, (str, bytes, dict, bytearray)):
        # these types are iterable but not considered sequences here
        return False
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def python_obj_to_str(value: Any) -> str:
    """Convert a Python scalar to a string representation for STAR files."""
    if isinstance(value, str):
        if value == "":
            return '""'
        elif " " in value:
            if value[0] == value[-1] == '"':
                return value
            return f'"{value}"'
    return str(value)
