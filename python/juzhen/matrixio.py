"""Read and write Juzhen's .matrix files with numpy alone (no C++ module needed).

A .matrix file is three native ints (rows, cols, transpose) followed by the
rows x cols elements in column-major order; transpose = 1 means the matrix is
the transpose of that. The element type is not stored: pass dtype.
"""

import numpy as np

__all__ = ["read_matrix", "write_matrix", "open_matrix"]

_HEADER = np.dtype(np.int32)


def _header(f):
    header = np.fromfile(f, dtype=_HEADER, count=3)
    if header.size != 3:
        raise ValueError("incomplete .matrix header")
    rows, cols, transpose = (int(v) for v in header)
    if rows < 0 or cols < 0 or transpose not in (0, 1):
        raise ValueError("invalid .matrix header")
    return rows, cols, transpose


def _shape(raw, rows, cols, transpose):
    # column-major rows x cols is C-order cols x rows
    m = raw.reshape(cols, rows)
    return m if transpose else m.T


def read_matrix(path, dtype=np.float32):
    """The matrix in path as a numpy array of its (Juzhen) shape."""
    with open(path, "rb") as f:
        rows, cols, transpose = _header(f)
        raw = np.fromfile(f, dtype=dtype, count=rows * cols)
    if raw.size != rows * cols:
        raise ValueError(f"{path}: truncated .matrix payload")
    return _shape(raw, rows, cols, transpose)


def open_matrix(path, dtype=np.float32):
    """Like read_matrix, but memory-maps the file instead of reading it."""
    with open(path, "rb") as f:
        rows, cols, transpose = _header(f)
    raw = np.memmap(path, dtype=dtype, mode="r", offset=3 * _HEADER.itemsize, shape=(rows * cols,))
    return _shape(raw, rows, cols, transpose)


def write_matrix(path, array, dtype=np.float32):
    """Write a 2-D array to path; read_matrix(path) gives it back."""
    a = np.asarray(array, dtype=dtype)
    if a.ndim != 2:
        raise ValueError("write_matrix needs a 2-D array")
    rows, cols = a.shape
    with open(path, "wb") as f:
        np.array([rows, cols, 0], dtype=_HEADER).tofile(f)
        np.asfortranarray(a).ravel(order="F").tofile(f)
