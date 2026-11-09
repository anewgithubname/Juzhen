"""Juzhen from Python.

Matrices hold float32 data in Juzhen's C++ memory pools, on the CPU or on the
GPU backend this build was compiled with::

    import juzhen as jz

    A = jz.array([[1, 2, 3], [4, 5, 6]])     # 2 x 3, on the CPU
    B = jz.array([[.1, .2], [.3, .4], [.5, .6]])
    C = jz.log(jz.exp(A @ B) + 1) / 5        # @ is matrix product, * is elementwise
    print(C)

Data is one sample per row (N x d), like numpy. Converting to or from numpy
always copies. Reading values (``.numpy()``, ``.item()``, ``print``) waits for
the GPU to finish, so do it every few steps rather than every step.

Each function here is a thin layer over the C++ code in cpp/ and ml/; see
python/src/core.cu for how a Python call reaches it.
"""

import abc
import atexit
import os

import numpy as np

if os.name == "nt":
    # Windows does not search PATH for an extension's DLLs: add the GPU
    # runtime's folder recorded at build time (OpenBLAS sits next to _core).
    from ._build_config import backend_dll_dirs

    for _dir in backend_dll_dirs:
        if os.path.isdir(_dir):
            os.add_dll_directory(_dir)

from . import _core
from ._core import abs, cos, exp, log, relu, sin, sqrt, square, tanh

__all__ = [
    "Matrix", "array", "zeros", "ones", "randn", "rand", "zeros_like", "ones_like",
    "exp", "log", "tanh", "sin", "cos", "sqrt", "square", "abs", "relu", "sigmoid",
    "sum", "mean", "max", "min", "argmax", "topk", "softmax", "logsumexp",
    "hstack", "vstack", "backend", "has_gpu", "manual_seed",
    "synchronize", "empty_cache", "save_checkpoint", "load_checkpoint",
    "nn", "optim", "data", "datasets", "matrixio", "Sequential", "Dense", "Linear",
]

# C++ output (std::cout, std::cerr) goes to sys.stdout and sys.stderr, so it
# shows in Jupyter too.
_core.attach_streams()


@atexit.register
def _at_exit():
    # Finish queued GPU work while Python is still running: the ROCm runtime
    # on Windows hangs if it is still busy when the process unloads it.
    try:
        _core.synchronize()
    finally:
        # C++ static destructors print profiler totals after Python has
        # shut down; that output is dropped.
        _core.detach_streams()


class Matrix(abc.ABC):
    """A juzhen matrix on any device: ``isinstance(x, jz.Matrix)`` works for all."""


Matrix.register(_core.cpu.Matrix)
if _core.has_gpu:
    Matrix.register(_core.gpu.Matrix)


def backend():
    """The GPU backend of this build: "cuda", "rocm", "metal", or "cpu" if none."""
    return _core.backend


def has_gpu():
    """Whether this build can use device="gpu"."""
    return _core.has_gpu


def _device(device):
    """The submodule of _core for a device name ("cpu", "gpu" or the backend name)."""
    if device == "cpu":
        return _core.cpu
    if device in ("gpu", _core.backend):
        if not _core.has_gpu:
            raise ValueError("this build of juzhen has no GPU backend; use device='cpu'")
        return _core.gpu
    raise ValueError(f"unknown device {device!r}; use 'cpu' or 'gpu'")


def _as_2d(data):
    """data as a C-order float32 2-D numpy array, one sample per row.

    A scalar becomes 1 x 1, a vector of length n becomes n x 1 (one value per
    sample), and an array with more dimensions keeps its first dimension and
    flattens the rest (N x C x H x W images become N x CHW, the layout of the
    convolution layers).
    """
    a = np.asarray(data, dtype=np.float32)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    return np.ascontiguousarray(a)


def array(data, device="cpu"):
    """A new matrix holding a copy of data (a numpy array, list or matrix)."""
    if isinstance(data, Matrix):
        data = data.numpy()
    return _device(device).from_numpy(_as_2d(data))


def zeros(rows, cols, device="cpu"):
    return _device(device).zeros(rows, cols)


def ones(rows, cols, device="cpu"):
    return _device(device).ones(rows, cols)


def randn(rows, cols, device="cpu"):
    """Standard normal samples; jz.manual_seed() makes them repeatable."""
    return _device(device).randn(rows, cols)


def rand(rows, cols, device="cpu"):
    """Uniform samples in [0, 1); jz.manual_seed() makes them repeatable."""
    return _device(device).rand(rows, cols)


def zeros_like(m):
    return zeros(*m.shape, device=m.device)


def ones_like(m):
    return ones(*m.shape, device=m.device)


def sigmoid(m):
    return 1 / (1 + exp(-m))


def sum(m, axis=None):
    """Sum over axis 0 (giving 1 x cols), axis 1 (rows x 1) or all (1 x 1).

    The result stays a matrix on m's device; use .item() for a number.
    """
    if axis is None:
        return _core.sum(_core.sum(m, 0), 1)
    return _core.sum(m, axis)


def mean(m, axis=None):
    """Mean over axis 0, axis 1 or all; a matrix like sum()."""
    rows, cols = m.shape
    count = rows * cols if axis is None else (rows if axis == 0 else cols)
    return sum(m, axis) / count


def max(m, axis=None):
    """Largest value over axis 0 (1 x cols), axis 1 (rows x 1) or all (1 x 1)."""
    if axis is None:
        return _core.max(_core.max(m, 0), 1)
    return _core.max(m, axis)


def min(m, axis=None):
    """Smallest value over axis 0 (1 x cols), axis 1 (rows x 1) or all (1 x 1)."""
    if axis is None:
        return _core.min(_core.min(m, 0), 1)
    return _core.min(m, axis)


def argmax(m, axis=1):
    """Position of the largest value in each row (axis=1, rows x 1) or column (axis=0).

    Positions are stored as floats, like every juzhen matrix; take
    ``.numpy().astype(int)`` for integers.
    """
    return _core.argmax(m, axis)


def topk(m, k, axis=1, largest=True):
    """The k largest (or smallest) values of each row (axis=1) or column (axis=0).

    Returns (values, positions), both rows x k for axis=1 (k x cols for
    axis=0), sorted from the best. Positions are floats, as in argmax().
    """
    return _core.topk(m, int(k), axis, bool(largest))


def softmax(m, axis=1):
    """exp(m) normalized to sum to one along each row (axis=1) or column (axis=0)."""
    return _core.softmax(m, axis)


def logsumexp(m, axis=1):
    """log(sum(exp(m))) of each row (axis=1, rows x 1) or column (axis=0), computed stably."""
    return _core.logsumexp(m, axis)


def _same_device(matrices, what):
    matrices = list(matrices)
    if not matrices:
        raise ValueError(f"{what} needs at least one matrix")
    for m in matrices:
        if not isinstance(m, Matrix):
            raise TypeError(f"{what} takes juzhen matrices, got {type(m).__name__}")
    devices = {m.device for m in matrices}
    if len(devices) > 1:
        raise ValueError(f"{what}: the matrices are on different devices {sorted(devices)}")
    return matrices, _device(matrices[0].device)


def hstack(matrices):
    """Matrices side by side (same number of rows), like numpy.hstack."""
    matrices, core = _same_device(matrices, "hstack")
    return core.stack(matrices, True)


def vstack(matrices):
    """Matrices one above the other (same number of columns), like numpy.vstack."""
    matrices, core = _same_device(matrices, "vstack")
    return core.stack(matrices, False)


# Shuffling in juzhen.data uses this generator, so manual_seed() covers it too.
_rng = np.random.default_rng()


def manual_seed(seed):
    """Seed every random number generator juzhen uses (CPU, GPU, data shuffling)."""
    global _rng
    _core.manual_seed(int(seed))
    _rng = np.random.default_rng(int(seed))


def synchronize():
    """Wait for the GPU to finish all queued work (for timing)."""
    _core.synchronize()


def empty_cache():
    """Free the pooled memory that no matrix is using."""
    _core.empty_cache()


from . import data, datasets, matrixio, nn, optim  # noqa: E402
from .nn import Dense, Linear, Sequential, load_checkpoint, save_checkpoint  # noqa: E402
