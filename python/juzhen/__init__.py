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

Comparisons work element by element and give a matrix of 1s and 0s:
``(A > 0) * A``, ``jz.mean(A == B)``. ``if A == B:`` is an error unless the
matrices are 1 x 1; write ``(A == B).numpy().all()``.

A 1 x 1 matrix can be used where its number is wanted: ``f"{jz.mean(A):.3f}"``,
``float()``, ``int()``, ``round()``, ``if jz.max(A) > 1:``. Functions of one
matrix can also be written as methods: ``A.sum(0)`` is ``jz.sum(A, 0)``.
Whole numbers (class numbers, token ids, the positions from ``argmax`` and
``topk``) are numpy arrays of integers.

Each function here is a thin layer over the C++ code in cpp/ and ml/; see
python/src/core.cu for how a Python call reaches it.
"""

import abc
import atexit
import math
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

def _array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """What numpy does when a matrix meets a numpy operator or function.

    number * A, number + A, number < A, ... with a numpy number give a matrix,
    as with a Python number (numpy used to make an array behind the user's
    back). A numpy array combined or compared with a matrix is refused: one
    of them has to be converted on purpose. Anything else (np.exp(A),
    np.sum(A)) computes in numpy on a copy of the values, as before.
    """
    if method == "__call__" and len(inputs) == 2 and not kwargs and ufunc in _OPERATORS:
        left, right = inputs
        other = right if left is self else left
        if isinstance(other, np.ndarray) and other.ndim > 0:
            if ufunc in _COMPARISONS:
                # numpy would compare an n x 1 matrix with n labels as an n x n table
                raise TypeError("cannot compare a numpy array with a juzhen matrix; make it a matrix first: "
                                f"jz.array(a, device='{self.device}'), which turns a 1-D array of n values "
                                "into n x 1. For the accuracy of a classifier use "
                                "1 - nn.error_rate(scores, labels)")
            raise TypeError("cannot combine a juzhen matrix with a numpy array; make it a matrix first: "
                            f"jz.array(a, device='{self.device}')")
        if isinstance(other, (np.number, np.ndarray, int, float)) and not isinstance(other, bool):
            number = float(other)
            return _OPERATORS[ufunc](number if left is other else left, number if right is other else right)
    converted = [x.numpy() if isinstance(x, Matrix) else x for x in inputs]
    return getattr(ufunc, method)(*converted, **kwargs)


_COMPARISONS = {np.equal: lambda a, b: a == b, np.not_equal: lambda a, b: a != b,
                np.less: lambda a, b: a < b, np.less_equal: lambda a, b: a <= b,
                np.greater: lambda a, b: a > b, np.greater_equal: lambda a, b: a >= b}
_OPERATORS = {np.add: lambda a, b: a + b, np.subtract: lambda a, b: a - b,
              np.multiply: lambda a, b: a * b, np.true_divide: lambda a, b: a / b, **_COMPARISONS}
_core.cpu.Matrix.__array_ufunc__ = _array_ufunc
if _core.has_gpu:
    _core.gpu.Matrix.__array_ufunc__ = _array_ufunc

_MATRIX_TYPES = (_core.cpu.Matrix, _core.gpu.Matrix) if _core.has_gpu else (_core.cpu.Matrix,)
_NUMBERS = (int, float, np.integer, np.floating)


def _not_a_matrix(name, x):
    """The TypeError for jz.name(x) when x is not a matrix: what came, and what to write."""
    text = f"jz.{name} takes a juzhen matrix, got "
    if isinstance(x, (np.ndarray, list, tuple)):
        kind = "numpy array" if isinstance(x, np.ndarray) else type(x).__name__
        return TypeError(f"{text}a {kind}; make it a matrix first: jz.{name}(jz.array(a)), or "
                         "jz.array(a, device='gpu') for the GPU")
    if isinstance(x, _NUMBERS) and not isinstance(x, bool):
        return TypeError(f"{text}the number {x}; use Python's math module for one number, or "
                         f"jz.array({x}) for a 1 x 1 matrix")
    if isinstance(x, nn.Loss):
        return TypeError(f"{text}a Loss; its 1 x 1 matrix is loss.value")
    return TypeError(f"{text}{'None' if x is None else 'a ' + type(x).__name__}")


def _axis_error(name, axis, choices):
    """The error for an axis that jz.name does not take (choices: the ones it does, in words)."""
    hint = ""
    if name in ("max", "min") and (type(axis) in _MATRIX_TYPES or isinstance(axis, (float, np.ndarray))):
        hint = (f". jz.{name}(A, axis) is the {'largest' if name == 'max' else 'smallest'} value along an axis; "
                "the larger of two matrices, element by element, is B + jz.relu(A - B) (the smaller: "
                "B - jz.relu(B - A))")
    shown = "a matrix" if type(axis) in _MATRIX_TYPES else repr(axis)
    return ValueError(f"jz.{name}: axis must be {choices}, got {shown}{hint}")


_ROW_OR_COLUMN = "0 (down each column) or 1 (along each row)"
_ANY_AXIS = "0 (down each column), 1 (along each row) or None (everything)"


def _call(function, m, axis, choices=_ROW_OR_COLUMN, name=None):
    """_core.function(m, axis), with a sentence instead of the binding's list of overloads.

    name is the juzhen function the user called, when it is not function.
    """
    try:
        return getattr(_core, function)(m, axis)
    except TypeError:
        if type(m) not in _MATRIX_TYPES:
            raise _not_a_matrix(name or function, m) from None
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise _axis_error(name or function, axis, choices) from None
        raise


def _reduce(function, m, axis, name=None):
    """sum, max or min over an axis, or over everything for axis=None."""
    if axis is None:
        return _call(function, _call(function, m, 0, _ANY_AXIS, name), 1)
    return _call(function, m, axis, _ANY_AXIS, name)


def _elementwise(name, doc):
    function = getattr(_core, name)

    def f(m):
        try:
            return function(m)
        except TypeError:
            if type(m) in _MATRIX_TYPES:
                raise
            raise _not_a_matrix(name, m) from None

    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc + " of every element: a new matrix on m's device."
    return f


exp = _elementwise("exp", "e to the power")
log = _elementwise("log", "The natural logarithm")
tanh = _elementwise("tanh", "The hyperbolic tangent")
sin = _elementwise("sin", "The sine")
cos = _elementwise("cos", "The cosine")
sqrt = _elementwise("sqrt", "The square root")
square = _elementwise("square", "The square")
abs = _elementwise("abs", "The absolute value")
relu = _elementwise("relu", "max(x, 0)")


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


def _shape(rows, cols, what):
    """rows and cols of a new matrix as two positive ints."""
    if isinstance(rows, (tuple, list)):
        raise TypeError(f"jz.{what}(rows, cols) takes two numbers, not a shape: write "
                        f"jz.{what}({', '.join(map(str, rows))})")
    if cols is None:
        raise TypeError(f"jz.{what}(rows, cols) takes two numbers: every juzhen matrix has rows and columns; "
                        f"for a column of {rows} values write jz.{what}({rows}, 1)")
    for name, value in (("rows", rows), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not math.isfinite(value) or value != int(value) or value < 1:
            raise ValueError(f"jz.{what}: {name} must be a positive whole number, got {value!r} "
                             "(a juzhen matrix cannot be empty)")
    return int(rows), int(cols)


def zeros(rows, cols=None, device="cpu"):
    """A rows x cols matrix of zeros."""
    return _device(device).zeros(*_shape(rows, cols, "zeros"))


def ones(rows, cols=None, device="cpu"):
    """A rows x cols matrix of ones."""
    return _device(device).ones(*_shape(rows, cols, "ones"))


def randn(rows, cols=None, device="cpu"):
    """Standard normal samples; jz.manual_seed() makes them repeatable."""
    return _device(device).randn(*_shape(rows, cols, "randn"))


def rand(rows, cols=None, device="cpu"):
    """Uniform samples in [0, 1); jz.manual_seed() makes them repeatable."""
    return _device(device).rand(*_shape(rows, cols, "rand"))


def zeros_like(m):
    return zeros(*m.shape, device=m.device)


def ones_like(m):
    return ones(*m.shape, device=m.device)


def sigmoid(m):
    """1 / (1 + exp(-x)) of every element."""
    if type(m) not in _MATRIX_TYPES:
        raise _not_a_matrix("sigmoid", m)
    return 1 / (1 + exp(-m))


def sum(m, axis=None):
    """Sum over axis 0 (giving 1 x cols), axis 1 (rows x 1) or all (1 x 1).

    The result stays a matrix on m's device; use .item() for a number.
    """
    return _reduce("sum", m, axis)


def mean(m, axis=None):
    """Mean over axis 0, axis 1 or all; a matrix like sum()."""
    total = _reduce("sum", m, axis, "mean")
    rows, cols = m.shape
    return total / (rows * cols if axis is None else (rows if axis == 0 else cols))


def max(m, axis=None):
    """Largest value over axis 0 (1 x cols), axis 1 (rows x 1) or all (1 x 1)."""
    return _reduce("max", m, axis)


def min(m, axis=None):
    """Smallest value over axis 0 (1 x cols), axis 1 (rows x 1) or all (1 x 1)."""
    return _reduce("min", m, axis)


def _positions(m):
    """A matrix of positions as a numpy array of integers (waits for the GPU)."""
    return m.numpy().astype(np.int64)


def argmax(m, axis=1):
    """Position of the largest value in each row (axis=1, the default) or column (axis=0).

    Positions are whole numbers, and whole numbers are numpy arrays in juzhen
    (as class numbers and token ids are): the result is a 1-D array of
    integers, one per row. It can index a matrix or an array (``X[p]``,
    ``names[p[0]]``), be compared with labels (``np.mean(p == y)``) and be
    given to the losses. Reading it waits for the GPU. For a matrix of 1s and
    0s that marks the largest values on the device, write
    ``A == jz.max(A, 1)``.
    """
    if axis is None:
        raise ValueError("argmax works along axis 1 (each row, the default) or axis 0 (each column); for the "
                         "position in the whole matrix read as one row, as numpy counts it, use "
                         "np.argmax(A.numpy())")
    return _positions(_call("argmax", m, axis)).ravel()


def topk(m, k, axis=1, largest=True):
    """The k largest (or smallest) values of each row (axis=1) or column (axis=0).

    Returns (values, positions), both rows x k for axis=1 (k x cols for
    axis=0), sorted from the best. The values are a matrix; the positions are
    a numpy array of integers, as in argmax(). A row with fewer than k numbers
    (NaN is skipped) is filled with NaN and position -1.
    """
    if isinstance(k, bool) or not isinstance(k, _NUMBERS) or k != int(k):
        raise ValueError(f"jz.topk: k must be a whole number (how many values to keep), got {k!r}")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError(f"jz.topk: largest must be True or False, got {largest!r}")
    try:
        values, positions = _core.topk(m, int(k), axis, bool(largest))
    except TypeError:
        if type(m) not in _MATRIX_TYPES:
            raise _not_a_matrix("topk", m) from None
        raise _axis_error("topk", axis, _ROW_OR_COLUMN) from None
    return values, _positions(positions)


def softmax(m, axis=1):
    """exp(m) normalized to sum to one along each row (axis=1) or column (axis=0)."""
    return _call("softmax", m, axis)


def logsumexp(m, axis=1):
    """log(sum(exp(m))) of each row (axis=1, rows x 1) or column (axis=0), computed stably."""
    return _call("logsumexp", m, axis)


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


def _number(m, what):
    """The value of a 1 x 1 matrix, wanted by what (waits for the GPU)."""
    if m.shape != (1, 1):
        rows, cols = m.shape
        raise TypeError(f"{what} needs one number, and this matrix is {rows} x {cols}; take one value "
                        "(A[0, 0]) or reduce it first (jz.sum(A), jz.mean(A), jz.max(A))")
    return m.item()


def _format(m, spec):
    # f"{A}" prints the matrix; f"{s:.3f}" formats the number of a 1 x 1 matrix
    return format(_number(m, f"the format '{spec}'"), spec) if spec else str(m)


def _round(m, digits=None):
    return round(_number(m, "round()"), digits)


def _method(name, function):
    """jz.name(A, ...) as the method A.name(...)."""
    def method(self, *args, **kwargs):
        if "out" in kwargs:
            # numpy's functions look for a method of their name: np.sum(A)
            # calls A.sum(axis=None, out=None). They keep doing what they did
            # before the methods existed, computing in numpy on a copy.
            return getattr(np, name)(self.numpy(), *args, **kwargs)
        return function(self, *args, **kwargs)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = f"A.{name}(...) is jz.{name}(A, ...)."
    return method


# Every function of one matrix can also be written as a method, and a 1 x 1
# matrix can stand where its number is wanted, as a Loss can.
_METHODS = {"sum": sum, "mean": mean, "max": max, "min": min, "argmax": argmax, "topk": topk,
            "softmax": softmax, "logsumexp": logsumexp, "exp": exp, "log": log, "tanh": tanh, "sin": sin,
            "cos": cos, "sqrt": sqrt, "square": square, "abs": abs, "relu": relu, "sigmoid": sigmoid}
for _class in [_core.cpu.Matrix] + ([_core.gpu.Matrix] if _core.has_gpu else []):
    for _name, _function in _METHODS.items():
        setattr(_class, _name, _method(_name, _function))
    _class.__abs__ = _method("abs", abs)
    _class.__float__ = lambda m: _number(m, "float()")
    _class.__int__ = lambda m: int(_number(m, "int()"))
    _class.__round__ = _round
    _class.__format__ = _format


# Shuffling in juzhen.data uses this generator, so manual_seed() covers it too.
_rng = np.random.default_rng()


def manual_seed(seed):
    """Seed every random number generator juzhen uses (CPU, GPU, data shuffling)."""
    global _rng
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 31:
        raise ValueError(f"manual_seed takes a whole number from 0 to {2 ** 31 - 1}, got {seed!r}")
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
