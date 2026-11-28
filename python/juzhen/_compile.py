"""jz.compile: record the matrix operations of a function once, then run the record.

    @jz.compile
    def step(W, X, Y, lr):
        P = jz.tanh(X @ W)
        E = P - Y
        W -= lr * (X.T @ (E * (1 - P * P))) / X.shape[0]
        return jz.mean(E * E)

The first call with matrices of some shapes runs the function as written, with
each matrix wrapped in a Traced that computes every operation for real and
also writes it down. The record is then written out as a Python function of
its own (step.explain() prints it), which every later call with those shapes
runs instead of the function. Python code that is not a matrix operation
(loops, prints, counters) therefore runs once, while recording.

What cannot be repeated from the record is refused while recording, with a
CompileError that says what to write instead: reading a value out of a matrix
(float(), .numpy(), if A > 0:), a matrix that is not an argument, jz.array().
Anything a Traced does not implement fails for the same reason: an operation
that is not written down would be missing from every later call.

Numbers. A float argument is a Number while recording, so the record uses the
argument and a later call may pass another value. Whole numbers, True, False,
None and strings are taken as part of what the function is: the record is
kept for their values, as it is for the shapes.

Not thread safe: one recording at a time.
"""

import builtins
import functools
import inspect
import linecache
import math
import operator
import os
import sys
import warnings

import numpy as np

from . import _core

__all__ = ["compile", "CompileError"]

_MATRIX_TYPES = (_core.cpu.Matrix, _core.gpu.Matrix) if _core.has_gpu else (_core.cpu.Matrix,)
_DEVICE_OF = {_core.cpu.Matrix: "cpu"}
if _core.has_gpu:
    _DEVICE_OF[_core.gpu.Matrix] = "gpu"
_HERE = os.path.dirname(os.path.abspath(__file__))
_MANY = 32  # records of one function before a warning

_active = None  # the Recording in progress, if any


class CompileError(RuntimeError):
    """Something in a function under jz.compile that cannot be repeated from its record."""


def _where():
    """File and line of the user's code that is running: the nearest frame outside this package."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not os.path.abspath(filename).startswith(_HERE):
            return filename, frame.f_lineno
        frame = frame.f_back
    return "?", 0


def _literal(x):
    """Source text of a constant number."""
    if isinstance(x, (bool, np.bool_)):
        return repr(bool(x))
    if isinstance(x, (int, np.integer)):
        return repr(int(x)) if x >= 0 else f"({int(x)!r})"
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "(-inf)"
    return repr(x) if x >= 0 and not (x == 0 and math.copysign(1, x) < 0) else f"({x!r})"


class _Line:
    """One recorded operation: `out = expr`, or `expr` alone for an in-place one."""
    __slots__ = ("out", "op", "args", "expr", "where")

    def __init__(self, out, op, args, expr, where):
        self.out, self.op, self.args, self.expr, self.where = out, op, args, expr, where

    def text(self):
        return self.expr if self.out is None else f"{self.out} = {self.expr}"


class Recording:
    def __init__(self, name):
        self.name = name
        self.lines = []
        self.count = 0
        self.open = True
        self.arrays = {}  # id of a numpy array that is an argument -> its name
        self.taken = set()  # names in use

    def name_for(self, wanted):
        """A Python name that nothing else in the record uses."""
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in wanted) or "arg"
        if name[0].isdigit() or name in _NAMESPACE or (name[0] == "t" and name[1:].isdigit()):
            name = "arg_" + name
        base, n = name, 1
        while name in self.taken:
            n += 1
            name = f"{base}_{n}"
        self.taken.add(name)
        return name

    def operand(self, x, what):
        """(the real value to compute with, its text in the record) for one operand of `what`."""
        t = type(x)
        if t is Traced:
            x._check(what)
            return x._m, x._name
        if t is Number:
            x._check(what)
            return x._v, x._src
        if t in _MATRIX_TYPES:
            raise CompileError(
                f"{what} uses a matrix that is not an argument of the function being compiled. The record "
                "would keep that one matrix for every later call, which is wrong as soon as the name is given "
                "another matrix. Pass it to the function as an argument")
        if isinstance(x, (bool, int, float, np.number)):
            return x, _literal(x)
        return x, None  # not something a record can hold; the real operation says what is wrong with it

    def add(self, real, op, args, expr):
        """Write down `expr`, which gave the matrix `real`; the Traced for it."""
        self.count += 1
        name = f"t{self.count}"
        self.lines.append(_Line(name, op, args, expr, _where()))
        return Traced(real, name, self)

    def operate(self, what, run, operands, template, op):
        """Compute run(*operands) for real and write it down as template.format(*texts)."""
        pairs = [self.operand(x, what) for x in operands]
        real = run(*[p[0] for p in pairs])  # the library's own errors come from here
        if real is NotImplemented:
            return NotImplemented
        foreign = [type(x).__name__ for x, p in zip(operands, pairs) if p[1] is None]
        if type(real) not in _MATRIX_TYPES and foreign:
            return real  # A == None and the like: an answer about the types, the same in every call
        if foreign:
            raise CompileError(f"{what} with a {foreign[0]} cannot be recorded by jz.compile")
        if type(real) not in _MATRIX_TYPES:
            raise CompileError(
                f"{what} reads a value out of a matrix. Inside a function compiled with jz.compile that "
                "happens once, while recording, and the record keeps the number it saw. Keep the value as a "
                "1 x 1 matrix (A[0:1, 0:1], jz.sum(A)) or return the matrix and read it outside")
        return self.add(real, op, tuple(p[1] for p in pairs), template.format(*[p[1] for p in pairs]))


def _ended(what):
    return CompileError(
        f"{what} uses a matrix or number from the recording of a compiled function after the recording has "
        "ended (it was stored in a list, an attribute or a global while the function ran). Later calls run the "
        "record and would not store it again: return it from the function instead")


def _escape(what, kind="matrix"):
    return CompileError(
        f"{what} inside a function compiled with jz.compile reads the value of a {kind} into Python. That "
        "would happen once, while recording, and every later call would repeat what the first one saw. "
        + ("Return the matrix and read it outside the function; inside, keep working with matrices "
           "(a comparison gives a matrix of 1s and 0s to multiply by). To debug, remove @jz.compile"
           if kind == "matrix" else
           "A float argument can be added, subtracted, multiplied, divided and raised to a power (x ** 0.5 "
           "for a square root). If the function is meant to branch or loop on it, pass a whole number (the "
           "record is then kept per value) or decide outside the function"))


_OPERATORS = {  # method name -> (operator function, text, name of the operation)
    "add": (operator.add, "{0} + {1}"), "sub": (operator.sub, "{0} - {1}"),
    "mul": (operator.mul, "{0} * {1}"), "truediv": (operator.truediv, "{0} / {1}"),
    "matmul": (operator.matmul, "{0} @ {1}"), "pow": (operator.pow, "{0} ** {1}"),
    "eq": (operator.eq, "{0} == {1}"), "ne": (operator.ne, "{0} != {1}"),
    "lt": (operator.lt, "{0} < {1}"), "le": (operator.le, "{0} <= {1}"),
    "gt": (operator.gt, "{0} > {1}"), "ge": (operator.ge, "{0} >= {1}"),
}
_IN_PLACE = {"iadd": (operator.iadd, "{0} += {1}"), "isub": (operator.isub, "{0} -= {1}"),
             "imul": (operator.imul, "{0} *= {1}"), "itruediv": (operator.itruediv, "{0} /= {1}")}
_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "truediv": "/", "matmul": "@", "pow": "**", "eq": "==",
           "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "iadd": "+=", "isub": "-=", "imul": "*=",
           "itruediv": "/="}

# what the names in a record stand for
_FUNCTIONS = ("exp", "log", "tanh", "sin", "cos", "sqrt", "square", "abs", "relu", "sum", "max", "min",
              "softmax", "logsumexp")
_NAMESPACE = {name: getattr(_core, name) for name in _FUNCTIONS}
_NAMESPACE.update(cpu=_core.cpu, inf=math.inf, nan=math.nan, fabs=math.fabs)  # abs is the matrix function
if _core.has_gpu:
    _NAMESPACE["gpu"] = _core.gpu


class Traced:
    """A matrix while a compiled function is being recorded: it computes for real and writes each step down."""
    __slots__ = ("_m", "_name", "_rec", "__weakref__")
    __array_ufunc__ = None  # numpy numbers leave the operators to this class
    __hash__ = None

    def __init__(self, real, name, rec):
        self._m, self._name, self._rec = real, name, rec

    def _check(self, what):
        if not self._rec.open or self._rec is not _active:
            raise _ended(what)

    # ---- what a record can hold ----

    @property
    def shape(self):
        return self._m.shape

    @property
    def device(self):
        return self._m.device

    def __len__(self):
        return len(self._m)

    @property
    def T(self):
        self._check(".T")
        return self._rec.add(self._m.T, "T", (self._name,), f"{self._name}.T")

    def copy(self):
        self._check(".copy()")
        return self._rec.add(self._m.copy(), "copy", (self._name,), f"{self._name}.copy()")

    def to(self, device):
        self._check(".to()")
        real = self._m.to(device)
        if real is self._m:
            return self
        return self._rec.add(real, "to", (self._name,), f"{self._name}.to({device!r})")

    def __neg__(self):
        self._check("-A")
        return self._rec.add(-self._m, "neg", (self._name,), f"-{self._name}")

    def __pos__(self):
        return self

    def __abs__(self):
        return self._function("abs")

    def __getitem__(self, key):
        self._check("A[...]")
        text = self._index(key)
        real = self._m[key]  # the library's own errors come from here
        if type(real) not in _MATRIX_TYPES:
            raise CompileError(
                "A[i, j] reads one value out of a matrix. Inside a function compiled with jz.compile that "
                "happens once, while recording, and the record keeps the number it saw. A[i:i+1, j:j+1] is "
                "the same value as a 1 x 1 matrix, which can be computed with and returned")
        if text is None:
            raise CompileError(
                "A[...] with an index computed inside a function compiled with jz.compile: the record would "
                "keep the positions of the first call. Use whole numbers and slices, or pass the positions "
                "to the function as an argument (a numpy array of integers) and use that argument unchanged")
        return self._rec.add(real, "getitem", (self._name,), f"{self._name}[{text}]")

    def _index(self, key):
        """Text of an index made of whole numbers, slices and numpy arrays that are arguments; None otherwise."""
        if isinstance(key, tuple):
            parts = [self._index(k) for k in key]
            return None if None in parts else ", ".join(parts)
        if isinstance(key, (bool, np.bool_)):
            return None
        if isinstance(key, (int, np.integer)):
            return repr(int(key))
        if isinstance(key, slice):
            parts = [key.start, key.stop, key.step]
            if all(p is None or (isinstance(p, (int, np.integer)) and not isinstance(p, bool)) for p in parts):
                start, stop, step = ["" if p is None else str(int(p)) for p in parts]
                return f"{start}:{stop}" + (f":{step}" if step else "")
            return None
        if isinstance(key, list) and all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k in key):
            return repr([int(k) for k in key])
        return self._rec.arrays.get(id(key))

    def _function(self, name, *constants):
        """jz.name(self, *constants) for the functions of _FUNCTIONS; the constants are whole numbers."""
        self._check(f"jz.{name}")
        real = getattr(_core, name)(self._m, *constants)
        text = ", ".join([self._name] + [repr(c) for c in constants])
        return self._rec.add(real, name, (self._name,), f"{name}({text})")

    def __repr__(self):
        rows, cols = self._m.shape
        return f"<{rows} x {cols} matrix on '{self._m.device}' being recorded by jz.compile; it has no values to show>"

    # ---- what it cannot: a value leaving for Python ----

    def numpy(self):
        raise _escape(".numpy()")

    def item(self):
        raise _escape(".item()")

    def __array__(self, *args, **kwargs):
        raise _escape("Converting a matrix to a numpy array")

    def __float__(self):
        raise _escape("float()")

    def __int__(self):
        raise _escape("int()")

    def __index__(self):
        raise _escape("Using a matrix as a whole number")

    def __round__(self, digits=None):
        raise _escape("round()")

    def __bool__(self):
        raise _escape("`if` or `while` on a matrix (or `and`, `or`, `not`)")

    def __format__(self, spec):
        if spec:
            raise _escape(f"Formatting a matrix with '{spec}'")
        return repr(self)

    def __iter__(self):
        # rows, as for a matrix: each one is written down
        return (self[i:i + 1] for i in range(len(self._m)))


def _operator_method(name, reflected):
    run, template = _OPERATORS[name]
    what = f"A {_SYMBOL[name]} B"

    def method(self, other):
        self._check(what)
        return self._rec.operate(what, run, (other, self) if reflected else (self, other), template, name)

    method.__name__ = f"__{'r' if reflected else ''}{name}__"
    return method


def _in_place_method(name):
    run, template = _IN_PLACE[name]
    what = f"A {_SYMBOL[name]} B"

    def method(self, other):
        self._check(what)
        rec = self._rec
        real, text = rec.operand(other, what)
        self._m = run(self._m, real)  # the same matrix; the library's own errors come from here
        if text is None:
            raise CompileError(f"{what} with a {type(other).__name__} cannot be recorded by jz.compile")
        rec.lines.append(_Line(None, name, (self._name, text), template.format(self._name, text), _where()))
        return self

    method.__name__ = f"__{name}__"
    return method


for _name in _OPERATORS:
    setattr(Traced, f"__{_name}__", _operator_method(_name, False))
    if _name in ("add", "sub", "mul", "truediv", "matmul"):
        setattr(Traced, f"__r{_name}__", _operator_method(_name, True))
for _name in _IN_PLACE:
    setattr(Traced, f"__{_name}__", _in_place_method(_name))


class Number:
    """A float argument while a compiled function is being recorded: arithmetic on it is kept as text."""
    __slots__ = ("_v", "_src", "_rec")
    __array_ufunc__ = None
    __hash__ = None

    def __init__(self, value, src, rec):
        self._v, self._src, self._rec = value, src, rec

    def _check(self, what):
        if not self._rec.open or self._rec is not _active:
            raise _ended(what)

    def _with(self, other, run, template, reflected):
        self._check("arithmetic on a number")
        if type(other) is Number:
            other._check("arithmetic on a number")
            value, text = other._v, other._src
        elif isinstance(other, (int, float, np.number)) and not isinstance(other, (bool, np.bool_)):
            value, text = other, _literal(other)
        else:
            return NotImplemented  # a Traced takes it from here
        pair = ((value, self._v), (text, self._src)) if reflected else ((self._v, value), (self._src, text))
        return Number(run(*pair[0]), "(" + template.format(*pair[1]) + ")", self._rec)

    def __neg__(self):
        self._check("arithmetic on a number")
        return Number(-self._v, f"(-{self._src})", self._rec)

    def __pos__(self):
        return self

    def __abs__(self):
        self._check("arithmetic on a number")
        return Number(abs(self._v), f"fabs({self._src})", self._rec)

    def __repr__(self):
        return f"<the number {self._src} of a function being recorded by jz.compile>"

    def __float__(self):
        raise _escape("float() of a float argument (math.sqrt, math.exp and the like call it)", "number")

    def __int__(self):
        raise _escape("int() of a float argument", "number")

    def __index__(self):
        raise _escape("Using a float argument as a whole number", "number")

    def __round__(self, digits=None):
        raise _escape("round() of a float argument", "number")

    def __bool__(self):
        raise _escape("`if` or `while` on a float argument", "number")

    def __format__(self, spec):
        if spec:
            raise _escape(f"Formatting a float argument with '{spec}'", "number")
        return repr(self)


def _number_method(name, reflected):
    run, template = _OPERATORS[name]

    def method(self, other):
        return self._with(other, run, template, reflected)

    return method


def _number_comparison(name):
    def method(self, other):
        if type(other) is Traced:
            return NotImplemented
        raise _escape(f"Comparing a float argument ({_SYMBOL[name]})", "number")

    return method


for _name in ("add", "sub", "mul", "truediv", "pow"):
    setattr(Number, f"__{_name}__", _number_method(_name, False))
    setattr(Number, f"__r{_name}__", _number_method(_name, True))
for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    setattr(Number, f"__{_name}__", _number_comparison(_name))


# ---- the functions of juzhen/__init__.py reach a recording through these ----

def recording():
    """Is a compiled function being recorded?"""
    return _active is not None


def create(kind, core, rows, cols):
    """jz.zeros, ones, randn or rand inside a recording: a new matrix in every call of the record."""
    module = "cpu" if core is _core.cpu else "gpu"
    return _active.add(getattr(core, kind)(rows, cols), kind, (), f"{module}.{kind}({rows}, {cols})")


def stack(matrices, horizontal):
    """jz.hstack or vstack inside a recording."""
    what = "jz.hstack" if horizontal else "jz.vstack"
    pairs = [_active.operand(m, what) for m in matrices]
    module = "cpu" if pairs[0][0].device == "cpu" else "gpu"
    real = getattr(_core, module).stack([p[0] for p in pairs], horizontal)
    texts = tuple(p[1] for p in pairs)
    return _active.add(real, "stack", texts, f"{module}.stack([{', '.join(texts)}], {horizontal})")


def refuse_array():
    return CompileError(
        "jz.array() inside a function compiled with jz.compile: the data would be converted once, while "
        "recording, and every later call would use that first matrix. Make the matrix outside the function "
        "and pass it as an argument; jz.zeros, ones, randn and rand can be used inside")


def refuse(name):
    return CompileError(f"jz.{name} gives numpy integers, which a function compiled with jz.compile cannot "
                        f"work with: positions are values read out of a matrix. Return the matrix and call "
                        f"jz.{name} outside, or mark the largest values on the device with A == jz.max(A, 1)")


# ---- the compiled function ----

class _Record:
    """The function written from one recording."""
    __slots__ = ("run", "source", "summary")


class Compiled:
    """A function under jz.compile: call it like the function; explain() prints what it runs."""

    def __init__(self, function):
        functools.update_wrapper(self, function)
        self._function = function
        self._records = {}
        self._warned = False
        try:
            parameters = inspect.signature(function).parameters.values()
            self._positional = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            self._positional = []

    def __call__(self, *args, **kwargs):
        if _active is not None:
            # called from a function that is being recorded: its operations join that record
            return self._function(*args, **kwargs)
        leaves = []
        key = (_key(args, leaves), _key(kwargs, leaves)) if kwargs else _key(args, leaves)
        record = self._records.get(key)
        if record is None:
            return self._record(key, args, kwargs)
        return record.run(*leaves)

    def __get__(self, instance, owner=None):
        return self if instance is None else functools.partial(self, instance)

    def explain(self):
        """Print the function that calls run, for each set of argument shapes recorded so far."""
        name = self._function.__name__
        if not self._records:
            print(f"{name} has not been called yet: jz.compile records it at the first call")
        for record in self._records.values():
            print(record.summary)
            print(record.source)

    def _record(self, key, args, kwargs):
        global _active
        function = self._function
        rec = Recording(function.__name__)
        names, described = [], []

        def wrap(x, name):
            """x with its matrices and floats replaced for the recording, in the order _key reads them."""
            t = type(x)
            if t in _MATRIX_TYPES:
                name = rec.name_for(name)
                names.append(name)
                described.append(f"{name}: {x.shape[0]} x {x.shape[1]} on '{x.device}'")
                return Traced(x, name, rec)
            if t is float or isinstance(x, np.floating):
                name = rec.name_for(name)
                names.append(name)
                described.append(f"{name}: a number")
                return Number(float(x), name, rec)
            if t is np.ndarray:
                name = rec.name_for(name)
                names.append(name)
                described.append(f"{name}: a numpy array of {x.dtype}, shape {x.shape}")
                rec.arrays[id(x)] = name
                return x
            if t is tuple or t is list:
                return t(wrap(item, f"{name}_{i}") for i, item in enumerate(x))
            if t is dict:
                return {k: wrap(x[k], f"{name}_{k}") for k in sorted(x)}
            return x

        positional = self._positional
        new_args = tuple(wrap(a, positional[i] if i < len(positional) else f"arg{i}") for i, a in enumerate(args))
        new_kwargs = {k: wrap(kwargs[k], k) for k in sorted(kwargs)}

        _active = rec
        try:
            result = function(*new_args, **new_kwargs)
            returned = _returned(result, rec)
        finally:
            rec.open = False
            _active = None

        record = _Record()
        record.summary = f"# {function.__name__}, recorded for " + ("; ".join(described) or "no matrices")
        record.source, record.run = _write(function, rec, names, returned[1], len(self._records))
        self._records[key] = record
        if len(self._records) == _MANY and not self._warned:
            self._warned = True
            warnings.warn(
                f"{function.__name__} has been recorded {_MANY} times by jz.compile, once for each combination "
                "of argument shapes and of whole-number, True/False and text arguments. If a whole number "
                "changes at every call (a step counter), pass it as a float: float arguments are not part "
                "of the record", stacklevel=3)
        return returned[0]


def _key(x, leaves):
    """What the record depends on in x; the matrices, floats and arrays go to `leaves` in order."""
    t = type(x)
    if t in _MATRIX_TYPES:
        leaves.append(x)
        return t, x.shape
    if t is float:
        leaves.append(x)
        return float
    if t is tuple or t is list:
        return t, tuple([_key(item, leaves) for item in x])
    if t is dict:
        return t, tuple([(k, _key(x[k], leaves)) for k in sorted(x)])
    if t is np.ndarray:
        leaves.append(x)
        return t, x.shape, x.dtype.str
    if isinstance(x, np.floating):
        leaves.append(float(x))
        return float
    if t is Traced or t is Number:
        raise _ended("Calling a compiled function")
    if x is None or t in (int, bool, str) or isinstance(x, (np.integer, np.bool_)):
        return t, x
    raise CompileError(
        f"a function compiled with jz.compile was given a {t.__name__}. For now its arguments can be matrices, "
        "numbers, numpy arrays, True, False, None, text, and lists, tuples and dicts of these")


def _returned(x, rec):
    """(what the first call returns, the text of the record's return value)."""
    t = type(x)
    if t is Traced:
        x._check("Returning a matrix")
        return x._m, x._name
    if t is Number:
        x._check("Returning a number")
        return x._v, x._src
    if t in _MATRIX_TYPES:
        raise CompileError("a function compiled with jz.compile returned a matrix that is not one of its "
                           "arguments and was not computed from them; pass it in as an argument")
    if t is tuple or t is list:
        pairs = [_returned(item, rec) for item in x]
        texts = ", ".join(p[1] for p in pairs)
        if t is tuple:
            return tuple(p[0] for p in pairs), f"({texts}{',' if len(pairs) == 1 else ''})"
        return [p[0] for p in pairs], f"[{texts}]"
    if t is dict:
        pairs = {k: _returned(v, rec) for k, v in x.items()}
        for k in pairs:
            if not isinstance(k, (str, int)):
                raise CompileError("a compiled function can return a dict with text or whole-number keys only")
        return {k: p[0] for k, p in pairs.items()}, "{" + ", ".join(f"{k!r}: {p[1]}" for k, p in pairs.items()) + "}"
    if t is np.ndarray and id(x) in rec.arrays:
        return x, rec.arrays[id(x)]
    if x is None or t in (bool, str):
        return x, repr(x)
    if isinstance(x, (int, float, np.number)):
        return x, _literal(x)
    raise CompileError(
        f"a function compiled with jz.compile returned a {t.__name__}. It can return matrices, the numbers it "
        "was given, and tuples, lists and dicts of them; anything else was computed once, while recording")


def _write(function, rec, names, returned, number):
    """The record as the source of a Python function, and that function."""
    files = {line.where[0] for line in rec.lines}
    width = max([len(line.text()) for line in rec.lines], default=0)
    body = []
    for line in rec.lines:
        filename, lineno = line.where
        origin = f"line {lineno}" if len(files) == 1 else f"{os.path.basename(filename)}:{lineno}"
        body.append(f"    {line.text():{width}}  # {origin}")
    body.append(f"    return {returned}")
    # under the function's own name, unless that is one of the names the record uses
    name = function.__name__
    if not name.isidentifier() or name in _NAMESPACE:
        name = "compiled"
    source = f"def {name}({', '.join(names)}):\n" + "\n".join(body) + "\n"
    filename = f"<jz.compile {function.__name__} #{number + 1}>"
    namespace = dict(_NAMESPACE)
    exec(builtins.compile(source, filename, "exec"), namespace)  # noqa: S102 - source written above from the record
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return source, namespace[name]


def compile(function):  # noqa: A001 - jz.compile, as torch.compile
    """Record the matrix operations of `function` at its first call and run the record afterwards.

    See the top of this file (juzhen/_compile.py) for what a compiled function may do.
    """
    if isinstance(function, Compiled):
        return function
    if not callable(function):
        raise TypeError("jz.compile takes a function: write @jz.compile above its def")
    return Compiled(function)
