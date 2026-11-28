"""juzhen.compile and the operations it is built on, on every device of the build."""

import numpy as np
import pytest

import juzhen as jz
from juzhen import _core


def plain(a, device):
    """a as a matrix stored column by column, as jz.randn and a matrix product give it."""
    M = jz.zeros(*a.shape, device=device)
    M += jz.array(a, device=device)
    return M


def orders(a, device):
    """Makers of a in the two storage orders: by rows (from numpy) and by columns."""
    return [lambda: jz.array(a, device=device), lambda: plain(a, device)]


def same(M, expected):
    """M has the bits of expected and is stored in the same order."""
    got, want = M.numpy(), expected.numpy()
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes(), f"largest difference {np.nanmax(np.abs(got - want))}"
    assert _core.stored_transposed(M) == _core.stored_transposed(expected)


def values(rng, shape):
    a = rng.standard_normal(shape).astype(np.float32)
    flat = a.reshape(-1)
    for i, special in enumerate([0.0, -0.0, np.inf, -np.inf, np.nan][:flat.size // 3]):
        flat[i * 3 % flat.size] = special
    return a


SHAPES = [(1, 1), (1, 9), (7, 1), (37, 53), (70, 70)]

# name of the operation, and what it must leave in its first operand
WITH_A_MATRIX = {"add_": lambda A, B: A + B, "sub_": lambda A, B: A - B, "rsub_": lambda A, B: B - A,
                 "mul_": lambda A, B: A * B, "radd_": lambda A, B: B + A, "rmul_": lambda A, B: B * A}
WITH_A_NUMBER = {"add_": lambda A, s: A + s, "sub_": lambda A, s: A - s, "rsub_": lambda A, s: s - A,
                 "mul_": lambda A, s: A * s, "div_": lambda A, s: A / s, "rdiv_": lambda A, s: s / A}
OF_ONE_MATRIX = {"neg_": lambda A: -A, "exp_": jz.exp, "log_": jz.log, "tanh_": jz.tanh, "square_": jz.square,
                 "sqrt_": jz.sqrt, "sin_": jz.sin, "cos_": jz.cos, "relu_": jz.relu, "abs_": jz.abs}


class TestTakeOver:
    """add_(A, B) and the like leave in A exactly what the operator gives: the bits and the storage order."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_with_a_matrix_in_every_pairing(self, device, rng, shape):
        a, b = values(rng, shape), values(rng, shape)
        for make_a in orders(a, device):
            for make_b in orders(b, device):
                for name, operator in WITH_A_MATRIX.items():
                    A, B = make_a(), make_b()
                    expected = operator(A, B)
                    assert getattr(_core, name)(A, B) is None
                    same(A, expected)
                    same(B, make_b())  # the other operand is left alone

    @pytest.mark.parametrize("shape", [(1, 9), (7, 1), (37, 53)])
    def test_the_other_operand_is_broadcast(self, device, rng, shape):
        a = values(rng, shape)
        for small in [(1, shape[1]), (shape[0], 1), (1, 1)]:
            b = values(rng, small)
            for make_a in orders(a, device):
                for make_b in orders(b, device):
                    for name, operator in WITH_A_MATRIX.items():
                        A, B = make_a(), make_b()
                        expected = operator(A, B)
                        getattr(_core, name)(A, B)
                        same(A, expected)

    @pytest.mark.parametrize("n", [1, 3, 70])
    def test_with_itself_and_its_own_transpose(self, device, rng, n):
        a = values(rng, (n, n))
        for make in orders(a, device):
            for name, operator in WITH_A_MATRIX.items():
                A = make()
                expected = operator(make(), make())
                getattr(_core, name)(A, A)
                same(A, expected)
                A = make()
                expected = operator(make(), make().T)
                getattr(_core, name)(A, A.T)
                same(A, expected)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_with_a_number(self, device, rng, shape):
        a = values(rng, shape)
        for make in orders(a, device):
            for name, operator in WITH_A_NUMBER.items():
                for s in (2.5, -3, 0.1, np.float32(1.5)):
                    A = make()
                    expected = operator(make(), s)
                    assert getattr(_core, name)(A, s) is None
                    same(A, expected)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_functions_of_one_matrix(self, device, rng, shape):
        a = values(rng, shape)
        for make in orders(a, device):
            for name, function in OF_ONE_MATRIX.items():
                A = make()
                expected = function(make())
                assert getattr(_core, name)(A) is None
                same(A, expected)

    @pytest.mark.parametrize("shape", [(1, 9), (7, 1), (37, 53)])
    def test_broadcasting_gives_what_numpy_gives(self, device, rng, shape):
        # the operators leave the result in the matrix they made for the broadcast operand; each entry is
        # still one float32 operation
        a = values(rng, shape)
        for small in [(1, shape[1]), (shape[0], 1), (1, 1)]:
            b = values(rng, small)
            for make_a in orders(a, device):
                for make_b in orders(b, device):
                    A, B = make_a(), make_b()
                    with np.errstate(all="ignore"):
                        for got, want in [(A + B, a + b), (B + A, b + a), (A - B, a - b), (B - A, b - a),
                                          (A * B, a * b), (B * A, b * a)]:
                            np.testing.assert_array_equal(got.numpy(), want)
                    same(A, make_a())
                    same(B, make_b())
        row, column = values(rng, (1, shape[1])), values(rng, (shape[0], 1))
        with np.errstate(all="ignore"):
            for make_row in orders(row, device):
                for make_column in orders(column, device):
                    R, C = make_row(), make_column()
                    for got, want in [(R + C, row + column), (C - R, column - row), (R * C, row * column)]:
                        np.testing.assert_array_equal(got.numpy(), want)

    def test_a_larger_operand_is_refused(self, device, rng):
        A = jz.array(values(rng, (1, 5)), device=device)
        B = jz.array(values(rng, (4, 5)), device=device)
        for name in WITH_A_MATRIX:
            with pytest.raises(ValueError, match="cannot broadcast"):
                getattr(_core, name)(A, B)
        same(A, jz.array(A.numpy(), device=device))

    def test_which_ones_use_the_storage_of_the_operand(self, device, rng):
        # a view made before the operation sees the new values exactly when the storage was used; this is
        # where the saving comes from, so a backend that stops doing it should be noticed
        a, b = values(rng, (37, 53)), values(rng, (37, 53))
        metal_gpu = device == "gpu" and jz.backend() == "metal"
        cuda_gpu = device == "gpu" and jz.backend() == "cuda"

        def in_storage(operation, make):
            A = make()
            view = A.T
            operation(A)
            return view.numpy().tobytes() == A.numpy().T.tobytes()

        for make_a in orders(a, device):
            for make_b in orders(b, device):
                B = make_b()
                for name, operator in WITH_A_MATRIX.items():
                    # whenever the operator stores its result the way A is stored; two matrices stored
                    # column by column, as computed ones are, always qualify
                    like_a = _core.stored_transposed(operator(make_a(), B)) == _core.stored_transposed(make_a())
                    assert like_a or _core.stored_transposed(make_a()) or _core.stored_transposed(B)
                    # CUDA multiplies two matrices that are stored differently into a new one
                    mixed = _core.stored_transposed(make_a()) != _core.stored_transposed(B)
                    expected = like_a and not (name in ("mul_", "rmul_") and mixed and cuda_gpu)
                    assert in_storage(lambda A: getattr(_core, name)(A, B), make_a) == expected, name
            for name in WITH_A_NUMBER:
                assert in_storage(lambda A: getattr(_core, name)(A, 1.5), make_a), name
            for name in OF_ONE_MATRIX:
                expected = name != "log_" and not (name == "exp_" and metal_gpu)
                assert in_storage(lambda A: getattr(_core, name)(A), make_a) == expected, name


# ---------------------------------------------------------------------------------------------------------
# jz.compile

def bits(result):
    """The bytes of every matrix and the values of everything else in a result."""
    if isinstance(result, jz.Matrix):
        return result.shape, result.numpy().tobytes()
    if isinstance(result, (tuple, list)):
        return [bits(item) for item in result]
    if isinstance(result, dict):
        return {key: bits(value) for key, value in result.items()}
    if isinstance(result, np.ndarray):
        return result.tobytes()
    return result


def records(function):
    return len(function._records)


def step(W, X, Y, lr):
    P = jz.tanh(X @ W)
    E = P - Y
    G = X.T @ (E * (1 - P * P)) / X.shape[0]
    W -= lr * G
    return jz.mean(E * E)


def everything(A, B, C, idx, s):
    """Every operation a record can hold; A and B are n x m, C is m x n."""
    out = [A + B, A - B, A * B, A / (jz.abs(B) + 1), A @ C, -A, abs(A), A ** 2, (jz.abs(A) + 1) ** 0.5,
           2 + A, 2 - A, 2 * A, 2 / (jz.abs(A) + 1), A + 2, A - 2, A * 2, A / 2,
           A < B, A <= B, A > B, A >= B, A == B, A != B, A > 0.5, 0.5 > A,
           jz.exp(A), jz.log(jz.abs(A) + 1), jz.tanh(A), jz.sin(A), jz.cos(A), jz.sqrt(jz.abs(A)),
           jz.square(A), jz.abs(A), jz.relu(A), jz.sigmoid(A), A.exp(), A.relu(),
           jz.sum(A), jz.sum(A, 0), jz.sum(A, 1), jz.mean(A), jz.mean(A, 0), jz.mean(A, 1),
           jz.max(A), jz.max(A, 0), jz.max(A, 1), jz.min(A), jz.min(A, 0), jz.min(A, 1), A.sum(1),
           jz.softmax(A), jz.softmax(A, 0), jz.logsumexp(A), jz.logsumexp(A, 0),
           A.T, A.T @ B, A.copy(), A.to(A.device),
           A[1:3], A[:, 0:2], A[2], A[1:, 1], A[[0, 2]], A[idx], A[idx, 0:2],
           jz.hstack([A, B]), jz.vstack([A, B, A]),
           jz.zeros_like(A) + A, jz.ones(2, 3, device=A.device), jz.ones_like(A) * s,
           s * A, A * s, A / s, s - A, (1 - s) * A, (s / 2) * A, -s * A, s ** 2 * A, 2 ** s * A, abs(s) * A,
           A > s, s > A]
    T = A + B
    T += A
    T -= 1.5
    T *= s
    T /= 3
    T -= B
    out.append(T)
    return out


class TestCompile:
    """A compiled function gives the bits of the function as written, at the recording call and after."""

    def test_a_training_step(self, device, rng):
        x, y = values(rng, (64, 20))[1:], rng.standard_normal((63, 3)).astype(np.float32)
        w = rng.standard_normal((20, 3)).astype(np.float32) * 0.1
        x[np.isnan(x) | np.isinf(x)] = 0.5
        compiled = jz.compile(step)
        for make_x in orders(x, device):
            for make_y in orders(y, device):
                X, Y = make_x(), make_y()
                W1, W2 = jz.array(w, device=device), jz.array(w, device=device)
                for i in range(4):
                    assert bits(compiled(W1, X, Y, 0.1 * (i + 1))) == bits(step(W2, X, Y, 0.1 * (i + 1)))
                    assert bits(W1) == bits(W2)
        assert records(compiled) == 1  # the storage order and the value of lr are not part of the record

    def test_every_operation(self, device, rng):
        a, b, c = values(rng, (5, 4)), values(rng, (5, 4)), values(rng, (4, 5))
        idx = np.array([3, 0, 3])
        compiled = jz.compile(everything)
        for make_a in orders(a, device):
            for make_b in orders(b, device):
                for s in (0.5, 2.0, np.float32(1.5)):
                    A, B, C = make_a(), make_b(), jz.array(c, device=device)
                    assert bits(compiled(A, B, C, idx, s)) == bits(everything(A, B, C, idx, s))
                    assert bits([A, B, C]) == bits([make_a(), make_b(), jz.array(c, device=device)])
        assert records(compiled) == 1

    def test_one_record_per_shape_and_per_whole_number(self, device, rng):
        def g(A, k, flag=True, name=None):
            B = A * k if flag else A - k
            for _ in range(k):  # a loop over a whole number is written out
                B = B + A
            return B

        f = jz.compile(g)
        A, B = jz.array(values(rng, (3, 4)), device=device), jz.array(values(rng, (2, 4)), device=device)
        for args, kwargs in [((A, 2), {}), ((A, 2), {}), ((B, 2), {}), ((A, 3), {}), ((A, 2), {"flag": False}),
                             ((A, 2), {"flag": False}), ((A, 2), {"name": "x"}), ((A, 2), {"flag": True})]:
            assert bits(f(*args, **kwargs)) == bits(g(*args, **kwargs))
        assert records(f) == 6

    def test_lists_tuples_and_dicts(self, device, rng):
        def f(pair, table, scale):
            return {"sum": pair[0] + table["b"] * scale, "both": (pair[1] - table["a"], [pair[0], scale * 2])}

        compiled = jz.compile(f)
        m = [jz.array(values(rng, (3, 3)), device=device) for _ in range(4)]
        for scale in (0.5, 3.0):
            got = compiled((m[0], m[1]), {"a": m[2], "b": m[3]}, scale)
            assert bits(got) == bits(f((m[0], m[1]), {"a": m[2], "b": m[3]}, scale))
            assert got["both"][1][0] is m[0]  # an argument that is returned is that matrix
        assert records(compiled) == 1

    def test_random_matrices_are_new_in_every_call(self, device, rng):
        def f(A):
            return A + jz.randn(*A.shape, device=A.device) + jz.rand(*A.shape, device=A.device)

        compiled = jz.compile(f)
        A = jz.array(values(rng, (6, 6)), device=device)
        jz.manual_seed(3)
        got = [bits(compiled(A)) for _ in range(3)]
        jz.manual_seed(3)
        assert got == [bits(f(A)) for _ in range(3)]
        assert got[1] != got[2]

    def test_python_code_runs_once(self, device, rng):
        seen = []

        @jz.compile
        def f(A):
            seen.append("called")
            assert isinstance(A, jz.Matrix)
            return A * 2

        A = jz.array(values(rng, (2, 2)), device=device)
        for _ in range(3):
            assert bits(f(A)) == bits(A * 2)
        assert seen == ["called"]

    def test_a_compiled_function_called_from_another(self, device, rng):
        @jz.compile
        def inner(A, s):
            return jz.tanh(A) * s

        @jz.compile
        def outer(A, s):
            return inner(A, s) + inner(A + 1, 2.0)

        A = jz.array(values(rng, (3, 3)), device=device)
        for s in (0.5, 1.5):
            assert bits(outer(A, s)) == bits(jz.tanh(A) * s + jz.tanh(A + 1) * 2.0)
        assert records(outer) == 1 and records(inner) == 0

    def test_explain(self, device, rng, capsys):
        compiled = jz.compile(step)
        compiled.explain()
        assert "has not been called yet" in capsys.readouterr().out
        X, Y = jz.array(values(rng, (8, 3))[2:], device=device), jz.zeros(6, 2, device=device)
        compiled(jz.zeros(3, 2, device=device), X, Y, 0.1)
        compiled.explain()
        out = capsys.readouterr().out
        assert f"# step, recorded for W: 3 x 2 on '{device}'; X: 6 x 3 on '{device}'" in out and "lr: a number" in out
        assert "def step(W, X, Y, lr):" in out and "t1 = X @ W" in out and "tanh_(t1)" in out
        assert "W -= t" in out and "return t" in out
        assert "# 5 new matrices in each call; as written, the function makes 13" in out

    def test_an_error_while_recording_leaves_no_record(self, device, rng):
        @jz.compile
        def f(A, B):
            return A @ B

        A = jz.array(values(rng, (2, 3)), device=device)
        with pytest.raises(Exception, match="2 x 3|2x3|dimension|shape"):  # the library's own message
            f(A, A)
        assert records(f) == 0
        assert bits(f(A, A.T)) == bits(A @ A.T)
        assert bits(jz.array([[1.0]], device=device)) == bits(jz.ones(1, 1, device=device))  # not recording now

    def test_many_records_give_a_warning(self, device):
        @jz.compile
        def f(A, t):
            return A * t

        A = jz.ones(2, 2, device=device)
        with pytest.warns(UserWarning, match="pass it as a float"):
            for t in range(40):
                f(A, t)


# small functions around the rule for reuse: (function of A and B, new matrices per call)
REUSE_CASES = [
    (lambda A, B: jz.exp(A), 1),                                 # an argument is never taken over
    (lambda A, B: jz.exp(A + B), 1),
    (lambda A, B: jz.tanh(jz.exp(-(A + B)) * 2 + 1) ** 2, 1),    # a chain in one matrix
    (lambda A, B: (A + B) * (A + B), 1),                         # written twice, computed once
    (lambda A, B: (A + B) * (B + A), 2),                         # two temporaries, the product goes into one
    (lambda A, B: jz.sum((A - B) * (A - B)) + jz.sum((A - B) * (A - B)), 3),
    (lambda A, B: (A.T @ B) + (A.T @ B), 1),
    (lambda A, B: [T := A + B, U := A + B, V := U.T, jz.exp(T) @ V][3], 3),  # U is T, so V shows T's storage
    (lambda A, B: (A + B, A + B), 2),                            # two results stay two matrices
    (lambda A, B: jz.zeros_like(A) + jz.ones_like(A) + jz.ones_like(A), 3),
    (lambda A, B: [T := A + B, T * 2][1], 1),
    (lambda A, B: [T := A + B, (T, T * 2)][1], 2),               # a returned matrix keeps its values
    (lambda A, B: [T := A + B, U := jz.exp(T), T + U][2], 2),    # read again later: not at exp, but at +
    (lambda A, B: [T := A + B, V := T.T, jz.exp(T) @ V][2], 3),  # a view shows the storage to a later line
    (lambda A, B: [T := A + B, V := T.T, U := V @ B, jz.exp(T) + U][3], 2),  # the view's last line is past
    (lambda A, B: [T := A + B, T * T][1], 1),                    # the same operand twice
    (lambda A, B: [T := A + B, T + T.T][1], 1),                  # with its own transpose
    (lambda A, B: [T := A + B, T - T.T][1], 1),
    (lambda A, B: [T := A - B, r := jz.sum(T, 1), r + T][2], 2),  # the result is larger than r: broadcast
    (lambda A, B: [T := A * B, r := jz.sum(T, 0), T - r][2], 2),
    (lambda A, B: B - (A + 1), 1),                               # the right operand is the temporary
    (lambda A, B: B + (A * 2), 1),
    (lambda A, B: B * (A - 2), 1),
    (lambda A, B: 2 - (A + B), 1),
    (lambda A, B: 2 / (jz.abs(A) + 1), 1),
    (lambda A, B: (A + B) / 3, 1),
    (lambda A, B: (A + B) / (jz.abs(B) + 1), 3),                 # no division of two matrices in place
    (lambda A, B: jz.log(jz.abs(A) + 1), 2),                     # nor log
    (lambda A, B: [T := A + B, T.__iadd__(A), T * 2][2], 1),     # with the function's own in-place operation
    (lambda A, B: [T := A @ B.T, jz.relu(T) @ T][1], 3),         # read again by the next line: not taken
    (lambda A, B: jz.sin(jz.cos(jz.sqrt(jz.abs(jz.square(jz.relu(A + B)))))), 1),
    (lambda A, B: (A < B) * 2 + 1, 1),
]


class TestCompileReuse:
    """Temporaries that nothing reads again are taken over; the bits stay those of the function as written."""

    def test_the_rule(self, device, rng):
        a, b = values(rng, (6, 6)), values(rng, (6, 6))
        for number, (function, new) in enumerate(REUSE_CASES):
            compiled = jz.compile(function)
            for make_a in orders(a, device):
                for make_b in orders(b, device):
                    A, B = make_a(), make_b()
                    for _ in range(2):  # the recording, then the record
                        assert bits(compiled(A, B)) == bits(function(A, B)), number
                    assert bits([A, B]) == bits([make_a(), make_b()]), number
            record, = compiled._records.values()
            assert record.new == new, (number, record.source)

    def test_the_same_expression_twice(self, device, rng):
        a, b = values(rng, (6, 6)), values(rng, (6, 6))

        def argument_changed_between(A, B):
            T = A + B
            A += B
            return T - (A + B)

        def result_changed(A, B):
            T = A + B
            U = A + B
            T += 1
            return T - U

        def view_changed(A, B):
            T = A * B
            U = A * B
            V = U.T
            V -= B
            return T + U

        def one_of_two_zeros_changed(A, B):
            Z1, Z2 = jz.zeros_like(A), jz.zeros_like(A)
            Z1 += A
            return Z1 + Z2

        for function in (argument_changed_between, result_changed, view_changed, one_of_two_zeros_changed):
            compiled = jz.compile(function)
            for make_a in orders(a, device):
                for _ in range(2):
                    A1, A2, B = make_a(), make_a(), jz.array(b, device=device)
                    assert bits(compiled(A1, B)) == bits(function(A2, B)), function.__name__
                    assert bits(A1) == bits(A2)

        both = jz.compile(lambda A, B: (A + B, A + B))
        A, B = jz.array(a, device=device), jz.array(b, device=device)
        for _ in range(2):
            first, second = both(A, B)
            assert first is not second and bits(first) == bits(second) == bits(A + B)

        random = jz.compile(lambda A: jz.randn(*A.shape, device=A.device) - jz.randn(*A.shape, device=A.device))
        for _ in range(2):
            assert np.abs(random(A).numpy()).max() > 0

    def test_a_training_step_makes_five_matrices(self, device, rng):
        compiled = jz.compile(step)
        compiled(jz.ones(20, 3, device=device), jz.ones(64, 20, device=device), jz.ones(64, 3, device=device), 0.1)
        record, = compiled._records.values()
        assert (record.new, record.new_as_written) == (5, 13)

    def test_without_reuse_the_record_is_the_function(self, device, rng, monkeypatch):
        from juzhen import _compile
        monkeypatch.setattr(_compile, "REUSE", False)
        a, b = values(rng, (6, 6)), values(rng, (6, 6))
        A, B = jz.array(a, device=device), jz.array(b, device=device)
        for function, _ in REUSE_CASES:
            compiled = jz.compile(function)
            assert bits(compiled(A, B)) == bits(compiled(A, B)) == bits(function(A, B))
            record, = compiled._records.values()
            assert record.new == record.new_as_written and "_(" not in record.source

    def test_only_views_share_storage_with_an_operand(self, device, rng, monkeypatch):
        # the rule rests on this: every recorded operation but .T gives a matrix with storage of its own
        from juzhen import _compile
        monkeypatch.setattr(_compile, "CHECK_STORAGE", True)
        a, b, c = values(rng, (5, 4)), values(rng, (5, 4)), values(rng, (4, 5))
        for make_a in orders(a, device):
            for make_b in orders(b, device):
                jz.compile(everything)(make_a(), make_b(), jz.array(c, device=device), np.array([3, 0, 3]), 0.5)
        for function, _ in REUSE_CASES:
            jz.compile(function)(jz.array(values(rng, (6, 6)), device=device), plain(values(rng, (6, 6)), device))


def refused(function, *args, match):
    with pytest.raises(jz.CompileError, match=match):
        jz.compile(function)(*args)


OUTSIDE = {}


class TestCompileRefusals:
    """What a record could not repeat is an error while recording, not a wrong result later."""

    def test_reading_a_value(self, device, rng):
        import math
        A = jz.array(values(rng, (3, 3)), device=device)
        once = "once, while recording"
        refused(lambda A: float(jz.sum(A)), A, match=once)
        refused(lambda A: int(jz.sum(A)), A, match=once)
        refused(lambda A: round(jz.sum(A)), A, match=once)
        refused(lambda A: A.numpy(), A, match=once)
        refused(lambda A: jz.sum(A).item(), A, match=once)
        refused(lambda A: np.asarray(A), A, match=once)
        refused(lambda A: f"{jz.sum(A):.3f}", A, match=once)
        refused(lambda A: A * 2 if jz.sum(A) > 0 else A, A, match="`if` or `while` on a matrix")
        refused(lambda A: A[0, 0], A, match=r"A\[i:i\+1, j:j\+1\]")
        refused(lambda A: math.sqrt(jz.sum(A)), A, match=once)
        refused(lambda A: jz.argmax(A), A, match="jz.argmax gives numpy integers")
        refused(lambda A: jz.topk(A, 2), A, match="jz.topk gives numpy integers")

    def test_a_matrix_being_recorded_can_be_printed(self, device, rng):
        shown = []
        f = jz.compile(lambda A: shown.append(f"{A} {A!r} {str(A)}") or A)
        f(jz.zeros(2, 3, device=device))
        assert shown[0].count("2 x 3 matrix") == 3 and "being recorded by jz.compile" in shown[0]

    def test_a_float_argument_outside_arithmetic(self, device, rng):
        import math
        A = jz.array(values(rng, (3, 3)), device=device)
        refused(lambda A, s: A * math.sqrt(s), A, 2.0, match=r"x \*\* 0.5")
        refused(lambda A, s: A if s > 0 else -A, A, 2.0, match="Comparing a float argument")
        refused(lambda A, s: A * int(s), A, 2.0, match="pass a whole number")
        refused(lambda A, s: [A for _ in range(s)], A, 2.0, match="pass a whole number")
        refused(lambda A, s: A * (1 if s else 2), A, 2.0, match="`if` or `while` on a float argument")
        refused(lambda A, s: f"{s:.2f}", A, 2.0, match="Formatting a float argument")

    def test_matrices_that_are_not_arguments(self, device, rng):
        A = jz.array(values(rng, (3, 3)), device=device)
        G = jz.ones(3, 3, device=device)
        not_an_argument = "not an argument of the function"
        refused(lambda A: A + G, A, match=not_an_argument)
        refused(lambda A: G - A, A, match=not_an_argument)
        refused(lambda A: G @ A, A, match=not_an_argument)
        refused(lambda A: A < G, A, match=not_an_argument)
        refused(lambda A: jz.hstack([A, G]), A, match=not_an_argument)
        refused(lambda A: G, A, match="returned a matrix that is not one of its arguments")
        refused(lambda A: A + jz.array([[1.0, 2.0, 3.0]], device=A.device), A, match=r"jz.array\(\) inside")

        def in_place(A):
            A += G
            return A

        before = bits(A)
        refused(in_place, A, match=not_an_argument)
        assert bits(A) == before

    def test_an_index_computed_inside(self, device, rng):
        A = jz.array(values(rng, (4, 3)), device=device)
        refused(lambda A, idx: A[idx[:2]], A, np.array([0, 1, 2]), match="pass the positions")
        refused(lambda A: A[np.array([0, 1])], A, match="pass the positions")

    def test_a_value_kept_after_the_recording(self, device, rng):
        A = jz.array(values(rng, (3, 3)), device=device)

        @jz.compile
        def f(A, s):
            OUTSIDE["matrix"], OUTSIDE["number"] = A * 2, s * 2
            return A

        f(A, 0.5)
        ended = "after the recording has ended"
        with pytest.raises(jz.CompileError, match=ended):
            OUTSIDE["matrix"] + 1
        with pytest.raises(jz.CompileError, match=ended):
            OUTSIDE["matrix"].T
        with pytest.raises(jz.CompileError, match=ended):
            jz.exp(OUTSIDE["matrix"])
        with pytest.raises(jz.CompileError, match=ended):
            OUTSIDE["number"] + 1
        with pytest.raises(jz.CompileError, match=ended):
            f(OUTSIDE["matrix"], 0.5)
        with pytest.raises(jz.CompileError, match=ended):
            jz.compile(lambda B: B + OUTSIDE["matrix"])(A)

    def test_arguments_and_results_of_other_kinds(self, device, rng):
        A = jz.array(values(rng, (3, 3)), device=device)
        refused(lambda A, x: A, A, object(), match="was given a object")
        refused(lambda A: np.zeros(3), A, match="returned a ndarray")
        refused(lambda A: {1.5: A}, A, match="text or whole-number keys")

    def test_the_library_still_says_what_is_wrong(self, device, rng):
        A = jz.array(values(rng, (3, 3)), device=device)
        with pytest.raises(TypeError, match="make it a matrix first"):
            jz.compile(lambda A: A + np.ones((3, 3), np.float32))(A)
        with pytest.raises(ValueError, match="axis must be"):
            jz.compile(lambda A: jz.sum(A, 2.5))(A)
        with pytest.raises(TypeError, match="needs two matrices"):
            jz.compile(lambda A: A @ 2)(A)
