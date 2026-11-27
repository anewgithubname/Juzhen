"""The matrix layer against numpy, on every device of the build."""

import numpy as np
import pytest
from conftest import close

import juzhen as jz


def pair(rng, device, shape=(4, 3)):
    a = rng.standard_normal(shape).astype(np.float32)
    return a, jz.array(a, device=device)


def test_helloworld_matches_numpy():
    a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    b = np.array([[.1, .2], [.3, .4], [.5, .6]], np.float32)
    C = jz.log(jz.exp(jz.array(a) @ jz.array(b)) + 1) / 5
    close(C, np.log(np.exp(a @ b) + 1) / 5)


class TestLayout:
    def test_numpy_roundtrip(self, device, rng):
        a, A = pair(rng, device)
        assert A.shape == (4, 3) and A.device == device
        close(A, a)
        assert A.numpy().flags["C_CONTIGUOUS"]  # a sample-first import stays C-order

    def test_transpose_shares_storage(self, device, rng):
        a, A = pair(rng, device)
        close(A.T, a.T)
        close(A.T.T, a)
        B = A.T
        B += 1  # in place on the view, like numpy
        close(A, a + 1)

    def test_vectors_scalars_and_images(self):
        assert jz.array(3.0).shape == (1, 1)
        assert jz.array([1, 2, 3]).shape == (3, 1)  # one value per sample
        images = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        close(jz.array(images), images.reshape(2, -1))

    def test_from_other_inputs(self, device):
        close(jz.array([[1, 2], [3, 4]], device=device), [[1, 2], [3, 4]])
        a = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
        close(jz.array(a, device=device), a)
        close(jz.array(jz.array(a), device=device), a)

    def test_array_protocol_and_item(self, device):
        A = jz.array([[1.5, 2], [3, 4]], device=device)
        np.testing.assert_array_equal(np.asarray(A), [[1.5, 2], [3, 4]])
        assert jz.array([[2.5]], device=device).item() == 2.5
        assert float(jz.array([[2.5]], device=device)) == 2.5
        with pytest.raises(ValueError):
            A.item()

    def test_repr(self):
        assert repr(jz.array([[1, 2]])) == "Matrix([[1., 2.]], device='cpu')"

    def test_isinstance(self, device):
        assert isinstance(jz.zeros(2, 2, device=device), jz.Matrix)
        assert not isinstance(np.zeros((2, 2)), jz.Matrix)

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            jz.array(np.zeros((0, 3)))

    def test_unknown_device(self):
        with pytest.raises(ValueError):
            jz.zeros(2, 2, device="tpu")
        if not jz.has_gpu():
            with pytest.raises(ValueError, match="no GPU"):
                jz.zeros(2, 2, device="gpu")


class TestArithmetic:
    def test_elementwise(self, device, rng):
        a, A = pair(rng, device)
        b, B = pair(rng, device)
        close(A + B, a + b)
        close(A - B, a - b)
        close(A * B, a * b)
        close(A / (B * B + 1), a / (b * b + 1))
        close(-A, -a)

    def test_scalars(self, device, rng):
        a, A = pair(rng, device)
        close(A + 2, a + 2)
        close(2 + A, 2 + a)
        close(A - 2, a - 2)
        close(2 - A, 2 - a)
        close(A * 3, a * 3)
        close(3 * A, 3 * a)
        close(A / 4, a / 4)
        close(1 / (A * A + 1), 1 / (a * a + 1))
        close(A ** 2, a ** 2)
        close((A * A) ** 0.5, np.abs(a))

    def test_broadcasting(self, device, rng):
        a, A = pair(rng, device)
        row, col = rng.standard_normal((1, 3)), rng.standard_normal((4, 1))
        close(A + jz.array(row, device=device), a + row)
        close(A * jz.array(col, device=device), a * col)
        close(jz.array(col, device=device) - A, col - a)
        close(A / jz.array([[2.0]], device=device), a / 2)
        close(jz.array(row, device=device) + jz.array(col, device=device), row + col)

    def test_shape_mismatch(self, device):
        with pytest.raises(ValueError, match="broadcast"):
            jz.zeros(2, 3, device=device) + jz.zeros(3, 2, device=device)
        with pytest.raises(ValueError, match="aligned"):
            jz.zeros(2, 3, device=device) @ jz.zeros(2, 3, device=device)

    def test_matmul_with_transposes(self, device, rng):
        a, A = pair(rng, device, (4, 3))
        b, B = pair(rng, device, (4, 5))
        close(A.T @ B, a.T @ b)
        close(B.T @ A, b.T @ a)
        close(A @ A.T, a @ a.T)

    def test_inplace(self, device, rng):
        a, A = pair(rng, device)
        A += 1
        A -= jz.array(np.ones((1, 3)), device=device)
        A *= 2
        A /= 4
        close(A, a / 2)
        B = A
        B += A
        assert B is A

    def test_inplace_cannot_grow(self, device):
        A = jz.zeros(1, 3, device=device)
        with pytest.raises(ValueError):
            A += jz.zeros(2, 3, device=device)


class TestFunctions:
    @pytest.mark.parametrize("name", ["exp", "tanh", "sin", "cos", "square", "relu"])
    def test_unary(self, device, rng, name):
        a, A = pair(rng, device)
        reference = {"relu": lambda x: np.maximum(x, 0)}.get(name) or getattr(np, name)
        close(getattr(jz, name)(A), reference(a))

    def test_log_sqrt_sigmoid(self, device, rng):
        a, A = pair(rng, device)
        close(jz.log(A * A + 1), np.log(a * a + 1))
        close(jz.sqrt(A * A), np.abs(a))
        close(jz.sigmoid(A), 1 / (1 + np.exp(-a)))

    def test_reductions(self, device, rng):
        a, A = pair(rng, device)
        close(jz.sum(A, 0), a.sum(0, keepdims=True))
        close(jz.sum(A, 1), a.sum(1, keepdims=True))
        close(jz.sum(A), [[a.sum()]])
        close(jz.mean(A, 0), a.mean(0, keepdims=True))
        close(jz.mean(A.T, 1), a.T.mean(1, keepdims=True))
        assert abs(jz.mean(A).item() - a.mean()) < 1e-5
        with pytest.raises(ValueError):
            jz.sum(A, 2)

    def test_stack_and_likes(self, device, rng):
        a, A = pair(rng, device)
        b, B = pair(rng, device, (3, 4))
        close(jz.hstack([A, A]), np.hstack([a, a]))
        close(jz.vstack([A, A.T.T]), np.vstack([a, a]))
        # mixed layouts: B.T has the other storage order
        close(jz.hstack([A, B.T, A @ B[:, :1]]), np.hstack([a, b.T, a @ b[:, :1]]))
        close(jz.vstack([A, B.T]), np.vstack([a, b.T]))
        with pytest.raises(ValueError, match="same number of rows"):
            jz.hstack([A, B])
        with pytest.raises(ValueError, match="at least one"):
            jz.vstack([])
        close(jz.zeros_like(A), np.zeros_like(a))
        close(jz.ones_like(A), np.ones_like(a))

    def test_max_min_abs_argmax(self, device, rng):
        a, A = pair(rng, device, (5, 7))
        for M, m in ((A, a), (A.T, a.T)):
            close(jz.max(M, 0), m.max(0, keepdims=True))
            close(jz.max(M, 1), m.max(1, keepdims=True))
            close(jz.min(M, 0), m.min(0, keepdims=True))
            close(jz.min(M, 1), m.min(1, keepdims=True))
            assert jz.argmax(M).tolist() == m.argmax(1).tolist()
            assert jz.argmax(M, 0).tolist() == m.argmax(0).tolist()
        assert jz.max(A).item() == pytest.approx(a.max())
        assert jz.min(A).item() == pytest.approx(a.min())
        close(jz.abs(A), np.abs(a))

    @pytest.mark.parametrize("largest", [True, False])
    def test_topk(self, device, rng, largest):
        a, A = pair(rng, device, (6, 9))
        order = np.argsort(-a if largest else a, axis=1, kind="stable")[:, :4]
        values, positions = jz.topk(A, 4, largest=largest)
        assert positions.tolist() == order.tolist()
        close(values, np.take_along_axis(a, order, 1))
        values, positions = jz.topk(A.T, 4, axis=0, largest=largest)
        assert positions.tolist() == order.T.tolist()
        close(values, np.take_along_axis(a, order, 1).T)
        assert A.numpy() == pytest.approx(a)  # the input is left alone
        with pytest.raises(ValueError, match="k must be"):
            jz.topk(A, 10)

    def test_softmax_logsumexp(self, device, rng):
        a, A = pair(rng, device, (4, 6))
        a = a * 30  # large enough that a naive exp() would overflow
        A = jz.array(a, device=device)
        def lse(x, axis):
            top = x.max(axis, keepdims=True)
            return np.log(np.exp(x - top).sum(axis, keepdims=True)) + top
        for axis in (0, 1):
            close(jz.logsumexp(A, axis), lse(a.astype(np.float64), axis), tol=1e-4)
            close(jz.softmax(A, axis), np.exp(a - lse(a.astype(np.float64), axis)), tol=1e-5)
        close(jz.sum(jz.softmax(A.T), 1), np.ones((6, 1)))


def keep_dims(key):
    """The numpy key that gives what juzhen gives: an int index keeps its dimension."""
    def one(k):
        return slice(k, k + 1 if k != -1 else None) if isinstance(k, int) else k
    return tuple(one(k) for k in key) if isinstance(key, tuple) else one(key)


class TestIndexing:
    @pytest.mark.parametrize("key", [
        (slice(1, 3), slice(None)), (slice(None), 1), 2, -1, slice(None, None, 2),
        ([3, 0], slice(None)), (slice(None), [2, 0]), (slice(1, None), slice(None, None, -1)),
        (np.array([1, 3]), slice(0, 2)),
    ])
    def test_matches_numpy(self, device, rng, key):
        a, A = pair(rng, device, (5, 4))
        close(A[key], a[keep_dims(key)])
        close(A.T[key], a.T[keep_dims(key)])

    def test_element_and_layout(self, device, rng):
        a, A = pair(rng, device, (5, 4))
        assert A[2, 3] == pytest.approx(a[2, 3])
        assert A[1:3].numpy().flags["C_CONTIGUOUS"]  # a batch of rows stays sample-first

    def test_errors(self, device):
        A = jz.zeros(3, 3, device=device)
        with pytest.raises(IndexError):
            A[3]
        with pytest.raises(IndexError):
            A[1:1]
        with pytest.raises(IndexError):
            A[0, 0, 0]


class TestDevices:
    def test_to(self, device, rng):
        a, A = pair(rng, device)
        assert A.to(device) is A
        close(A.to("cpu"), a)
        assert A.to("cpu").device == "cpu"

    @pytest.mark.skipif(not jz.has_gpu(), reason="needs a GPU build")
    def test_mixing_devices_says_so(self, rng):
        a, A = pair(rng, "cpu")
        G = A.to("gpu")
        for operation in (lambda: A + G, lambda: G - A, lambda: A * G, lambda: G / A, lambda: A @ G.T):
            with pytest.raises(ValueError, match="different devices"):
                operation()
        with pytest.raises(ValueError, match="different devices"):
            A += G
        close(A + 1, a + 1)  # numbers still work

    def test_random_and_seed(self, device):
        jz.manual_seed(7)
        a = jz.randn(50, 40, device=device).numpy()
        jz.manual_seed(7)
        np.testing.assert_array_equal(a, jz.randn(50, 40, device=device).numpy())
        assert abs(a.mean()) < 0.1 and abs(a.std() - 1) < 0.1
        u = jz.rand(50, 40, device=device).numpy()
        assert u.min() >= 0 and u.max() <= 1

    def test_empty_cache_keeps_live_matrices(self, device, rng):
        a, A = pair(rng, device)
        for _ in range(3):
            A + A
        jz.empty_cache()
        jz.synchronize()
        close(A + A, 2 * a)


class TestNumbersMethodsPositions:
    def test_one_by_one_matrix_is_a_number(self, device, rng):
        a, A = pair(rng, device, (4, 5))
        s, total = jz.sum(A), float(a.sum())
        assert f"{s:.2f}" == f"{total:.2f}" and "{:9.3e}".format(s) == f"{total:9.3e}"
        assert f"{s}" == str(s) == repr(s)  # without a format it prints as a matrix
        assert round(s, 2) == pytest.approx(round(total, 2)) and isinstance(round(s, 2), float)
        assert round(s) == round(total) and isinstance(round(s), int)
        assert int(s) == int(total) and float(s) == pytest.approx(total, abs=1e-5)
        assert (s > total - 1) and not (s > total + 1) and max(s, total + 1) == total + 1
        assert int(jz.sum(A > 0)) == int((a > 0).sum())
        assert f"{jz.mean(A):.4f}" == f"{a.mean():.4f}"

    def test_a_larger_matrix_is_not_a_number(self, device, rng):
        _, A = pair(rng, device, (4, 5))
        for use, what in ((lambda: f"{A:.2f}", r"the format '\.2f'"), (lambda: round(A), r"round\(\)"),
                          (lambda: round(A, 1), r"round\(\)"), (lambda: int(A), r"int\(\)"),
                          (lambda: float(A), r"float\(\)")):
            with pytest.raises(TypeError, match=what + r" needs one number, and this matrix is 4 x 5.*jz\.sum\(A\)"):
                use()
        assert f"{A}" == repr(A)

    def test_methods_are_the_functions(self, device, rng):
        a, A = pair(rng, device, (4, 5))
        for name in ("sum", "mean", "max", "min"):
            function, method = getattr(jz, name), getattr(A, name)
            close(method(), function(A))
            close(method(0), function(A, 0))
            close(method(axis=1), function(A, 1))
        for name in ("exp", "tanh", "sin", "cos", "square", "abs", "relu", "sigmoid", "softmax", "logsumexp"):
            close(getattr(A, name)(), getattr(jz, name)(A))
        close(A.abs().sqrt(), np.sqrt(np.abs(a)))
        close(A.abs().log(), np.log(np.abs(a)))
        close(A.softmax(axis=0), jz.softmax(A, 0))
        close(abs(A), np.abs(a))
        close(A - A.mean(axis=0), a - a.mean(0))  # the numpy habit
        assert A.sum(1).shape == (4, 1) and A.sum(1).device == device
        assert A.argmax().tolist() == a.argmax(1).tolist() and A.argmax(0).tolist() == a.argmax(0).tolist()
        values, positions = A.topk(2)
        close(values, jz.topk(A, 2)[0])
        assert positions.tolist() == jz.topk(A, 2)[1].tolist()
        assert "jz.sum(A, ...)" in A.sum.__doc__

    def test_numpy_functions_still_compute_in_numpy(self, device, rng):
        # np.sum(A) looks for A.sum and calls it with out=None: the answer stays numpy's
        a, A = pair(rng, device, (4, 5))
        for f in (np.sum, np.mean, np.max, np.min, np.argmax):
            for axis in (None, 0, 1):
                got, want = f(A, axis=axis), f(a, axis=axis)
                assert not isinstance(got, jz.Matrix) and np.shape(got) == np.shape(want)
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.exp(A), np.exp(a), rtol=1e-5)
        np.testing.assert_allclose(np.std(A), np.std(a), rtol=1e-5)

    def test_positions_are_numpy_integers(self, device, rng):
        scores = jz.array(rng.standard_normal((6, 3)), device=device)
        s = scores.numpy()
        y = s.argmax(1)
        y[0] = (y[0] + 1) % 3  # 5 of 6 right
        p = jz.argmax(scores)
        assert isinstance(p, np.ndarray) and p.dtype == np.int64 and p.shape == (6,)
        assert jz.argmax(scores, 0).shape == (3,)
        assert np.mean(p == y) == pytest.approx(5 / 6)          # accuracy, written the numpy way
        assert ["cat", "dog", "bird"][p[0]] == ["cat", "dog", "bird"][s[0].argmax()]
        close(scores[p], s[p])                                   # rows of a matrix
        close(jz.nn.softmax_cross_entropy(scores, p).value, jz.nn.softmax_cross_entropy(scores, s.argmax(1)).value)
        assert (jz.argmax(jz.nn.one_hot(y, 3, device=device)) == y).all()   # one-hot back to class numbers
        assert ((scores == jz.max(scores, 1)).numpy().argmax(1) == p).all()  # the same on the device
        values, positions = jz.topk(scores, 2, axis=0, largest=False)
        assert isinstance(values, jz.Matrix) and positions.dtype == np.int64 and positions.shape == (2, 3)
        assert y[positions].shape == (2, 3)                      # labels of the nearest, as in knn.py
        with pytest.raises(ValueError, match=r"axis 1 \(each row.*np\.argmax\(A\.numpy\(\)\)"):
            jz.argmax(scores, None)
        values, positions = jz.topk(jz.array([[np.nan, np.nan, 1.0]], device=device), 2)
        assert positions.tolist() == [[2, -1]] and np.isnan(values.numpy()[0, 1])


def plain(a, device):
    """a as a matrix stored column by column, as jz.randn and a matrix product give it."""
    M = jz.zeros(*a.shape, device=device)
    M += jz.array(a, device=device)
    assert a.shape[0] == 1 or a.shape[1] == 1 or M.numpy().flags["F_CONTIGUOUS"]
    return M


class TestStorageOrders:
    """A matrix from numpy is stored row by row, one made by juzhen column by column; every pairing must agree."""

    @pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (37, 53), (70, 70)])
    def test_elementwise_in_every_pairing(self, device, rng, shape):
        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal(shape).astype(np.float32) + 3
        for A in (jz.array(a, device=device), plain(a, device)):
            for B in (jz.array(b, device=device), plain(b, device)):
                close(A + B, a + b)
                close(A - B, a - b)
                close(A * B, a * b)
                close(A / B, a / b)
                close(A < B, a < b)
                C = A.copy()
                C += B
                close(C, a + b)
                C -= B
                close(C, a, tol=1e-4)
                close(A, a)   # the operands are left alone
                close(B, b)

    @pytest.mark.parametrize("n", [1, 3, 70])
    def test_in_place_with_its_own_transpose(self, device, rng, n):
        # the right side is the storage being written: this used to give wrong numbers on the CPU
        a = rng.standard_normal((n, n)).astype(np.float32)
        for make in (lambda: jz.array(a, device=device), lambda: plain(a, device)):
            A = make()
            A += A.T
            close(A, a + a.T)
            A = make()
            A -= A.T
            close(A, a - a.T)
            A = make()
            B = A.T
            B += A
            close(A, a + a.T)       # B is a view of A
            A = make()
            A += A
            close(A, 2 * a)
            A = make()
            close(A + A.T, a + a.T)
            close(A * A.T, a * a.T)
            close(A, a)

    def test_rows_and_columns_by_number(self, device, rng):
        a = rng.standard_normal((50, 23)).astype(np.float32)
        rows = np.array([49, 0, 7, 7, -1, 12])
        cols = [22, 3, 3, 0]
        for A, expected in ((jz.array(a, device=device), a), (plain(a, device), a),
                            (jz.array(a.T.copy(), device=device).T, a)):
            for picked, want in ((A[rows], expected[rows]), (A[rows.tolist()], expected[rows]),
                                 (A[:, cols], expected[:, cols]), (A[rows, cols], expected[np.ix_(rows, cols)]),
                                 (A[::7], expected[::7]), (A[::-1, ::3], expected[::-1, ::3]),
                                 (A[5, cols], expected[5:6, cols]), (A[rows, 2], expected[rows, 2:3]),
                                 (A.T[cols], expected.T[cols]), (A.T[:, rows], expected.T[:, rows])):
                assert picked.device == device
                close(picked, want)
            close(A, expected)
        big = rng.standard_normal((3000, 40)).astype(np.float32)
        order = rng.permutation(3000)
        close(jz.array(big, device=device)[order], big[order])
