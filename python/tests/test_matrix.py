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
            close(jz.argmax(M), m.argmax(1)[:, None])
            close(jz.argmax(M, 0), m.argmax(0)[None, :])
        assert jz.max(A).item() == pytest.approx(a.max())
        assert jz.min(A).item() == pytest.approx(a.min())
        close(jz.abs(A), np.abs(a))

    @pytest.mark.parametrize("largest", [True, False])
    def test_topk(self, device, rng, largest):
        a, A = pair(rng, device, (6, 9))
        order = np.argsort(-a if largest else a, axis=1, kind="stable")[:, :4]
        values, positions = jz.topk(A, 4, largest=largest)
        close(positions, order)
        close(values, np.take_along_axis(a, order, 1))
        values, positions = jz.topk(A.T, 4, axis=0, largest=largest)
        close(positions, order.T)
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
