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
                 "mul_": lambda A, B: A * B}
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
                    expected = like_a and not (name == "mul_" and mixed and cuda_gpu)
                    assert in_storage(lambda A: getattr(_core, name)(A, B), make_a) == expected, name
            for name in WITH_A_NUMBER:
                assert in_storage(lambda A: getattr(_core, name)(A, 1.5), make_a), name
            for name in OF_ONE_MATRIX:
                expected = name != "log_" and not (name == "exp_" and metal_gpu)
                assert in_storage(lambda A: getattr(_core, name)(A), make_a) == expected, name
