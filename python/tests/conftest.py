import numpy as np
import pytest

import juzhen as jz

DEVICES = ["cpu"] + (["gpu"] if jz.has_gpu() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def close(m, expected, tol=1e-5):
    """Compare a juzhen matrix with a numpy array (float32 tolerances)."""
    got = m.numpy() if isinstance(m, jz.Matrix) else np.asarray(m)
    expected = np.asarray(expected, dtype=np.float32)
    assert got.shape == expected.shape, f"shape {got.shape} != {expected.shape}"
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)
