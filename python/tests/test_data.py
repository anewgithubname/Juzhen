"""Data loaders, .matrix files and the MNIST data set."""

import os

import numpy as np
import pytest
from conftest import close

import juzhen as jz
from juzhen.data import DataLoader, MatrixFileLoader
from juzhen.matrixio import open_matrix, read_matrix, write_matrix

MNIST = os.path.join(jz._core.project_dir, "datasets", "MNIST")
has_mnist = os.path.exists(os.path.join(MNIST, "train_x.matrix")) or os.path.exists(os.path.join(MNIST, "dataset.zip"))


def test_dataloader_batches(device):
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    Y = np.arange(10, dtype=np.float32)
    loader = DataLoader(X, Y, batch_size=4, device=device)
    batches = list(loader)
    assert len(loader) == len(batches) == 2  # drop_last
    close(batches[1][0], X[4:8])
    close(batches[1][1], Y[4:8, None])
    assert batches[0][0].device == device
    assert len(DataLoader(X, batch_size=4, drop_last=False)) == 3
    assert list(DataLoader(X, batch_size=4, drop_last=False))[-1].shape == (2, 2)


def test_dataloader_shuffle_is_seeded():
    X = np.arange(100, dtype=np.float32).reshape(50, 2)

    def order():
        jz.manual_seed(3)
        return np.concatenate([b.numpy()[:, 0] for b in DataLoader(X, batch_size=10, shuffle=True)])

    first = order()
    np.testing.assert_array_equal(first, order())
    assert sorted(first) == list(X[:, 0]) and not np.array_equal(first, X[:, 0])


def test_dataloader_checks():
    with pytest.raises(ValueError, match="rows"):
        DataLoader(np.zeros((5, 2)), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="fewer than one batch"):
        DataLoader(np.zeros((5, 2)), batch_size=8)


def test_matrix_files(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_matrix(tmp_path / "a.matrix", a)
    np.testing.assert_array_equal(read_matrix(tmp_path / "a.matrix"), a)
    np.testing.assert_array_equal(open_matrix(tmp_path / "a.matrix"), a)
    # a transposed file, as Juzhen writes a matrix with the transpose flag set
    with open(tmp_path / "t.matrix", "wb") as f:
        np.array([4, 3, 1], np.int32).tofile(f)
        a.tofile(f)  # C-order 3 x 4 is column-major 4 x 3
    np.testing.assert_array_equal(read_matrix(tmp_path / "t.matrix"), a)
    with open(tmp_path / "bad.matrix", "wb") as f:
        np.array([4, 3, 0], np.int32).tofile(f)
    with pytest.raises(ValueError, match="truncated"):
        read_matrix(tmp_path / "bad.matrix")


@pytest.mark.skipif(not has_mnist, reason="datasets/MNIST is not present")
def test_mnist():
    X, Y, X_test, Y_test = jz.datasets.mnist()
    assert X.shape == (60000, 784) and Y.shape == (60000, 10)
    assert X_test.shape == (10000, 784) and Y_test.shape == (10000, 10)
    first = X[:2].numpy()
    assert 0 <= first.min() and first.max() <= 1 and first.max() > 0.5
    close(jz.sum(Y[:100], 1), np.ones((100, 1)))

    loader = MatrixFileLoader(MNIST, "test", batch_size=1000)
    X_i, labels = next(iter(loader))
    assert len(loader) == 10 and X_i.shape == (1000, 784)
    close(X_i / 255, X_test[:1000].numpy())
    close(jz.nn.one_hot(labels, 10), Y_test[:1000].numpy())
