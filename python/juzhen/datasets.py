"""Small data sets, as juzhen matrices with one sample per row."""

import os
import zipfile

import numpy as np

import juzhen as jz

__all__ = ["mnist"]


def _mnist_folder(path):
    if path is None:
        path = os.path.join(jz._core.project_dir, "datasets", "MNIST")
    if not os.path.exists(os.path.join(path, "train_x.matrix")):
        archive = os.path.join(path, "dataset.zip")
        if not os.path.exists(archive):
            raise FileNotFoundError(f"no MNIST data in {path}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(path)
    return path


def mnist(path=None):
    """MNIST as (X, Y, X_test, Y_test), the data of examples/demo_mnist.cu.

    X is 60000 x 784 with pixels in [0, 1], Y is 60000 x 10 one-hot; the test
    set has 10000 rows. The matrices are on the CPU; path defaults to
    datasets/MNIST in the Juzhen source tree.
    """
    folder = _mnist_folder(path)
    read = jz.matrixio.read_matrix

    def split(name):
        X = read(os.path.join(folder, f"{name}_x.matrix"), np.float32).T / np.float32(255)
        labels = read(os.path.join(folder, f"{name}_y.matrix"), np.int32).ravel()
        return jz.array(X), jz.nn.one_hot(labels, 10)

    X, Y = split("train")
    X_test, Y_test = split("test")
    return X, Y, X_test, Y_test
