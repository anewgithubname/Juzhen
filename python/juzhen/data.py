"""Batches of training data.

DataLoader keeps the data in host memory and shuffles it each epoch;
MatrixFileLoader streams .matrix files from disk in order (for data sets too
big for memory). Both copy each batch into a juzhen matrix on the device.
"""

import os

import numpy as np

import juzhen as jz

__all__ = ["DataLoader", "MatrixFileLoader"]


def _host(a):
    """a as a 2-D float32 numpy array, one sample per row."""
    return jz._as_2d(a.numpy() if isinstance(a, jz.Matrix) else a)


class DataLoader:
    """Batches of (X, Y) rows, as juzhen matrices on device.

    One pass of ``for X_i, Y_i in loader`` is one epoch. With shuffle=True the
    rows are visited in a new random order each epoch (jz.manual_seed() makes
    it repeatable). drop_last=True skips a last, smaller batch; layers with a
    fixed batch size (convolutions, transformers) need that.
    """

    def __init__(self, X, Y=None, batch_size=32, shuffle=False, drop_last=True, device="cpu"):
        self.X = _host(X)
        self.Y = None if Y is None else _host(Y)
        if self.Y is not None and len(self.Y) != len(self.X):
            raise ValueError(f"X has {len(self.X)} rows but Y has {len(self.Y)}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if drop_last and len(self.X) < batch_size:
            raise ValueError(f"only {len(self.X)} samples, fewer than one batch of {batch_size}")
        self.batch_size, self.shuffle, self.drop_last, self.device = batch_size, shuffle, drop_last, device

    @property
    def features(self):
        return self.X.shape[1]

    def __len__(self):
        n = len(self.X) // self.batch_size
        return n if self.drop_last or len(self.X) % self.batch_size == 0 else n + 1

    def __iter__(self):
        n = len(self.X)
        order = jz._rng.permutation(n) if self.shuffle else None
        for b in range(len(self)):
            if order is None:
                rows = slice(b * self.batch_size, min((b + 1) * self.batch_size, n))
            else:
                rows = order[b * self.batch_size:(b + 1) * self.batch_size]
            X_i = jz.array(self.X[rows], device=self.device)
            if self.Y is None:
                yield X_i
            else:
                yield X_i, jz.array(self.Y[rows], device=self.device)


class MatrixFileLoader:
    """Batches read in order from ``{folder}/{split}_x.matrix`` and ``{split}_y.matrix``.

    The files are memory-mapped, so only the batches in use are read. The x
    file holds one sample per column (features x N), as Juzhen writes it; the
    y file holds integer labels (1 x N). Yields (X_i, labels_i): a matrix on
    device and a numpy int array (see nn.one_hot).
    """

    def __init__(self, folder, split, batch_size=32, drop_last=True, device="cpu"):
        x = jz.matrixio.open_matrix(os.path.join(folder, f"{split}_x.matrix"), np.float32)
        y = jz.matrixio.open_matrix(os.path.join(folder, f"{split}_y.matrix"), np.int32)
        self.X, self.labels = x.T, y.ravel()  # rows of X.T are samples
        if len(self.labels) != len(self.X):
            raise ValueError(f"{split}: {len(self.X)} samples but {len(self.labels)} labels")
        self.batch_size, self.drop_last, self.device = batch_size, drop_last, device

    def __len__(self):
        n = len(self.X) // self.batch_size
        return n if self.drop_last or len(self.X) % self.batch_size == 0 else n + 1

    def __iter__(self):
        for b in range(len(self)):
            rows = slice(b * self.batch_size, (b + 1) * self.batch_size)
            yield jz.array(self.X[rows], device=self.device), np.asarray(self.labels[rows])
