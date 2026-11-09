"""examples/knn.cu in Python: 7-nearest-neighbour classification of MNIST.

The squared distances between test and training images come from one
matrix product, |a|^2 + |b|^2 - 2 a.b; jz.topk picks the seven nearest
training images of each test image and they vote on the label. The test
set goes through in chunks so the distance matrix stays small.

    python knn.py [--device gpu] [--chunk 2000]
"""

import argparse
import time

import numpy as np

import juzhen as jz

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="gpu" if jz.has_gpu() else "cpu")
parser.add_argument("--k", type=int, default=7)
parser.add_argument("--chunk", type=int, default=2000, help="test images per distance matrix")
args = parser.parse_args()
device = args.device

print("K-Nearest Neighbour Prediction for MNIST Dataset:")
start = time.perf_counter()
X, Y, X_test, Y_test = jz.datasets.mnist()
labels, test_labels = Y.numpy().argmax(1), Y_test.numpy().argmax(1)
X = X.to(device)
print(f"X: {X.shape}, X_test: {X_test.shape}  (loaded in {time.perf_counter() - start:.1f} s)")

start = time.perf_counter()
train_norms = jz.sum(jz.square(X), 1)                  # 60000 x 1
predicted = []
for first in range(0, X_test.shape[0], args.chunk):
    A = X_test[first:first + args.chunk].to(device)     # chunk x 784
    # one column per test image: topk then reads each column in place
    distances = train_norms + jz.sum(jz.square(A), 1).T - 2 * (X @ A.T)   # 60000 x chunk
    _, nearest = jz.topk(distances, args.k, axis=0, largest=False)       # k x chunk positions
    votes = labels[nearest.numpy().astype(int).T]                         # chunk x k labels
    counts = np.zeros((len(votes), 10), dtype=int)
    np.add.at(counts, (np.arange(len(votes))[:, None], votes), 1)
    predicted.append(counts.argmax(1))                  # ties go to the smaller label
predicted = np.concatenate(predicted)

print(f"k-nearest neighbour: {time.perf_counter() - start:.1f} s")
print("----------------------------------------")
print(f"Misclassification Rate: {np.mean(predicted != test_labels):.4f}")
