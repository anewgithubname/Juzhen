"""The five-line interface: an MNIST classifier with fit() and evaluate()."""

import sys

import juzhen as jz

device = sys.argv[1] if len(sys.argv) > 1 else ("gpu" if jz.has_gpu() else "cpu")
jz.manual_seed(1)

X, Y, X_test, Y_test = jz.datasets.mnist()
model = jz.Sequential(jz.Dense(256, activation="relu"), jz.Linear(10), device=device)
model.fit(X, Y, epochs=3, lr=1e-3)
print(model.evaluate(X_test, Y_test))
