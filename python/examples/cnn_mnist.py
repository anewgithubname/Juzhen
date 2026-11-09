"""examples/demo_cnn_mnist.cu in Python: a small CNN on MNIST.

Three 3x3 convolutions (16, 32 and 64 channels; the last two with stride 2),
a 256-unit ReLU layer and a linear output, trained with AdamW on batches of
64. The C++ demo draws a terminal dashboard; this prints the same numbers:
the loss, the predictions for eight test images, and the test accuracy
after each epoch.

    python cnn_mnist.py [--device gpu] [--epochs 10]

The convolutions run on the GPU, or on the CPU in a CPU-only build.
"""

import argparse
import time

import numpy as np

import juzhen as jz
from juzhen import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="gpu" if jz.has_gpu() else "cpu")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=43)
args = parser.parse_args()
device = args.device

jz.manual_seed(args.seed)
start = time.perf_counter()

batch, classes = 64, 10
X, Y, X_test, Y_test = jz.datasets.mnist()   # pixels in [0, 1], labels one-hot
steps_per_epoch = X.shape[0] // batch
test_batches = X_test.shape[0] // batch

# 1 x 28 x 28 -> 16 x 28 x 28 -> 32 x 14 x 14 -> 64 x 7 x 7 -> 256 -> 10
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, image_size=28, batch_size=batch, padding=1),
    nn.Conv2d(16, 32, 3, image_size=28, batch_size=batch, padding=1, stride=2),
    nn.Conv2d(32, 64, 3, image_size=14, batch_size=batch, padding=1, stride=2),
    nn.Dense(256, activation="relu"),
    nn.Linear(classes),
    device=device)

# the C++ demo's initialization: every weight and bias from N(0, 0.01^2)
init = np.random.default_rng(args.seed)
for p in model.parameters():
    p.assign(jz.array(0.01 * init.standard_normal(p.shape), device=device))
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

# eight test images whose predictions are shown while training
shown = X_test[:batch].to(device)
shown_labels = Y_test[:8].numpy().argmax(1)
print(model)
print(f"device {device}, {steps_per_epoch} steps per epoch\n")

losses = []
for epoch in range(args.epochs):
    for step in range(steps_per_epoch):
        rows = slice(step * batch, (step + 1) * batch)
        X_i, Y_i = X[rows].to(device), Y[rows].to(device)

        opt.zero_grad()
        loss = nn.softmax_cross_entropy(model(X_i), Y_i)
        model.backward(loss.grad)
        opt.step()
        losses.append(loss.item())

        global_step = epoch * steps_per_epoch + step
        if global_step % 100 == 0:
            predicted = jz.argmax(model.predict(shown)).numpy()[:8, 0].astype(int)
            marks = " ".join(f"{t}{'=' if t == p else '!'}{p}" for t, p in zip(shown_labels, predicted))
            print(f"epoch {epoch + 1} step {global_step:5d}  loss {losses[-1]:.4f}  "
                  f"true=predicted: {marks}")

    errors = 0.0
    for b in range(test_batches):
        rows = slice(b * batch, (b + 1) * batch)
        errors += nn.error_rate(model.predict(X_test[rows].to(device)), Y_test[rows].to(device)).item()
    print(f"--- epoch {epoch + 1}: test accuracy {100 * (1 - errors / test_batches):.2f}%  "
          f"({time.perf_counter() - start:.1f} s)")

print(f"total time: {time.perf_counter() - start:.1f} s")
