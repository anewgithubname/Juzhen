"""examples/demo_mnist.cu in Python: the same network, optimizer and loop.

Runs on the GPU when the build has one (--device cpu to override) and
prints the test misclassification rate every 1000 steps, like the C++ demo.
"""

import argparse
import time

import juzhen as jz
from juzhen import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="gpu" if jz.has_gpu() else "cpu")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
device = args.device

jz.manual_seed(args.seed)
t0 = time.perf_counter()

k, batch = 10, 32
X, Y, XT, YT = jz.datasets.mnist()        # one sample per row, Y one-hot
num_batches = X.shape[0] // batch
XT, YT = XT.to(device), YT.to(device)     # the test set goes to the device once

net = nn.Sequential(
    nn.Dense(784, 1024),                  # tanh, the C++ Layer<D>
    nn.Dense(1024, 128),
    nn.Linear(128, k),                    # no activation, LinearLayer<D>
    device=device)
opt = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0)  # the C++ demo's settings

for it in range(args.steps):
    b = it % num_batches
    X_i = X[b * batch:(b + 1) * batch].to(device)
    Y_i = Y[b * batch:(b + 1) * batch].to(device)

    opt.zero_grad()
    pred = net(X_i)
    loss = nn.softmax_cross_entropy(pred, Y_i)
    net.backward(loss.grad)
    opt.step()

    if it % 1000 == 0:
        err = nn.error_rate(net.predict(XT), YT).item()  # .item() waits for the GPU
        print(f"Misclassification Rate: {err:.4f}")

print(f"time: {(time.perf_counter() - t0) * 1000:.0f} ms")
