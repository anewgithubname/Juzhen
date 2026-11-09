"""Fit y = sin(x) on [-3, 3] with a small network and a hand-written training loop.

    python fit_sin.py [cpu|gpu]
"""

import sys
import time

import numpy as np

import juzhen as jz

device = sys.argv[1] if len(sys.argv) > 1 else ("gpu" if jz.has_gpu() else "cpu")
jz.manual_seed(0)

x = np.linspace(-3, 3, 1024, dtype=np.float32).reshape(-1, 1)
X = jz.array(x, device=device)            # 1024 samples, 1 feature each
Y = jz.sin(X)                             # the targets, computed by Juzhen

# The first layer gives its input size (1), so the layers exist right away
# and the optimizer can be made before the network has seen any data.
model = jz.Sequential(jz.Dense(1, 64), jz.Dense(64), jz.Linear(1), device=device)
opt = jz.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)

start = time.time()
for step in range(2001):
    loss = jz.nn.mse(model(X), Y)         # the value and its gradient
    opt.zero_grad()
    model.backward(loss.grad)
    opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}  mse {loss.item():.6f}")

pred = model.predict(X).numpy()
print(f"largest error: {np.abs(pred - np.sin(x)).max():.4f}")
print(f"{jz.backend()} backend, {device}: {time.time() - start:.2f} s")
