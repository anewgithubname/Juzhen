"""examples/demo_rectified.cu in Python: rectified flow from a Gaussian to a ring.

A velocity network v(x, t) is trained so that moving Gaussian samples along
it from t = 0 to 1 lands them on a ring of ten small clusters (radius 2).
Training regresses v(x_t, t) on x1 - x0 at x_t = (1 - t) x0 + t x1; sampling
integrates dx/dt = v(x, t) with Euler steps. The C++ demo animates this in a
terminal dashboard; this prints the loss and draws the final samples.

    python rectified_flow.py [--device gpu] [--iters 8000]
"""

import argparse
import math
import time

import numpy as np

import juzhen as jz
from juzhen import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="gpu" if jz.has_gpu() else "cpu")
parser.add_argument("--iters", type=int, default=8000)
parser.add_argument("--euler-steps", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
device = args.device

jz.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
batch, shown, radius = 256, 600, 2.0


def sample_ring(n, modes=10, sigma=0.1):
    """n points around a circle of the given radius, in `modes` small clusters."""
    theta = 2 * math.pi * rng.integers(0, modes, n) / modes
    centres = radius * np.stack([np.cos(theta), np.sin(theta)], 1)
    return jz.array(centres + sigma * rng.standard_normal((n, 2)), device=device)


def generate(n, steps):
    """Euler integration of dx/dt = v(x, t) from Gaussian noise."""
    z = jz.randn(n, 2, device=device)
    for s in range(steps):
        t = jz.ones(n, 1, device=device) * (s / steps)
        z = z + model.predict(jz.hstack([z, t])) * (1 / steps)
    return z.numpy()


def radius_mse(points):
    return float(np.mean((np.linalg.norm(points, axis=1) - radius) ** 2))


def scatter(generated, reference, width=64, height=26, extent=4.5):
    """A text plot: o generated, . reference, @ both."""
    grid = [[" "] * width for _ in range(height)]
    for points, mark in ((reference, "."), (generated, "o")):
        cols = np.clip(((points[:, 0] + extent) / (2 * extent) * (width - 1)).astype(int), 0, width - 1)
        rows = np.clip(((extent - points[:, 1]) / (2 * extent) * (height - 1)).astype(int), 0, height - 1)
        for r, c in zip(rows, cols):
            grid[r][c] = "@" if grid[r][c] not in (" ", mark) else mark
    return "\n".join("|" + "".join(row) + "|" for row in grid)


# v(x, t): input (x, y, t), output the velocity (vx, vy)
model = nn.Sequential(
    nn.Dense(3, 128, activation="relu"),
    nn.Dense(128, activation="relu"),
    nn.Dense(128, activation="relu"),
    nn.Linear(2),
    device=device)
# He initialization. The default (weights from N(0, 0.001^2), as in C++)
# makes this deep ReLU network output almost exactly zero, and whether
# training ever leaves that point depends on luck.
for p in model.parameters():
    fan_in = p.shape[1]
    scale = np.sqrt(2 / fan_in) if p.name.endswith("weight") else 0.0
    p.assign(jz.array(scale * rng.standard_normal(p.shape), device=device))
opt =optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

start = time.perf_counter()
ema = None
for i in range(args.iters):
    x0 = jz.randn(batch, 2, device=device)
    x1 = sample_ring(batch)
    t = jz.rand(batch, 1, device=device)
    xt = x0 * (1 - t) + x1 * t                     # t broadcasts over both columns

    opt.zero_grad()
    loss = nn.mse(model(jz.hstack([xt, t])), x1 - x0)
    model.backward(loss.grad)
    opt.step()

    value = loss.item()
    ema = value if ema is None else 0.95 * ema + 0.05 * value
    if i % 500 == 0 or i == args.iters - 1:
        print(f"iter {i:5d}  loss {value:.4f}  ema {ema:.4f}  ({time.perf_counter() - start:.1f} s)")

generated = generate(shown, args.euler_steps)
reference = sample_ring(shown).numpy()
print(f"\nODE samples (o) after {args.euler_steps} Euler steps vs reference (.):")
print(scatter(generated, reference))
print(f"[done] gen_radius_mse={radius_mse(generated):.6f} ref_radius_mse={radius_mse(reference):.6f}"
      f"  ({time.perf_counter() - start:.1f} s)")
