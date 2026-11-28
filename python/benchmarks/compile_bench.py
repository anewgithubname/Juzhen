"""What jz.compile gains: a few formulas as written and under @jz.compile, median milliseconds per call.

    python compile_bench.py [cpu] [gpu] [--quick]

A compiled function gives the bits of the function as written; the gain is the temporaries it does not
allocate ("new matrices": as written -> compiled). It is largest for chains of + - * on large matrices and
small where a matrix product or exp/tanh takes the time.
"""
import statistics
import sys
import time

import numpy as np

import juzhen as jz

QUICK = "--quick" in sys.argv
DEVICES = [d for d in sys.argv[1:] if d in ("cpu", "gpu")] or (["cpu"] + (["gpu"] if jz.has_gpu() else []))


def chain(A, B):
    return jz.sum((A - B) * (A - B))


def axpy(A, B):
    return A + 0.5 * B


def sigmoid(A, B):
    return 1 / (1 + jz.exp(-A))


def adam(W, G, M, V, lr):
    M_new = 0.9 * M + 0.1 * G
    V_new = 0.999 * V + 0.001 * (G * G)
    W -= lr * M_new / (jz.sqrt(V_new) + 1e-8)
    return M_new, V_new


def step(W, X, Y, lr):
    P = jz.tanh(X @ W)
    E = P - Y
    G = X.T @ (E * (1 - P * P)) / X.shape[0]
    W -= lr * G
    return jz.mean(E * E)


def rbf(X, Y):
    sx = jz.sum(X * X, 1)
    sy = jz.sum(Y * Y, 1).T
    D = sx + sy - 2 * (X @ Y.T)
    return jz.exp(-D / (2 * 1.5 * 1.5))


def measure(call):
    for _ in range(3):
        call()
    jz.synchronize()
    start = time.perf_counter()
    call()
    jz.synchronize()
    once = time.perf_counter() - start
    times = []
    for _ in range(int(min(2000, max(7, (0.15 if QUICK else 0.4) / max(once, 1e-7))))):
        start = time.perf_counter()
        call()
        jz.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def bits(result):
    return b"".join(m.numpy().tobytes() for m in (result if isinstance(result, tuple) else (result,)))


def compare(function, size, device, make):
    compiled = jz.compile(function)
    same = bits(function(*make())) == bits(compiled(*make())) == bits(compiled(*make()))
    args = make()
    # the two in turn, twice, so that a drift of the machine shows up in both
    runs = [[measure(lambda: f(*args)) for f in (function, compiled)] for _ in range(2)]
    written, fast = [min(a, b) for a, b in zip(*runs)]
    record, = compiled._records.values()
    print(f"{device:4} {function.__name__:8} {size:16} as written {written:9.4f}  compiled {fast:9.4f}  "
          f"x{written / fast:5.2f}  new matrices {record.new_as_written:2} -> {record.new:2}  "
          f"{'same bits' if same else 'DIFFERENT RESULT'}", flush=True)


rng = np.random.default_rng(0)


def randn(device, *shape):
    """Stored column by column, as the results of juzhen's own operations are."""
    M = jz.zeros(*shape, device=device)
    M += jz.array(rng.standard_normal(shape, dtype=np.float32), device=device)
    return M


print(f"backend {jz.backend()}, devices {DEVICES}", flush=True)
for device in DEVICES:
    for n in ([256, 1024] if QUICK else [256, 1024, 4096]):
        A, B = randn(device, n, n), randn(device, n, n)
        size = f"{n}x{n}"
        compare(chain, size, device, lambda: (A, B))
        compare(axpy, size, device, lambda: (A, B))
        compare(sigmoid, size, device, lambda: (A, B))
        W, V = randn(device, n, n), jz.square(randn(device, n, n))
        compare(adam, size, device, lambda: (W, A, B, V, 0.0))  # lr 0: the same work, and W stays put
    for n, d, c in ([(256, 784, 10)] if QUICK else [(256, 784, 10), (4096, 1024, 1024)]):
        X, Y, W = randn(device, n, d), randn(device, n, c), randn(device, d, c) * 0.01
        compare(step, f"{n}x{d}->{c}", device, lambda: (W, X, Y, 0.0))
    for n, d in ([(512, 64)] if QUICK else [(512, 64), (4096, 64)]):
        X, Y = randn(device, n, d), randn(device, n, d)
        compare(rbf, f"{n}x{n} d={d}", device, lambda: (X, Y))
