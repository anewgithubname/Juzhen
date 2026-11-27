"""Basic linear algebra, Juzhen against PyTorch: the same operations on the same numbers.

    python linalg_vs_torch.py all                      # both libraries, every device both have
    python linalg_vs_torch.py all --devices cpu --sizes 256 1024 4096 --threads 1
    python linalg_vs_torch.py run --lib juzhen --device gpu --out juzhen_gpu.json
    python linalg_vs_torch.py compare juzhen_gpu.json torch_gpu.json

Each library runs in a process of its own, so that their thread pools do not
disturb each other. All matrices are n x n float32 and come from the same numpy
arrays; the result of every operation is reduced to two numbers (its sum and the
sum of its absolute values) and the comparison says when the libraries disagree.

A time is the median over the repetitions of one call, in milliseconds. On the
GPU the clock stops after the device has finished (jz.synchronize(),
torch.cuda.synchronize()). Nothing is changed in either library's settings
unless --threads is given.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

OPERATIONS = ["A @ B", "A.T @ B", "A @ v", "A + B", "A * B", "A / B", "2.5 * A", "A + 0.5 * B", "A.T + B",
              "A += B", "exp(A)", "sum(A)", "sum(A, axis=0)", "max(A, axis=1)", "sum(A * A)", "A[n/4 rows]"]


def inputs(n):
    import numpy as np
    rng = np.random.default_rng(n)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = np.abs(rng.standard_normal((n, n), dtype=np.float32)) + 0.5   # safe to divide by
    v = rng.standard_normal((n, 1), dtype=np.float32)
    rows = rng.permutation(n)[: n // 4]
    return a, b, v, rows


def juzhen_setup(device, n):
    import juzhen as jz
    a, b, v, rows = inputs(n)
    A, B, V = (jz.array(x, device=device) for x in (a, b, v))
    C = A.copy()

    def add_in_place():
        nonlocal C
        C += B

    def fresh_in_place():
        D = A.copy()
        D += B
        return D

    calls = {
        "A @ B": lambda: A @ B, "A.T @ B": lambda: A.T @ B, "A @ v": lambda: A @ V,
        "A + B": lambda: A + B, "A * B": lambda: A * B, "A / B": lambda: A / B,
        "2.5 * A": lambda: 2.5 * A, "A + 0.5 * B": lambda: A + 0.5 * B, "A.T + B": lambda: A.T + B,
        "A += B": add_in_place, "exp(A)": lambda: jz.exp(A),
        "sum(A)": lambda: jz.sum(A), "sum(A, axis=0)": lambda: jz.sum(A, 0), "max(A, axis=1)": lambda: jz.max(A, 1),
        "sum(A * A)": lambda: jz.sum(jz.square(A)), "A[n/4 rows]": lambda: A[rows],
    }
    sync = jz.synchronize if device == "gpu" else (lambda: None)
    return calls, {"A += B": fresh_in_place}, sync, lambda r: r.numpy()


def torch_setup(device, n):
    import torch
    where = "cuda" if device == "gpu" else "cpu"
    a, b, v, rows = inputs(n)
    A, B, V = (torch.from_numpy(x).to(where) for x in (a, b, v))
    index = torch.from_numpy(rows).to(where)
    C = A.clone()

    def add_in_place():
        nonlocal C
        C += B

    def fresh_in_place():
        D = A.clone()
        D += B
        return D

    calls = {
        "A @ B": lambda: A @ B, "A.T @ B": lambda: A.T @ B, "A @ v": lambda: A @ V,
        "A + B": lambda: A + B, "A * B": lambda: A * B, "A / B": lambda: A / B,
        "2.5 * A": lambda: 2.5 * A, "A + 0.5 * B": lambda: A + 0.5 * B, "A.T + B": lambda: A.T + B,
        "A += B": add_in_place, "exp(A)": lambda: torch.exp(A),
        "sum(A)": lambda: torch.sum(A), "sum(A, axis=0)": lambda: torch.sum(A, 0, keepdim=True),
        "max(A, axis=1)": lambda: torch.amax(A, 1, keepdim=True),
        "sum(A * A)": lambda: torch.sum(torch.square(A)), "A[n/4 rows]": lambda: A[index],
    }
    sync = torch.cuda.synchronize if device == "gpu" else (lambda: None)
    return calls, {"A += B": fresh_in_place}, sync, lambda r: r.detach().cpu().numpy()


def measure(call, sync, budget):
    """Median and smallest time of one call in milliseconds, and the number of repetitions."""
    for _ in range(3):
        call()
    sync()
    start = time.perf_counter()
    call()
    sync()
    once = time.perf_counter() - start
    repetitions = int(min(3000, max(7, budget / max(once, 1e-7))))
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        sync()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3, min(times) * 1e3, repetitions


def environment(lib, device):
    info = {"lib": lib, "device": device, "python": platform.python_version(), "system": platform.system(),
            "logical_cpus": os.cpu_count(),
            "thread_variables": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                            "MKL_NUM_THREADS") if k in os.environ}}
    if lib == "torch":
        import torch
        info["version"] = torch.__version__
        info["threads"] = torch.get_num_threads()
        if device == "gpu":
            info["gpu"] = torch.cuda.get_device_name(0)
            info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    else:
        import juzhen as jz
        info["backend"] = jz.backend()
        info["tf32_matmul"] = os.environ.get("NVIDIA_TF32", "0") not in ("", "0")
    return info


def run(lib, device, sizes, budget, threads, out):
    if threads:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(threads)
    import numpy as np
    if lib == "torch":
        import torch
        if threads:
            torch.set_num_threads(threads)
        if device == "gpu" and not torch.cuda.is_available():
            sys.exit("this PyTorch has no CUDA device")
        setup = torch_setup
    else:
        import juzhen as jz
        if device == "gpu" and not jz.has_gpu():
            sys.exit("this Juzhen build has no GPU")
        setup = juzhen_setup
    report = {"environment": environment(lib, device), "results": []}
    for n in sizes:
        calls, fresh, sync, to_numpy = setup(device, n)
        for name in OPERATIONS:
            value = to_numpy(fresh.get(name, calls[name])()).astype(np.float64)
            median, best, repetitions = measure(calls[name], sync, budget)
            report["results"].append({"operation": name, "n": n, "median_ms": median, "min_ms": best,
                                      "repetitions": repetitions, "shape": list(np.atleast_2d(value).shape),
                                      "sum": float(value.sum()), "abs_sum": float(np.abs(value).sum())})
            print(f"{lib:>6} {device} n={n:<5} {name:<16} {median:10.4f} ms  ({repetitions} repetitions)", flush=True)
        del calls, fresh
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


def same(x, y):
    scale = max(abs(x["abs_sum"]), abs(y["abs_sum"]), 1e-30)
    return (x["shape"] == y["shape"] and abs(x["abs_sum"] - y["abs_sum"]) <= 1e-3 * scale
            and abs(x["sum"] - y["sum"]) <= 1e-3 * scale)


def compare(juzhen_file, torch_file):
    ours, theirs = (json.load(open(p)) for p in (juzhen_file, torch_file))
    print(f"\n## {ours['environment']['device']}")
    print("juzhen:", json.dumps(ours["environment"]))
    print("torch: ", json.dumps(theirs["environment"]))
    table = {(r["operation"], r["n"]): r for r in theirs["results"]}
    sizes = sorted({r["n"] for r in ours["results"]})
    print("\nmedian milliseconds, juzhen | torch; the last number is torch / juzhen (above 1: juzhen is faster)\n")
    print("| operation | " + " | ".join(f"n = {n}" for n in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    differ = []
    for name in OPERATIONS:
        cells = []
        for n in sizes:
            x = next(r for r in ours["results"] if r["operation"] == name and r["n"] == n)
            y = table[(name, n)]
            if not same(x, y):
                differ.append((name, n, x, y))
            cells.append(f"{x['median_ms']:.3f} \\| {y['median_ms']:.3f} \\| {y['median_ms'] / x['median_ms']:.2f}")
        print(f"| `{name}` | " + " | ".join(cells) + " |")
    if differ:
        print("\nRESULTS DIFFER:")
        for name, n, x, y in differ:
            print(f"  {name} n={n}: juzhen shape {x['shape']} sum {x['sum']:.6g} |sum| {x['abs_sum']:.6g};"
                  f" torch shape {y['shape']} sum {y['sum']:.6g} |sum| {y['abs_sum']:.6g}")
    else:
        print("\nevery operation gave the same result in both libraries (sum and sum of absolute values to 1e-3)")


def run_all(devices, sizes, budget, threads, folder):
    os.makedirs(folder, exist_ok=True)
    for device in devices:
        files = []
        for lib in ("juzhen", "torch"):
            out = os.path.join(folder, f"{lib}_{device}{'_t' + str(threads) if threads else ''}.json")
            command = [sys.executable, os.path.abspath(__file__), "run", "--lib", lib, "--device", device, "--out", out,
                       "--budget", str(budget), "--sizes", *map(str, sizes)]
            if threads:
                command += ["--threads", str(threads)]
            if subprocess.run(command).returncode != 0:
                print(f"skipping {device}: {lib} could not run there")
                break
            files.append(out)
        if len(files) == 2:
            compare(*files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "all"):
        p = commands.add_parser(name)
        p.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
        p.add_argument("--budget", type=float, default=0.5, help="seconds of timing for each operation and size")
        p.add_argument("--threads", type=int, default=0, help="CPU threads for both libraries (0: their defaults)")
        if name == "run":
            p.add_argument("--lib", choices=["juzhen", "torch"], required=True)
            p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
            p.add_argument("--folder", default="linalg_vs_torch_results")
    p = commands.add_parser("compare")
    p.add_argument("juzhen_file")
    p.add_argument("torch_file")
    args = parser.parse_args()
    if args.command == "run":
        run(args.lib, args.device, args.sizes, args.budget, args.threads, args.out)
    elif args.command == "all":
        run_all(args.devices, args.sizes, args.budget, args.threads, args.folder)
    else:
        compare(args.juzhen_file, args.torch_file)
