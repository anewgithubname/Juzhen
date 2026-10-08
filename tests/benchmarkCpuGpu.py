"""Compare CPU/ROCm benchmarkCpuGpu builds; verify outputs before reporting speedups.

Example: python tests/benchmarkCpuGpu.py --cpu build_health_cpu/benchmarkCpuGpu.exe
         --gpu build_health_rocm/benchmarkCpuGpu.exe --output res/cpu_gpu_benchmark
Requires NumPy. Executable environments (including HIP DLL search paths) must be configured.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import numpy as np


def read_matrix(path):
    with path.open("rb") as f:
        shape = np.fromfile(f, dtype="<i4", count=2)
        data = np.fromfile(f, dtype="<f4")
    if len(shape) != 2 or np.prod(shape) != data.size:
        raise ValueError(f"Malformed result: {path}")
    return data.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu-threads", default="default,1,4,16")
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.warmup) < 1:
        parser.error("rounds, iterations and warmup must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = args.cpu_threads.split(",")
    jobs = ["CPU_" + setting for setting in settings] + ["ROCm"]
    runs = {job: [] for job in jobs}
    parity = []
    for index in range(args.rounds):
        # Rotate order to avoid always measuring a backend at the same point.
        order = jobs[index % len(jobs):] + jobs[:index % len(jobs)]
        for job in order:
            directory = out / f"round_{index}" / job
            directory.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(JUZHEN_BENCH_WARMUP=str(args.warmup),
                       JUZHEN_BENCH_ITERS=str(args.iterations), JUZHEN_BENCH_OUTPUT=str(directory))
            if job.startswith("CPU_"):
                threads = job.removeprefix("CPU_")
                if threads == "default":
                    env.pop("OPENBLAS_NUM_THREADS", None)
                    env.pop("OMP_NUM_THREADS", None)
                else:
                    env["OPENBLAS_NUM_THREADS"] = threads
                    env["OMP_NUM_THREADS"] = threads
            exe = Path(args.gpu if job == "ROCm" else args.cpu).resolve()
            result = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=300)
            (directory / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{job} exited {result.returncode}; see {directory / 'run.log'}")
            rows = {}
            for line in result.stdout.splitlines():
                if line.startswith("RESULT "):
                    fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
                    rows[fields["case"]] = {key: float(fields[key]) for key in ("mean_ms", "p50_ms", "p95_ms")}
            if len(rows) != 10:
                raise RuntimeError(f"Expected 10 workloads, got {len(rows)} for {job}")
            metadata = next(line for line in result.stdout.splitlines() if line.startswith("META "))
            runs[job].append({"metadata": metadata, "cases": rows})
            print(f"round {index+1} {job}: small train={rows['tf_small_train']['mean_ms']:.3f} ms, "
                  f"large train={rows['tf_large_train']['mean_ms']:.3f} ms; {metadata}", flush=True)
        gpu_dir = out / f"round_{index}" / "ROCm"
        for job in jobs:
            if job == "ROCm":
                continue
            cpu_dir = out / f"round_{index}" / job
            gpu_files = sorted(gpu_dir.glob("*.bin"))
            if not gpu_files or {p.name for p in gpu_files} != {p.name for p in cpu_dir.glob('*.bin')}:
                raise RuntimeError("Output sets do not match")
            worst = 0.0
            for path in gpu_files:
                ref, got = read_matrix(cpu_dir / path.name), read_matrix(path)
                error = float(np.linalg.norm(got-ref) / max(np.linalg.norm(ref), 1e-12))
                if not np.isfinite(error) or error > 1e-4:
                    raise RuntimeError(f"Parity failed: {job} {path.name} rel_l2={error}")
                worst = max(worst, error)
            parity.append({"round": index, "cpu": job, "matrices": len(gpu_files), "max_rel_l2": worst})
    medians = {job: {case: statistics.median(run['cases'][case]['mean_ms'] for run in records)
                    for case in records[0]['cases']} for job, records in runs.items()}
    comparisons = {}
    for case, gpu_ms in medians['ROCm'].items():
        best = min((job for job in jobs if job != 'ROCm'), key=lambda job: medians[job][case])
        comparisons[case] = {'gpu_ms': gpu_ms, 'best_cpu': best, 'best_cpu_ms': medians[best][case],
                             'speedup_vs_best_cpu': medians[best][case] / gpu_ms}
        if 'CPU_default' in medians:
            comparisons[case]['default_cpu_ms'] = medians['CPU_default'][case]
            comparisons[case]['speedup_vs_default_cpu'] = medians['CPU_default'][case] / gpu_ms
    report = {'settings': vars(args), 'runs': runs, 'parity': parity, 'medians_ms': medians,
              'comparisons': comparisons}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
