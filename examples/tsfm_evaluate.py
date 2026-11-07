"""
Score the backtest forecasts of examples/demo_tsfm.cu as in the Chronos paper
(Ansari et al., 2024): per dataset, MASE of the median forecast and the mean
weighted quantile loss (WQL) over quantiles 0.1..0.9, each divided by the
same metric of seasonal naive; datasets are aggregated by geometric mean.
Naive (last value) is shown as a second baseline.

MASE skips series whose in-sample seasonal error is zero (for every method);
NaN targets are ignored. Seasonal naive repeats the last season of the
context (missing values filled with the last observed one) and serves as
its own quantile forecast.

Usage: python examples/tsfm_evaluate.py DATA_DIR FORECAST_DIR [--split eval_zs]
"""

import argparse
import os
import struct
import sys

import numpy as np

TAUS = np.arange(1, 10) / 10.0


def read_series(path):
    with open(path, "rb") as fp:
        assert fp.read(4) == b"TSF1", path
        n, season, horizon, _ = struct.unpack("<4i", fp.read(16))
        off = np.frombuffer(fp.read(8 * (n + 1)), dtype="<i8")
        val = np.frombuffer(fp.read(4 * int(off[-1])), dtype="<f4")
    return [val[off[i]:off[i + 1]] for i in range(n)], season, horizon


def read_forecast(path):
    """Quantile forecasts (n, q, h). Files from before the mean head was
    dropped carry a mean row first, which is skipped."""
    with open(path, "rb") as fp:
        assert fp.read(4) == b"TSFC", path
        n, h, q, _ = struct.unpack("<4i", fp.read(16))
        v = np.frombuffer(fp.read(), dtype="<f4")
    rows = len(v) // (n * h) if n * h else q
    assert rows in (q, q + 1) and rows * n * h == len(v), path
    return v.reshape(n, rows, h)[:, rows - q:]


def ffill(x):
    x = x.astype(np.float64).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    x[: np.argmax(ok)] = x[np.argmax(ok)]
    return x


def baselines(ctx, h, m):
    c = ffill(ctx)
    naive = np.full(h, c[-1])
    if len(c) >= m:
        seasonal = c[len(c) - m + np.arange(h) % m]
    else:
        seasonal = naive
    return seasonal, naive


def score(series, season, horizon, quantiles, median):
    """Per-method (MASE, WQL) for the model and the two baselines."""
    mase = {"model": [], "seasonal_naive": [], "naive": []}
    wql_num = {k: 0.0 for k in mase}
    wql_den = 0.0
    for i, s in enumerate(series):
        ctx, y = s[:-horizon].astype(np.float64), s[-horizon:].astype(np.float64)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        sn, nv = baselines(ctx, horizon, season)
        preds = {
            "model": (median[i], quantiles[i]),
            "seasonal_naive": (sn, np.broadcast_to(sn, (len(TAUS), horizon))),
            "naive": (nv, np.broadcast_to(nv, (len(TAUS), horizon))),
        }
        lag = season if len(ctx) > season else 1
        d = np.abs(ctx[lag:] - ctx[:-lag])
        d = d[np.isfinite(d)]
        scale = d.mean() if len(d) else 0.0
        yo = y[ok]
        wql_den += np.abs(yo).sum()
        for k, (point, q) in preds.items():
            if scale > 0:
                mase[k].append(np.abs(point[ok] - yo).mean() / scale)
            u = yo[None, :] - q[:, ok]
            wql_num[k] += (2 * np.maximum(TAUS[:, None] * u, (TAUS[:, None] - 1) * u)).sum(axis=1).mean()
    out = {}
    for k in mase:
        out[k] = (np.mean(mase[k]) if mase[k] else np.nan,
                  wql_num[k] / wql_den if wql_den > 0 else np.nan)
    return out, len(mase["model"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("forecast_dir")
    ap.add_argument("--split", default="eval_zs")
    args = ap.parse_args()

    fdir = os.path.join(args.forecast_dir, args.split)
    names = sorted(f[:-4] for f in os.listdir(fdir) if f.endswith(".bin"))
    rel = {"model": ([], []), "naive": ([], [])}
    print(f"{'dataset':32s} {'series':>7s} {'H':>3s}   {'MASE':>7s} {'snaive':>7s} {'rel':>6s}"
          f"   {'WQL':>7s} {'snaive':>7s} {'rel':>6s}")
    for name in names:
        series, season, horizon = read_series(os.path.join(args.data_dir, args.split, name + ".bin"))
        q = read_forecast(os.path.join(fdir, name + ".bin"))
        assert len(series) == len(q), name
        res, used = score(series, season, horizon, q, q[:, len(TAUS) // 2])
        m, w = res["model"]
        sm, sw = res["seasonal_naive"]
        print(f"{name:32s} {len(series):7d} {horizon:3d}   {m:7.3f} {sm:7.3f} {m / sm:6.3f}"
              f"   {w:7.3f} {sw:7.3f} {w / sw:6.3f}")
        for k in rel:
            rel[k][0].append(res[k][0] / sm)
            rel[k][1].append(res[k][1] / sw)

    def gmean(v):
        v = np.array(v)
        v = v[np.isfinite(v) & (v > 0)]
        return np.exp(np.log(v).mean()) if len(v) else np.nan

    print(f"\ngeometric mean over {len(names)} datasets, relative to seasonal naive (lower is better):")
    for k, (rm, rw) in rel.items():
        print(f"  {k:15s} MASE {gmean(rm):.3f}   WQL {gmean(rw):.3f}")


if __name__ == "__main__":
    sys.exit(main())
