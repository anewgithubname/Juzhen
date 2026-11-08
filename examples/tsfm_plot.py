"""
Plot forecasts of examples/demo_tsfm.cu next to what actually happened: one
panel per dataset, each with a series drawn at random. The panel shows the
end of the context, the true continuation, the median forecast and the 10-90%
quantile band.

Usage: python examples/tsfm_plot.py DATA_DIR FORECAST_DIR [--split eval_zs]
           [--datasets ETTh nn5 ...] [--seed 0] [--out tsfm_forecasts.png]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsfm_evaluate import read_forecast, read_series  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"
FORECAST = "#2a78d6"

DEFAULT = ["ETTh", "monash_traffic", "nn5", "monash_tourism_monthly", "monash_hospital", "m4_quarterly"]


def pick(series, horizon, rng):
    """A random series with a fully observed, non-constant target."""
    for i in rng.permutation(len(series)):
        y = series[i][-horizon:]
        if np.isfinite(y).all() and np.ptp(y) > 0:
            return int(i)
    return 0


def panel(ax, name, series, season, horizon, q, i):
    s = series[i].astype(np.float64)
    ctx, y = s[:-horizon], s[-horizon:]
    shown = min(len(ctx), max(3 * horizon, 4 * season, 24))
    t_ctx = np.arange(-shown, 0)
    t_fut = np.arange(horizon)

    ax.plot(t_ctx, ctx[-shown:], color=MUTED, lw=1.2)
    ax.plot(np.r_[-1, t_fut], np.r_[ctx[-1], y], color=INK, lw=1.2)
    ax.fill_between(t_fut, q[i, 0], q[i, 8], color=FORECAST, alpha=0.2, lw=0)
    ax.plot(t_fut, q[i, 4], color=FORECAST, lw=2)
    ax.axvline(-0.5, color=MUTED, lw=0.8, ls=(0, (2, 3)))

    ax.set_title(f"{name}  (series {i})", fontsize=10, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("forecast_dir")
    ap.add_argument("--split", default="eval_zs")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tsfm_forecasts.png")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    cols = 2 if len(a.datasets) > 1 else 1
    rows = (len(a.datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 2.6 * rows), squeeze=False)
    for ax, name in zip(axes.flat, a.datasets):
        series, season, horizon = read_series(os.path.join(a.data_dir, a.split, name + ".bin"))
        q = read_forecast(os.path.join(a.forecast_dir, a.split, name + ".bin"))
        panel(ax, name, series, season, horizon, q, pick(series, horizon, rng))
    for ax in axes.flat[len(a.datasets):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("steps after the forecast origin", fontsize=8, color=MUTED)

    handles = [
        plt.Line2D([], [], color=MUTED, lw=1.2, label="context"),
        plt.Line2D([], [], color=INK, lw=1.2, label="actual"),
        plt.Line2D([], [], color=FORECAST, lw=2, label="median forecast"),
        plt.Rectangle((0, 0), 1, 1, color=FORECAST, alpha=0.2, lw=0, label="10-90% quantiles"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
