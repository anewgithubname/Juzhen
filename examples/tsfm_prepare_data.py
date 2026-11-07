"""
Download the time-series corpus for examples/demo_tsfm.cu and convert it to
flat binary files the C++ trainer can memory-load.

Sources are the public Chronos collections on Hugging Face
(autogluon/chronos_datasets) plus the ETT CSVs from GitHub. The split follows
the Chronos paper (Ansari et al., 2024):

  train/    pretraining corpus: the Chronos "in-domain" datasets and a few
            more of its training sets (no WeatherBench, no Kaggle sets). For
            the in-domain datasets the final test window is cut off, so the
            in-domain backtest stays out of training.
  eval_id/  the 15 in-domain backtests (seen datasets, unseen windows)
  eval_zs/  the 27 zero-shot backtests ("Benchmark II"), never trained on

Every list column of a dataset other than `timestamp` becomes a univariate
series, as in Chronos' evaluation. Missing values are stored as NaN.

File format (little endian), one file per dataset:
  char[4]  "TSF1"
  int32    n_series, season, horizon, reserved
  int64    offsets[n_series + 1]          (in values)
  float32  values[offsets[n_series]]
For eval files every series ends with its `horizon` test points; the season
is GluonTS' default seasonality for the frequency (used by MASE and seasonal
naive).

Requires numpy and pyarrow.
Usage: python examples/tsfm_prepare_data.py OUT_DIR [--only NAME ...]
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HF = "https://huggingface.co"
REPO = "autogluon/chronos_datasets"

# name -> (offset, prediction_length) from chronos-forecasting
# scripts/evaluation/configs/{in-domain,zero-shot}.yaml
IN_DOMAIN = {
    "electricity_15min": (-5376, 24),
    "monash_electricity_hourly": (-24, 24),
    "monash_electricity_weekly": (-8, 8),
    "monash_kdd_cup_2018": (-48, 48),
    "m4_daily": (-14, 14),
    "m4_hourly": (-48, 48),
    "m4_monthly": (-18, 18),
    "m4_weekly": (-13, 13),
    "monash_pedestrian_counts": (-48, 48),
    "taxi_30min": (-48, 48),
    "uber_tlc_hourly": (-24, 24),
    "uber_tlc_daily": (-7, 7),
    "monash_rideshare": (-24, 24),
    "monash_temperature_rain": (-30, 30),
    "monash_london_smart_meters": (-48, 48),
}
ZERO_SHOT = {
    "monash_traffic": (-24, 24),
    "monash_australian_electricity": (-48, 48),
    "ercot": (-24, 24),
    "ETTm": (-96, 24),
    "ETTh": (-24, 24),
    "exchange_rate": (-30, 30),
    "nn5": (-56, 56),
    "monash_nn5_weekly": (-8, 8),
    "monash_weather": (-30, 30),
    "monash_covid_deaths": (-30, 30),
    "monash_fred_md": (-12, 12),
    "m4_quarterly": (-8, 8),
    "m4_yearly": (-6, 6),
    "dominick": (-8, 8),
    "m5": (-28, 28),
    "monash_tourism_monthly": (-24, 24),
    "monash_tourism_quarterly": (-8, 8),
    "monash_tourism_yearly": (-4, 4),
    "monash_car_parts": (-12, 12),
    "monash_hospital": (-12, 12),
    "monash_cif_2016": (-12, 12),
    "monash_m1_yearly": (-6, 6),
    "monash_m1_quarterly": (-8, 8),
    "monash_m1_monthly": (-18, 18),
    "monash_m3_monthly": (-18, 18),
    "monash_m3_yearly": (-6, 6),
    "monash_m3_quarterly": (-8, 8),
}
# Chronos training sets that have no backtest
TRAIN_ONLY = [
    "mexico_city_bikes",
    "solar_1h",
    "ushcn_daily",
    "wiki_daily_100k",
    "wind_farms_hourly",
    "wind_farms_daily",
]


def fetch(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"  downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def hf_files(name):
    with urllib.request.urlopen(f"{HF}/api/datasets/{REPO}/tree/main/{name}") as r:
        return sorted(e["path"] for e in json.load(r) if e["path"].endswith(".parquet"))


def season_of(step_seconds, business_days=False):
    """GluonTS default seasonality for a sampling interval."""
    s = int(round(step_seconds))
    day = 86400
    if s == day and business_days:
        return 5
    if s < 60:
        return 3600 // s if s and 3600 % s == 0 else 1
    if s < 3600:  # minutes: one day
        m = s // 60
        return 1440 // m if 1440 % m == 0 else 1
    if s < day:  # hours: one day
        h = s // 3600
        return 24 // h if 24 % h == 0 else 1
    if s < 28 * day:  # daily, weekly
        return 1
    if s < 80 * day:  # monthly
        return 12
    if s < 300 * day:  # quarterly
        return 4
    return 1  # yearly


def load_parquet_series(name, raw_dir):
    """All univariate series of a Chronos dataset, plus its sampling interval."""
    series, step, business = [], None, False
    for f in hf_files(name):
        local = os.path.join(raw_dir, f)
        fetch(f"{HF}/datasets/{REPO}/resolve/main/{f}", local)
        table = pq.read_table(local)
        if step is None:
            ts = table.column("timestamp").combine_chunks()
            for i in range(len(ts)):
                t = ts[i].values.cast(pa.int64()).to_numpy()
                if len(t) > 2:
                    unit = ts.type.value_type.unit
                    scale = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]
                    step = float(np.median(np.diff(t))) / scale
                    # 1970-01-01 was a Thursday: weekday = (days + 3) % 7, Mon = 0
                    weekday = ((t // (scale * 86400)).astype(np.int64) + 3) % 7
                    business = bool(np.all(weekday < 5))
                    break
        cols = [c for c in table.column_names
                if c != "timestamp" and pa.types.is_list(table.schema.field(c).type)]
        for row in range(table.num_rows):
            for c in cols:
                v = table.column(c)[row].values
                if v is None:
                    continue
                v = pc.cast(v, pa.float64()).to_numpy(zero_copy_only=False)
                series.append(v.astype(np.float32))
    return series, step, business


def load_ett(name, raw_dir):
    series = []
    for region in (1, 2):
        local = os.path.join(raw_dir, "ETT", f"{name}{region}.csv")
        fetch(f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}{region}.csv",
              local)
        data = np.genfromtxt(local, delimiter=",", skip_header=1, dtype=np.float64,
                             usecols=range(1, 8))
        series += [data[:, j].astype(np.float32) for j in range(data.shape[1])]
    return series, 3600.0 if name == "ETTh" else 900.0, False


def write(path, series, season, horizon):
    offsets = np.zeros(len(series) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in series])
    tmp = path + ".tmp"
    with open(tmp, "wb") as fp:
        fp.write(b"TSF1")
        fp.write(struct.pack("<4i", len(series), season, horizon, 0))
        fp.write(offsets.astype("<i8").tobytes())
        for s in series:
            fp.write(np.ascontiguousarray(s, dtype="<f4").tobytes())
    os.replace(tmp, path)


def backtest(series, offset, horizon):
    """Series cut so that its last `horizon` points are the test window."""
    out = []
    for s in series:
        end = len(s) + offset + horizon
        if end - horizon >= 1:
            out.append(s[:end])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--only", nargs="*", help="process only these datasets")
    args = ap.parse_args()

    raw = os.path.join(args.out_dir, "raw")
    for sub in ("train", "eval_id", "eval_zs"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    names = list(IN_DOMAIN) + TRAIN_ONLY + list(ZERO_SHOT)
    if args.only:
        names = [n for n in names if n in args.only]
    for name in names:
        print(name, flush=True)
        if name in ("ETTh", "ETTm"):
            series, step, business = load_ett(name, raw)
        else:
            series, step, business = load_parquet_series(name, raw)
        season = season_of(step, business)
        n_values = sum(len(s) for s in series)
        print(f"  {len(series)} series, {n_values} values, step {step:.0f} s, season {season}",
              flush=True)

        if name in ZERO_SHOT:
            offset, h = ZERO_SHOT[name]
            write(os.path.join(args.out_dir, "eval_zs", name + ".bin"),
                  backtest(series, offset, h), season, h)
            continue
        if name in IN_DOMAIN:
            offset, h = IN_DOMAIN[name]
            write(os.path.join(args.out_dir, "eval_id", name + ".bin"),
                  backtest(series, offset, h), season, h)
            series = [s[:len(s) + offset] for s in series if len(s) + offset > 0]
        write(os.path.join(args.out_dir, "train", name + ".bin"), series, season, 0)

    print("done")


if __name__ == "__main__":
    sys.exit(main())
