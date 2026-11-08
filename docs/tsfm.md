# A small time-series foundation model

[`examples/demo_tsfm.cu`](../examples/demo_tsfm.cu) trains a decoder-only
forecasting model in the style of TimesFM (Das, Kong, Sen and Zhou, ICML 2024)
on public time series plus synthetic data. It then forecasts the Chronos
benchmarks (Ansari et al., 2024) zero-shot. The demo needs the CUDA backend.

With 39M parameters, trained for 40 minutes on one RTX 4090, its zero-shot
accuracy beats every Chronos-T5 model and is as good as or better than
Chronos-Bolt-small. It is about 0.01 behind Chronos-Bolt-base. On the datasets
it was trained on, it is level with the smallest Chronos models but does not
improve with size. See [Results](#results).

## Model

- **Tokens.** A context of 512 points is cut into 16 patches of 32. Each token
  has 80 inputs: the patch's normalized values, its padding mask (missing
  values and points before the series start), and 16 sinusoidal features of
  the token position. An MLP (`Linear`, ReLU, `Linear`) maps them to `d_model`.
- **Backbone.** Causal pre-LayerNorm `TransformerLayer` blocks and a final
  LayerNorm.
- **Head.** An MLP maps each token to the next 128 points, as quantiles 0.1 to
  0.9. Because attention is causal, token *j* forecasts from the first
  32(*j*+1) points only. One window therefore trains all 16 context lengths.
  Hiding 0 to 31 leading points of each window covers the lengths in between.
- **Normalization.** All values in a window are shifted and scaled by the mean
  and standard deviation of the first patch that has data, as in TimesFM. This
  keeps the normalization causal. The scale is floored at 0.25 times the
  standard deviation of the context, so a flat first patch does not blow up
  the rest of the window.
- **Loss.** The pinball loss summed over the nine quantiles, averaged over the
  targets that exist. A CUDA kernel computes it and its gradient on the device.
- **Forecasting.** The context is right-aligned in the 512-point window. The
  last token's quantiles are sorted and denormalized. Horizons up to 128 are
  supported. Longer contexts are truncated to their last 512 points.

| | `d_model` | Blocks | Heads | Parameters |
|---|---|---|---|---|
| Default | 256 | 8 | 4 | 6.8M |
| Large | 512 | 12 | 8 | 39M |

## Data

[`examples/tsfm_prepare_data.py`](../examples/tsfm_prepare_data.py) downloads
the data and converts it to flat binaries. Most series come from the Chronos
collection on Hugging Face (`autogluon/chronos_datasets`). ETTh and ETTm come
from the ETDataset CSVs. The split follows the Chronos paper:

| Directory | Contents |
|---|---|
| `train/` | 21 datasets, 209,696 series, 866M values. These are close to the Chronos training datasets; WeatherBench and the two Kaggle-only sets are left out. For the 15 in-domain datasets, the final test window is cut off. |
| `eval_id/` | The 15 in-domain backtests: datasets seen in training, with unseen windows. |
| `eval_zs/` | The 27 zero-shot backtests (Chronos Benchmark II). None of these datasets is used in training. |

The download is 2.4 GB of parquet and the converted files take 5 GB. Each
list column of a dataset becomes a univariate series. Missing values are kept
as NaN.

A training window comes from the synthetic generator with probability 0.3.
The generator combines trends, seasonalities, Gaussian-process random features,
random walks, AR noise, level shifts, spikes, and intermittent and count
series. Otherwise the window comes from a real series: a dataset is drawn with
weight proportional to the square root of its size, then a point uniformly
within that dataset.

## Usage

```bash
pip install numpy pyarrow
python examples/tsfm_prepare_data.py datasets/tsfm

cmake -S . -B build_cuda -DNVIDIA_CUDA=ON -DROCM_HIP=OFF -DAPPLE_SILICON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --target demo_tsfm -j

# train the default model, then forecast both backtests to res/tsfm/forecasts
NVIDIA_TF32=1 ./build_cuda/demo_tsfm
# the large model
NVIDIA_TF32=1 TSFM_DMODEL=512 TSFM_LAYERS=12 TSFM_HEADS=8 TSFM_LR=3e-4 ./build_cuda/demo_tsfm

python examples/tsfm_evaluate.py datasets/tsfm res/tsfm/forecasts --split eval_zs
python examples/tsfm_evaluate.py datasets/tsfm res/tsfm/forecasts --split eval_id
```

`TSFM_MODE=forecast` loads `TSFM_CKPT` and only forecasts, and `TSFM_RESUME=1`
continues training from it. The other settings are environment variables
listed at the top of the source file. During training the demo prints the
in-domain MASE every 5,000 steps. This is a quick check on the first 2,048
series of each dataset.

On an RTX 4090 a step of 256 windows takes:

| | FP32 | `NVIDIA_TF32=1` |
|---|---|---|
| Default (6.8M) | 10.0 ms (17 min for 100k steps) | 6.7 ms (12 min) |
| Large (39M) | 34.7 ms (58 min) | 23.3 ms (40 min) |

TF32 does not change accuracy measurably. Paired by seed, its zero-shot MASE
differs from FP32 by -0.001 (standard deviation 0.007).

## Evaluation

[`examples/tsfm_evaluate.py`](../examples/tsfm_evaluate.py) follows the
Chronos paper:

- **MASE** of the median forecast, and the mean weighted quantile loss
  (**WQL**) over quantiles 0.1 to 0.9.
- Each metric is divided by the same metric of seasonal naive, which uses the
  GluonTS seasonality of the frequency.
- Datasets are aggregated by geometric mean. Lower is better and 1 means
  seasonal naive.

The seasonal naive MASE and WQL match those published in the Chronos
repository on all 27 zero-shot datasets. In-domain, the seasonal naive MASE
matches on 14 of the 15 datasets. On `monash_kdd_cup_2018` it is 0.973 against
0.994, probably because missing values are handled differently.

## Results

All runs use the defaults: 100k steps of 256 windows, learning rate 5e-4
(3e-4 for the large model) and AdamW with weight decay 0.01. Each entry gives
the mean ± standard deviation over seeds. The default model has eight runs:
four seeds, each in FP32 and TF32. The large model has eleven seeds.

| Model | Zero-shot MASE | Zero-shot WQL | In-domain MASE | In-domain WQL |
|---|---|---|---|---|
| This demo, 6.8M | 0.813 ± 0.007 | 0.652 ± 0.007 | 0.739 ± 0.003 | 0.588 ± 0.007 |
| This demo, 39M | 0.800 ± 0.007 | 0.634 ± 0.007 | 0.739 ± 0.006 | 0.601 ± 0.008 |
| Chronos-T5-tiny, 8M | 0.870 | 0.711 | 0.765 | 0.629 |
| Chronos-Bolt-tiny, 9M | 0.845 | 0.668 | 0.740 | 0.573 |
| Chronos-T5-small, 46M | 0.830 | 0.665 | 0.730 | 0.609 |
| Chronos-Bolt-small, 48M | 0.819 | 0.636 | 0.703 | 0.544 |
| Chronos-T5-base, 200M | 0.816 | 0.642 | 0.701 | 0.579 |
| Chronos-Bolt-base, 205M | 0.791 | 0.624 | 0.680 | 0.534 |
| Chronos-T5-large, 710M | 0.821 | 0.650 | 0.694 | 0.560 |

The Chronos numbers are single runs taken from
`scripts/evaluation/results` in the chronos-forecasting repository. Chronos
also trained on more data: WeatherBench, TSMixup augmentations and 1M
KernelSynth series.

- **Zero-shot.** At equal size the demo matches or beats both Chronos
  families. The large model improves on the default one: MASE by 0.013 (Welch
  t = -3.9) and WQL by 0.018 (t = -5.7).
- **In-domain.** The default model beats Chronos-T5-tiny and is level with
  Chronos-Bolt-tiny. Unlike Chronos, the demo does not improve with size, so
  the large model trails Chronos-Bolt-small by 0.036 in MASE and 0.057 in WQL.
  The largest drop from the default to the large model is on `m4_daily`,
  described under [Limitations](#limitations). The WQL on `monash_rideshare`
  also drops, from 0.72 to 0.79 times seasonal naive.

**A mean head trained with MSE.** TimesFM trains a mean forecast with MSE.
An earlier version of the demo did the same, next to the quantile heads. It
was worse with one seed per model:

| | MSE + pinball | Pinball only (one seed) |
|---|---|---|
| 6.8M, zero-shot MASE / WQL | 0.833 / 0.667 | 0.806 / 0.647 |
| 6.8M, in-domain MASE / WQL | 0.765 / 0.615 | 0.736 / 0.582 |
| 39M, zero-shot MASE / WQL | 0.821 / 0.661 | 0.792 / 0.618 |

After first-patch normalization, the MSE is dominated by a few heavy-tailed
series. The forecasts and metrics use only the quantiles, so the mean head
was dropped.

## Limitations

- **Long trending series.** On `m4_daily` the large model's MASE is 1.35 times
  that of seasonal naive. With a context cut to the last 256 points it drops to
  about 1.0, the default model's level (one seed, forecasting only). Over a long trending context, the
  first patch lies far from the forecast origin, and the last values are about
  ten scales away from it after normalization. The large model extrapolates
  poorly there. Other datasets still gain from the full context.
- **Below seasonal naive.** Averaged over the large model's seeds, four
  zero-shot datasets score worse than seasonal naive: `exchange_rate` (1.27),
  `monash_australian_electricity` (1.18), `dominick` (1.03) and
  `monash_tourism_quarterly` (1.02).
- **Seed noise.** Seeds differ by about 0.007 in the aggregate scores. Datasets
  with only 5 to 8 series vary the most from seed to seed. Differences near
  0.01 need several seeds.
- **Benchmark reuse.** The choice of loss was made after looking at the
  zero-shot scores. An independent benchmark such as GIFT-Eval would be a
  cleaner test.
- **Fixed shapes.** The network is built for one batch size and 16 tokens.
  Forecasting pads the last batch, and short series still run all 16 tokens.
