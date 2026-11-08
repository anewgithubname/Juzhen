/**
 * @file demo_tsfm.cu
 * @brief A small decoder-only time-series foundation model in the style of
 * TimesFM (Das, Kong, Sen and Zhou, "A decoder-only foundation model for
 * time-series forecasting", ICML 2024), pretrained on public series plus
 * synthetic data and evaluated zero-shot.
 *
 * Model. A context of L points is cut into L/p patches of p points. Each
 * patch becomes one token: its normalized values, its padding mask and a
 * sinusoidal position code go through an MLP to d_model. A stack of causal
 * pre-LayerNorm transformer blocks follows, then a final LayerNorm and an MLP
 * head that predicts the next H_out points (H_out > p, as in TimesFM) as
 * nine quantiles, 0.1 .. 0.9, trained with the pinball loss (a mean head
 * trained with MSE, as in TimesFM, made the zero-shot forecasts worse: the
 * MSE is dominated by heavy-tailed series). Because attention is causal, token j forecasts
 * from the first (j+1)p points only, so one window trains every context
 * length that is a multiple of p; masking 0..p-1 leading points of the
 * window covers the rest.
 *
 * Normalization is causal too: every value in a window is shifted and scaled
 * by the mean and standard deviation of the first patch that has data. The
 * scale is floored at TSFM_SIGMA_FLOOR times the context's standard deviation
 * so that flat first patches do not blow up the rest of the window.
 *
 * Data. examples/tsfm_prepare_data.py writes the Chronos training corpus (and
 * the backtests) as flat binaries. Training windows come from those series
 * (datasets weighted by the square root of their size, points uniformly
 * within a dataset) and, with probability TSFM_SYNTH, from a synthetic
 * generator (trends, multi-seasonality, Gaussian-process random features, AR
 * noise, level shifts, spikes, intermittent and count series).
 *
 * CUDA only: the loss and its gradient are computed by a kernel in this file.
 *
 * Modes (TSFM_MODE):
 *   train     train, checkpointing to TSFM_CKPT; then forecast the backtests
 *   forecast  load TSFM_CKPT and forecast the backtests only
 * Forecasts go to TSFM_OUT/<split>/<dataset>.bin; examples/tsfm_evaluate.py
 * scores them (MASE, WQL, relative to seasonal naive).
 *
 * Environment variables:
 *   TSFM_DATA     data directory from tsfm_prepare_data.py  (default: datasets/tsfm)
 *   TSFM_MODE     train | forecast                         (default: train)
 *   TSFM_CKPT     checkpoint path                          (default: res/tsfm/model.ckpt)
 *   TSFM_RESUME   1 = continue training from TSFM_CKPT     (default: 0)
 *   TSFM_OUT      forecast directory                       (default: res/tsfm/forecasts)
 *   TSFM_SPLITS   backtest splits to forecast              (default: eval_zs,eval_id)
 *   TSFM_CONTEXT  context length L                         (default: 512)
 *   TSFM_PATCH    input patch length p                     (default: 32)
 *   TSFM_HORIZON  output patch length H_out                (default: 128)
 *   TSFM_DMODEL   model width                              (default: 256)
 *   TSFM_LAYERS   transformer blocks                       (default: 8)
 *   TSFM_HEADS    attention heads                          (default: 4)
 *   TSFM_BATCH    windows per step                         (default: 256)
 *   TSFM_STEPS    training steps                           (default: 100000)
 *   TSFM_LR       peak learning rate                       (default: 5e-4)
 *   TSFM_WD       AdamW weight decay                       (default: 0.01)
 *   TSFM_SYNTH    fraction of synthetic windows            (default: 0.3)
 *   TSFM_SIGMA_FLOOR  scale floor, x context std           (default: 0.25)
 *   TSFM_LOG      steps between log lines                  (default: 200)
 *   TSFM_VAL      steps between in-domain backtest checks  (default: 5000)
 *   TSFM_SEED     random seed                              (default: 1)
 *   NVIDIA_TF32   1 = TF32 tensor-core GEMMs (read by the Juzhen launcher)
 */

#include "../cpp/juzhen.hpp"
#include "../ml/checkpoint.hpp"
#include "../ml/layer.hpp"
#include "../ml/optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace Juzhen;
namespace fs = std::filesystem;

#ifndef CUDA
#error "demo_tsfm needs the CUDA backend"
#endif

static int env_int(const char* key, int default_v) {
    const char* v = getenv(key);
    return v ? atoi(v) : default_v;
}
static float env_float(const char* key, float default_v) {
    const char* v = getenv(key);
    return v ? (float)atof(v) : default_v;
}
static string env_str(const char* key, const string& default_v) {
    const char* v = getenv(key);
    return v ? string(v) : default_v;
}

static const float NaN = numeric_limits<float>::quiet_NaN();
constexpr int kQ = 9;  // quantile heads
constexpr int kPos = 16;       // sinusoidal position features per token
constexpr float kClamp = 100;  // bound on normalized values

// Runs fn(i) for i in [0, n) on all hardware threads.
static void parallel_for(int n, const function<void(int)>& fn) {
    const int T = max(1, min(n, (int)thread::hardware_concurrency()));
    vector<thread> pool;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            for (int i = t; i < n; i += T) fn(i);
        });
    for (auto& th : pool) th.join();
}

static void make_parent_dirs(const string& path) {
    const fs::path dir = fs::path(path).parent_path();
    error_code ec;
    if (!dir.empty()) fs::create_directories(dir, ec);
}

// ───────────────────────────── data ─────────────────────────────

// One file written by tsfm_prepare_data.py: n series stored back to back.
struct SeriesSet {
    string name;
    int season = 1, horizon = 0;
    vector<int64_t> off;
    vector<float> val;
    size_t size() const { return off.size() - 1; }
    int64_t len(size_t i) const { return off[i + 1] - off[i]; }
    const float* series(size_t i) const { return val.data() + off[i]; }
};

static bool load_tsf(const string& path, SeriesSet& s) {
    unique_ptr<FILE, decltype(&fclose)> fp(fopen(path.c_str(), "rb"), &fclose);
    if (!fp) return false;
    char magic[4];
    int32_t hdr[4];
    if (fread(magic, 1, 4, fp.get()) != 4 || memcmp(magic, "TSF1", 4) != 0 ||
        fread(hdr, sizeof(int32_t), 4, fp.get()) != 4 || hdr[0] < 0)
        return false;
    s.name = fs::path(path).stem().string();
    s.season = max(1, hdr[1]);
    s.horizon = hdr[2];
    s.off.resize((size_t)hdr[0] + 1);
    if (fread(s.off.data(), sizeof(int64_t), s.off.size(), fp.get()) != s.off.size()) return false;
    s.val.resize((size_t)s.off.back());
    return fread(s.val.data(), sizeof(float), s.val.size(), fp.get()) == s.val.size();
}

static vector<SeriesSet> load_dir(const string& dir) {
    vector<string> paths;
    error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".bin") paths.push_back(e.path().string());
    sort(paths.begin(), paths.end());
    vector<SeriesSet> sets(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
        if (!load_tsf(paths[i], sets[i])) {
            cerr << "cannot read " << paths[i] << endl;
            exit(1);
        }
    return sets;
}

// ─────────────────────────── synthetic ──────────────────────────

struct Rng {
    mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    double u(double a = 0, double b = 1) { return uniform_real_distribution<double>(a, b)(g); }
    double n() { return normal_distribution<double>(0, 1)(g); }
    double logu(double a, double b) { return exp(u(log(a), log(b))); }
    int i(int a, int b) { return uniform_int_distribution<int>(a, b)(g); }  // inclusive
    bool coin(double p) { return u() < p; }
};

static uint64_t mix(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t z = a * 0x9E3779B97F4A7C15ull ^ (b + 0xBF58476D1CE4E5B9ull) * 0x94D049BB133111EBull ^ c;
    z ^= z >> 31;
    z *= 0xD6E8FEB86659FD93ull;
    return z ^ (z >> 29);
}

// rescale v to zero mean, unit std (left alone if constant)
static void standardize(vector<double>& v) {
    double m = 0, s = 0;
    for (double x : v) m += x;
    m /= v.size();
    for (double x : v) s += (x - m) * (x - m);
    s = sqrt(s / v.size());
    for (double& x : v) x = s > 1e-12 ? (x - m) / s : 0;
}

// A random series of n points: a weighted sum of trend, seasonal, smooth
// (GP random features), random-walk and AR components, then optional level
// shifts, spikes, multiplicative / intermittent / count transforms and a
// random affine map.
static void synth_series(int n, Rng& r, float* out) {
    vector<double> y(n, 0.0), c(n);
    const double PI = 3.14159265358979323846;

    if (r.coin(0.6)) {  // trend
        const double kind = r.u();
        if (kind < 0.4) {
            for (int t = 0; t < n; ++t) c[t] = t;
        } else if (kind < 0.7) {  // piecewise linear
            const int k = r.i(1, 4);
            vector<int> cp(k);
            for (int& x : cp) x = r.i(0, n - 1);
            sort(cp.begin(), cp.end());
            double slope = r.n(), v = 0;
            for (int t = 0, j = 0; t < n; ++t) {
                while (j < k && cp[j] == t) { slope = r.n(); ++j; }
                c[t] = (v += slope);
            }
        } else if (kind < 0.85) {  // exponential
            const double rate = r.u(-4, 4) / n;
            for (int t = 0; t < n; ++t) c[t] = exp(rate * t);
        } else {  // logistic
            const double t0 = r.u(0, n), w = r.logu(1, n / 4.0);
            for (int t = 0; t < n; ++t) c[t] = 1 / (1 + exp(-(t - t0) / w));
        }
        standardize(c);
        const double w = r.u(0, 2);
        for (int t = 0; t < n; ++t) y[t] += w * c[t];
    }

    static const double canonical[] = {4, 5, 6, 7, 12, 24, 48, 52, 60, 96, 144, 168, 288, 336, 365};
    const int seasons = r.coin(0.2) ? 0 : r.i(1, 3);
    for (int s = 0; s < seasons; ++s) {
        double P = r.coin(0.6) ? canonical[r.i(0, 14)] : r.logu(2, n / 3.0);
        if (P > n / 2.0) P = r.logu(2, n / 2.0);
        if (r.coin(0.15)) {  // pulses
            const double duty = r.u(0.05, 0.5), phase = r.u(0, P);
            for (int t = 0; t < n; ++t) c[t] = fmod(t + phase, P) < duty * P ? 1 : 0;
        } else {
            const int K = r.i(1, max(1, min(6, (int)(P / 2))));
            const double decay = r.u(0, 2);
            vector<double> a(K), b(K);
            for (int k = 0; k < K; ++k) {
                a[k] = r.n() / pow(k + 1, decay);
                b[k] = r.n() / pow(k + 1, decay);
            }
            for (int t = 0; t < n; ++t) {
                double v = 0;
                for (int k = 0; k < K; ++k) {
                    const double ang = 2 * PI * (k + 1) * t / P;
                    v += a[k] * cos(ang) + b[k] * sin(ang);
                }
                c[t] = v;
            }
        }
        standardize(c);
        if (r.coin(0.3)) {  // amplitude drift
            const double g = r.u(-0.8, 2);
            for (int t = 0; t < n; ++t) c[t] *= max(0.0, 1 + g * t / n);
        }
        const double w = r.u(0.3, 2);
        for (int t = 0; t < n; ++t) y[t] += w * c[t];
    }

    if (r.coin(0.35)) {  // smooth GP sample from random Fourier features
        const int F = 48;
        const double ell = r.logu(2, n);
        vector<double> om(F), ph(F), co(F);
        for (int f = 0; f < F; ++f) {
            om[f] = r.n() / ell;
            ph[f] = r.u(0, 2 * PI);
            co[f] = r.n();
        }
        for (int t = 0; t < n; ++t) {
            double v = 0;
            for (int f = 0; f < F; ++f) v += co[f] * cos(om[f] * t + ph[f]);
            c[t] = v;
        }
        standardize(c);
        const double w = r.u(0.3, 1.5);
        for (int t = 0; t < n; ++t) y[t] += w * c[t];
    }

    if (r.coin(0.15)) {  // random walk
        double v = 0;
        for (int t = 0; t < n; ++t) c[t] = (v += r.n());
        standardize(c);
        const double w = r.u(0.2, 1.5);
        for (int t = 0; t < n; ++t) y[t] += w * c[t];
    }

    {  // AR(2) noise
        double p1 = r.u(-0.5, 0.99), p2 = r.coin(0.3) ? r.u(-0.4, 0.4) : 0;
        if (fabs(p1) + fabs(p2) > 0.99) p2 = 0;
        double a1 = 0, a2 = 0;
        for (int t = 0; t < n; ++t) {
            const double v = p1 * a1 + p2 * a2 + r.n();
            a2 = a1;
            a1 = v;
            c[t] = v;
        }
        standardize(c);
        const double w = r.logu(0.01, 1.0);
        for (int t = 0; t < n; ++t) y[t] += w * c[t];
    }

    standardize(y);
    if (r.coin(0.1))  // level shifts
        for (int k = r.i(1, 3); k > 0; --k) {
            const int t0 = r.i(0, n - 1);
            const double d = r.n();
            for (int t = t0; t < n; ++t) y[t] += d;
        }
    if (r.coin(0.1))  // spikes
        for (int k = r.i(1, max(1, n / 50)); k > 0; --k) y[r.i(0, n - 1)] += r.n() * r.u(2, 8);

    const double kind = r.u();
    if (kind < 0.15) {  // multiplicative
        const double s = r.u(0.1, 1.0);
        for (double& v : y) v = exp(s * v);
    } else if (kind < 0.27) {  // intermittent: clipped at a random level
        const double cut = r.u(-0.5, 1.5);
        for (double& v : y) v = max(0.0, v - cut);
    } else if (kind < 0.37) {  // counts
        const double mean = r.logu(0.3, 50), s = r.u(0.2, 1.0);
        for (double& v : y) {
            const double rate = mean * exp(s * v - s * s / 2);
            v = (double)poisson_distribution<int>(min(rate, 1e6))(r.g);
        }
    }

    const double scale = pow(10.0, r.u(-2, 4)), shift = r.coin(0.5) ? r.n() * r.u(0, 5) : 0;
    for (int t = 0; t < n; ++t) out[t] = (float)(scale * (y[t] + shift));
}

// ─────────────────────── model input/output ─────────────────────

struct Config {
    int context = 512, patch = 32, horizon = 128;
    int dm = 256, layers = 8, heads = 4, batch = 256;
    float sigma_floor = 0.25f;
    int tokens() const { return context / patch; }
    int window() const { return context + horizon; }
    int in_dim() const { return 2 * patch + kPos; }
    int out_dim() const { return horizon * kQ; }
    int cols() const { return tokens() * batch; }
};

struct Norm {
    double mu = 0, sigma = 1;
    int j0 = -1;  // first token with data; -1 if the context is empty
};

// Normalization of a window (x[0..context) is the context, NaN = missing):
// mean and std of the first patch with data, std floored relative to the
// whole context's std.
static Norm window_norm(const Config& c, const float* x) {
    Norm nm;
    double os = 0, oss = 0;
    int on = 0;
    for (int k = 0; k < c.context; ++k)
        if (isfinite(x[k])) {
            if (nm.j0 < 0) nm.j0 = k / c.patch;
            os += x[k];
            ++on;
        }
    if (nm.j0 < 0) return nm;
    const double omu = os / on;
    for (int k = 0; k < c.context; ++k)
        if (isfinite(x[k])) oss += (x[k] - omu) * (x[k] - omu);
    const double osd = sqrt(oss / on);

    double s = 0, ss = 0;
    int m = 0;
    for (int k = nm.j0 * c.patch; k < (nm.j0 + 1) * c.patch; ++k)
        if (isfinite(x[k])) { s += x[k]; ++m; }
    nm.mu = s / m;
    for (int k = nm.j0 * c.patch; k < (nm.j0 + 1) * c.patch; ++k)
        if (isfinite(x[k])) ss += (x[k] - nm.mu) * (x[k] - nm.mu);
    nm.sigma = max({sqrt(ss / m), c.sigma_floor * osd, 1e-6 * max(1.0, fabs(omu))});
    return nm;
}

static float normed(const Norm& nm, float v) {
    return (float)min<double>(kClamp, max<double>(-kClamp, (v - nm.mu) / nm.sigma));
}

// Token features of one window into columns col0 .. col0 + tokens - 1.
static void write_tokens(const Config& c, const float* x, const Norm& nm, Matrix<float>& tok, int col0) {
    const int p = c.patch;
    for (int j = 0; j < c.tokens(); ++j) {
        const int col = col0 + j;
        for (int i = 0; i < p; ++i) {
            const float v = x[j * p + i];
            const bool ok = isfinite(v);
            tok.elem(i, col) = ok ? normed(nm, v) : 0.0f;
            tok.elem(p + i, col) = ok ? 0.0f : 1.0f;
        }
        for (int f = 0; f < kPos / 2; ++f) {
            const double w = pow(100.0, -(double)f / (kPos / 2));
            tok.elem(2 * p + 2 * f, col) = (float)sin(j * w);
            tok.elem(2 * p + 2 * f + 1, col) = (float)cos(j * w);
        }
    }
}

// A training batch: tokens and per-token targets (the next `horizon` points,
// NaN where a target does not count), with the number of counted targets.
struct Batch {
    Matrix<float> tokens, target;
    vector<float> window;  // batch x window(), scratch
    vector<int> counted;   // counted targets per window
    double count = 0;
    Batch(const Config& c)
        : tokens("tokens", c.in_dim(), c.cols()), target("target", c.horizon, c.cols()),
          window((size_t)c.batch * c.window()), counted(c.batch) {}
};

class Sampler {
public:
    Sampler(const Config& c, const vector<SeriesSet>& sets, float synth, uint64_t seed)
        : c_(c), sets_(sets), synth_(sets.empty() ? 1.0f : synth), seed_(seed) {
        vector<double> w;
        for (const auto& s : sets) w.push_back(sqrt((double)s.val.size()));
        pick_ = discrete_distribution<int>(w.begin(), w.end());
    }

    // Fills batch `b` deterministically from (seed, step).
    void fill(Batch& b, uint64_t step) {
        parallel_for(c_.batch, [&](int i) {
            Rng r(mix(seed_, step, i));
            float* x = &b.window[(size_t)i * c_.window()];
            Norm nm;
            for (int tries = 0; tries < 20; ++tries) {
                draw(r, x);
                nm = window_norm(c_, x);
                if (nm.j0 >= 0) break;
            }
            const int col0 = i * c_.tokens();
            write_tokens(c_, x, nm, b.tokens, col0);
            int counted = 0;
            for (int j = 0; j < c_.tokens(); ++j)
                for (int h = 0; h < c_.horizon; ++h) {
                    const int pos = (j + 1) * c_.patch + h;
                    const bool ok = nm.j0 >= 0 && j >= nm.j0 && pos < c_.window() && isfinite(x[pos]);
                    b.target.elem(h, col0 + j) = ok ? normed(nm, x[pos]) : NaN;
                    counted += ok;
                }
            b.counted[i] = counted;
        });
        b.count = 0;
        for (int v : b.counted) b.count += v;
    }

private:
    void draw(Rng& r, float* x) {
        const int W = c_.window(), p = c_.patch;
        if (r.u() < synth_) {
            synth_series(W, r, x);
        } else {
            const auto& s = sets_[discrete_distribution<int>(pick_.param())(r.g)];
            const int64_t g = uniform_int_distribution<int64_t>(0, s.off.back() - 1)(r.g);
            const size_t i = upper_bound(s.off.begin(), s.off.end(), g) - s.off.begin() - 1;
            const int64_t len = s.len(i), cut = max<int64_t>(1, g - s.off[i]);
            const float* v = s.series(i);
            for (int k = 0; k < W; ++k) {
                const int64_t t = cut - c_.context + k;
                x[k] = t >= 0 && t < len ? v[t] : NaN;
            }
        }
        // short contexts: hide whole leading patches
        if (r.coin(0.1)) {
            const int k = r.i(1, c_.tokens() - 1) * p;
            for (int t = 0; t < k; ++t) x[t] = NaN;
        }
        // contexts that are not a multiple of p: hide 0..p-1 leading points
        int f = 0;
        while (f < c_.context && !isfinite(x[f])) ++f;
        if (f < c_.context && f % p == 0) {
            const int hide = r.i(0, p - 1);
            for (int t = f; t < f + hide; ++t) x[t] = NaN;
        }
    }

    const Config& c_;
    const vector<SeriesSet>& sets_;
    float synth_;
    uint64_t seed_;
    discrete_distribution<int> pick_;
};

// Pinball loss summed over the quantile heads, per counted target, and its
// gradient with respect to the network output P; one thread per (h, column).
// P and G are out_dim x cols (row k*horizon + h holds quantile k of step h),
// Y is horizon x cols (NaN = not counted), all column major. Each block adds
// its share of the mean loss to *loss, so the host reads the loss only when
// it logs.
__global__ void pinball_grad_kernel(const float* __restrict__ P, const float* __restrict__ Y,
                                    float* __restrict__ G, int H, size_t n, float inv, double* loss) {
    __shared__ double part[256];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double l = 0;
    if (i < n) {
        const size_t base = (i / H) * (size_t)kQ * H + i % H;
        const float y = Y[i];
        const bool ok = isfinite(y);
        const float tau[kQ] = {.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f};
        for (int k = 0; k < kQ; ++k) {
            float g = 0;
            if (ok) {
                const float t = tau[k], u = y - P[base + (size_t)k * H];
                l += u > 0 ? t * u : (t - 1) * u;
                g = (u > 0 ? -t : 1 - t) * inv;
            }
            G[base + (size_t)k * H] = g;
        }
    }
    part[threadIdx.x] = l;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && part[0] != 0) atomicAdd(loss, part[0] * inv);
}

class DeviceLoss {
    double* acc_ = nullptr;

public:
    DeviceLoss() {
        CudaErrorCheck(cudaMalloc(&acc_, sizeof(double)));
        CudaErrorCheck(cudaMemset(acc_, 0, sizeof(double)));
    }
    ~DeviceLoss() { cudaFree(acc_); }
    DeviceLoss(const DeviceLoss&) = delete;
    DeviceLoss& operator=(const DeviceLoss&) = delete;

    // gradient of the loss at P for targets Y; the loss is accumulated
    Matrix<CUDAfloat> grad(const Config& c, const Matrix<CUDAfloat>& P, const Matrix<CUDAfloat>& Y, double count) {
        if (P.get_transpose() || Y.get_transpose() || (int)P.num_row() != c.out_dim() ||
            (int)Y.num_row() != c.horizon || P.num_col() != Y.num_col())
            throw std::invalid_argument("pinball_grad layout mismatch");
        Matrix<CUDAfloat> G("grad", P.num_row(), P.num_col());
        const size_t n = Y.num_row() * Y.num_col();
        const float inv = count > 0 ? (float)(1.0 / count) : 0.0f;
        pinball_grad_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
            reinterpret_cast<const float*>(P.data()), reinterpret_cast<const float*>(Y.data()),
            const_cast<float*>(reinterpret_cast<const float*>(G.data())), c.horizon, n, inv, acc_);
        CudaErrorCheck(cudaGetLastError());
        return G;
    }
    // loss accumulated since the last call
    double take() {
        double v = 0;
        CudaErrorCheck(cudaMemcpy(&v, acc_, sizeof(double), cudaMemcpyDeviceToHost));
        CudaErrorCheck(cudaMemset(acc_, 0, sizeof(double)));
        return v;
    }
};

// ─────────────────────────── forecasting ────────────────────────

struct Forecast {
    int H = 0;
    vector<float> v;  // per series: quantile k [H], k = 0 .. kQ-1
};

// Forecasts the last `horizon` points of every series of a backtest set from
// the points before them, batch-by-batch with the training-shaped network.
static Forecast forecast_set(const Config& c, Sequential<CUDAfloat>& model, const SeriesSet& s,
                             size_t max_series = 0) {
    Forecast fc;
    fc.H = s.horizon;
    const size_t n = max_series ? min(max_series, s.size()) : s.size();
    const int stride = kQ * fc.H, last = c.tokens() - 1;
    fc.v.assign(n * stride, 0.0f);
    Matrix<float> tok("tokens", c.in_dim(), c.cols());
    vector<float> win((size_t)c.batch * c.window());
    vector<Norm> norms(c.batch);
    for (size_t first = 0; first < n; first += c.batch) {
        const int nb = (int)min<size_t>(c.batch, n - first);
        parallel_for(c.batch, [&](int b) {
            float* x = &win[(size_t)b * c.window()];
            std::fill(x, x + c.window(), NaN);
            if (b < nb) {
                const size_t i = first + b;
                const int64_t ctx = s.len(i) - s.horizon;
                const int64_t take = min<int64_t>(ctx, c.context);
                const float* v = s.series(i) + ctx - take;
                for (int64_t k = 0; k < take; ++k) x[c.context - take + k] = v[k];
            }
            norms[b] = window_norm(c, x);
            write_tokens(c, x, norms[b], tok, b * c.tokens());
        });
        Matrix<CUDAfloat> X(tok);
        auto P = model.eval(X).to_host();
        for (int b = 0; b < nb; ++b) {
            float* out = &fc.v[(first + b) * stride];
            const Norm& nm = norms[b];
            const int col = b * c.tokens() + last;
            for (int h = 0; h < fc.H; ++h) {
                float q[kQ];
                for (int k = 0; k < kQ; ++k) q[k] = P.elem(k * c.horizon + h, col);
                sort(q, q + kQ);  // no quantile crossing
                for (int k = 0; k < kQ; ++k)
                    out[k * fc.H + h] = nm.j0 < 0 ? 0.0f : (float)(q[k] * nm.sigma + nm.mu);
            }
        }
    }
    return fc;
}

static bool write_forecast(const string& path, const Forecast& fc) {
    make_parent_dirs(path);
    unique_ptr<FILE, decltype(&fclose)> fp(fopen(path.c_str(), "wb"), &fclose);
    if (!fp) return false;
    const int32_t hdr[4] = {(int32_t)(fc.v.size() / (kQ * max(1, fc.H))), fc.H, kQ, 0};
    return fwrite("TSFC", 1, 4, fp.get()) == 4 && fwrite(hdr, sizeof(int32_t), 4, fp.get()) == 4 &&
           fwrite(fc.v.data(), sizeof(float), fc.v.size(), fp.get()) == fc.v.size();
}

// MASE of the median forecast relative to seasonal naive, for a quick check
// during training. Series whose seasonal error is zero are skipped.
static double relative_mase(const SeriesSet& s, const Forecast& fc) {
    const int H = fc.H, m = s.season;
    const size_t n = fc.v.size() / (kQ * H);
    double model = 0, naive = 0;
    size_t used = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* y = s.series(i);
        const int64_t ctx = s.len(i) - H, lag = ctx > m ? m : 1;
        double scale = 0;
        int cnt = 0;
        for (int64_t t = lag; t < ctx; ++t)
            if (isfinite(y[t]) && isfinite(y[t - lag])) { scale += fabs(y[t] - y[t - lag]); ++cnt; }
        if (!cnt || scale == 0) continue;
        scale /= cnt;
        float last = 0;
        for (int64_t t = 0; t < ctx; ++t)
            if (isfinite(y[t])) last = y[t];
        const float* med = &fc.v[(i * kQ + kQ / 2) * H];
        double em = 0, en = 0;
        int k = 0;
        for (int h = 0; h < H; ++h) {
            const float target = y[ctx + h];
            if (!isfinite(target)) continue;
            float sn = ctx >= lag ? y[ctx - lag + h % lag] : last;
            if (!isfinite(sn)) sn = last;
            em += fabs(med[h] - target);
            en += fabs(sn - target);
            ++k;
        }
        if (!k) continue;
        model += em / k / scale;
        naive += en / k / scale;
        ++used;
    }
    return used && naive > 0 ? model / naive : NaN;
}

// ───────────────────────────── main ─────────────────────────────

int compute() {
    Config c;
    c.context = env_int("TSFM_CONTEXT", 512);
    c.patch = env_int("TSFM_PATCH", 32);
    c.horizon = env_int("TSFM_HORIZON", 128);
    c.dm = env_int("TSFM_DMODEL", 256);
    c.layers = env_int("TSFM_LAYERS", 8);
    c.heads = env_int("TSFM_HEADS", 4);
    c.batch = env_int("TSFM_BATCH", 256);
    c.sigma_floor = env_float("TSFM_SIGMA_FLOOR", 0.25f);
    const string data = env_str("TSFM_DATA", "datasets/tsfm");
    const string mode = env_str("TSFM_MODE", "train");
    const string ckpt = env_str("TSFM_CKPT", "res/tsfm/model.ckpt");
    const string out = env_str("TSFM_OUT", "res/tsfm/forecasts");
    const string splits = env_str("TSFM_SPLITS", "eval_zs,eval_id");
    const int steps = env_int("TSFM_STEPS", 100000);
    const float lr = env_float("TSFM_LR", 5e-4f), wd = env_float("TSFM_WD", 0.01f);
    const float synth = env_float("TSFM_SYNTH", 0.3f);
    const int log_every = env_int("TSFM_LOG", 200), val_every = env_int("TSFM_VAL", 5000);
    const int seed = env_int("TSFM_SEED", 1);
    if (c.context % c.patch || c.dm % c.heads || (mode != "train" && mode != "forecast")) {
        cerr << "need TSFM_CONTEXT divisible by TSFM_PATCH, TSFM_DMODEL by TSFM_HEADS, and "
                "TSFM_MODE train or forecast" << endl;
        return 1;
    }

    GPUSampler gpu_sampler(seed);
    global_rand_gen.seed(seed);

    const int N = c.tokens(), cols = c.cols();
    cout << "=== TSFM: context " << c.context << " in " << N << " patches of " << c.patch << ", output patch "
         << c.horizon << "; transformer " << c.layers << "x(d_model " << c.dm << ", " << c.heads
         << " heads), batch " << c.batch << " ===" << endl;

    // in-MLP -> blocks -> LayerNorm -> out-MLP; Sequential runs back() to front()
    ReluLayer<CUDAfloat> in1(c.dm, c.in_dim(), cols);
    in1.W() = Matrix<CUDAfloat>::randn(c.dm, c.in_dim()) * sqrtf(2.0f / c.in_dim());
    in1.b() = Matrix<CUDAfloat>::zeros(c.dm, 1);
    LinearLayer<CUDAfloat> in2(c.dm, c.dm, cols);
    in2.W() = Matrix<CUDAfloat>::randn(c.dm, c.dm) * sqrtf(1.0f / c.dm);
    in2.b() = Matrix<CUDAfloat>::zeros(c.dm, 1);
    vector<unique_ptr<TransformerLayer<CUDAfloat>>> blocks;
    for (int i = 0; i < c.layers; ++i)
        blocks.push_back(make_unique<TransformerLayer<CUDAfloat>>(c.dm, c.dm, 4 * c.dm, N, c.batch, c.heads, true));
    LayerNormLayer<CUDAfloat> ln(c.dm, cols);
    ReluLayer<CUDAfloat> out1(c.dm, c.dm, cols);
    out1.W() = Matrix<CUDAfloat>::randn(c.dm, c.dm) * sqrtf(2.0f / c.dm);
    out1.b() = Matrix<CUDAfloat>::zeros(c.dm, 1);
    LinearLayer<CUDAfloat> out2(c.out_dim(), c.dm, cols);
    out2.W() = Matrix<CUDAfloat>::randn(c.out_dim(), c.dm) * sqrtf(0.1f / c.dm);
    {  // quantile heads start at the standard normal quantiles
        Matrix<float> b0("b0", c.out_dim(), 1);
        static const float z[kQ] = {-1.2816f, -.8416f, -.5244f, -.2533f, 0, .2533f, .5244f, .8416f, 1.2816f};
        for (int k = 0; k < kQ; ++k)
            for (int h = 0; h < c.horizon; ++h) b0.elem(k * c.horizon + h, 0) = z[k];
        out2.b() = Matrix<CUDAfloat>(b0);
    }

    list<Layer<CUDAfloat>*> net{&out2, &out1, &ln};
    for (int i = c.layers - 1; i >= 0; --i) net.push_back(blocks[i].get());
    net.push_back(&in2);
    net.push_back(&in1);
    Sequential<CUDAfloat> model(net);
    size_t n_params = 0;
    for (auto* l : net)
        for (auto& [name, m] : l->checkpoint_parameters()) n_params += m->num_row() * m->num_col();
    cout << "parameters: " << n_params << endl;

    AdamWOptions options;
    options.lr = lr;
    options.weight_decay = wd;
    AdamW<CUDAfloat> optimizer(model.parameters(), options);

    TrainingProgress progress{};
    string err;
    const bool resume = env_int("TSFM_RESUME", 0) != 0;
    if (mode == "forecast" || resume) {
        if (!load_checkpoint(model.layers(), ckpt, progress, &err)) {
            cerr << "cannot load " << ckpt << ": " << err << endl;
            return 1;
        }
        cout << "loaded " << ckpt << " at step " << progress.step << endl;
    }

    auto t_load = chrono::steady_clock::now();
    vector<SeriesSet> val_sets = load_dir(data + "/eval_id");

    if (mode == "train") {
        const vector<SeriesSet> train = load_dir(data + "/train");
        size_t n_values = 0, n_series = 0;
        for (const auto& s : train) { n_values += s.val.size(); n_series += s.size(); }
        cout << "train corpus: " << train.size() << " datasets, " << n_series << " series, " << n_values
             << " values (" << setprecision(1) << fixed
             << chrono::duration<double>(chrono::steady_clock::now() - t_load).count() << " s to load); "
             << "synthetic fraction " << (train.empty() ? 1.0f : synth) << endl;

        Sampler sampler(c, train, synth, (uint64_t)seed);
        Batch batches[2] = {Batch(c), Batch(c)};
        DeviceLoss dloss;
        const int warmup = min(2000, max(1, steps / 20));
        const double PI = 3.14159265358979323846;
        auto lr_at = [&](int step) {
            if (step < warmup) return lr * (step + 1) / warmup;
            const double t = (double)(step - warmup) / max(1, steps - warmup);
            return (float)(lr * (0.1 + 0.45 * (1 + cos(PI * t))));
        };

        int n_log = 0, skipped = 0;
        auto t0 = chrono::steady_clock::now();
        const int start = (int)progress.step;
        future<void> next = async(launch::async, [&] { sampler.fill(batches[start % 2], start); });
        for (int step = start; step < steps; ++step) {
            next.get();
            Batch& b = batches[step % 2];
            if (step + 1 < steps)
                next = async(launch::async, [&, step] { sampler.fill(batches[(step + 1) % 2], step + 1); });

            const float lr_t = lr_at(step);
            optimizer.set_lr(lr_t);
            Matrix<CUDAfloat> X(b.tokens);
            optimizer.zero_grad();
            model.backward(dloss.grad(c, model.eval(X), Matrix<CUDAfloat>(b.target), b.count));
            ++n_log;
            try {
                optimizer.clip_grad_norm(1.0);
                optimizer.step();
            } catch (const std::runtime_error&) {
                ++skipped;  // non-finite gradient
            }
            progress.step = step + 1;

            if ((step + 1) % log_every == 0 || step + 1 == steps) {
                const double sum_loss = dloss.take();
                const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                cout << "step " << setw(7) << step + 1 << "  lr " << scientific << setprecision(2) << lr_t << fixed
                     << setprecision(4) << "  loss " << sum_loss / max(1, n_log) << setprecision(0) << "  ("
                     << secs << " s";
                if (skipped) cout << ", " << skipped << " skipped";
                cout << ")" << endl;
                n_log = 0;
            }
            const bool last = step + 1 == steps;
            if ((step + 1) % val_every == 0 || last) {
                // geometric mean over in-domain backtests (first 2048 series each)
                double lg = 0;
                int k = 0;
                for (const auto& s : val_sets) {
                    if (s.horizon > c.horizon) continue;
                    const double r = relative_mase(s, forecast_set(c, model, s, 2048));
                    if (isfinite(r) && r > 0) { lg += log(r); ++k; }
                }
                cout << "  in-domain MASE / seasonal naive (geo-mean, " << k << " sets): " << setprecision(3)
                     << (k ? exp(lg / k) : NaN) << endl;
                make_parent_dirs(ckpt);
                if (!save_checkpoint(model.layers(), ckpt, progress, &err))
                    cerr << "checkpoint failed: " << err << endl;
            }
        }
    }

    // forecasts for the backtests
    stringstream ss(splits);
    string split;
    while (getline(ss, split, ',')) {
        const vector<SeriesSet> sets = split == "eval_id" ? std::move(val_sets) : load_dir(data + "/" + split);
        double lg = 0;
        int k = 0;
        for (const auto& s : sets) {
            if (s.horizon > c.horizon) {
                cout << "  skip " << s.name << ": horizon " << s.horizon << " > output patch" << endl;
                continue;
            }
            auto t = chrono::steady_clock::now();
            const Forecast fc = forecast_set(c, model, s);
            const string path = out + "/" + split + "/" + s.name + ".bin";
            if (!write_forecast(path, fc)) {
                cerr << "cannot write " << path << endl;
                return 1;
            }
            const double r = relative_mase(s, fc);
            if (isfinite(r) && r > 0) { lg += log(r); ++k; }
            cout << "  " << split << "/" << std::left << setw(32) << s.name << std::right << setw(7) << s.size()
                 << " series  MASE/naive " << setprecision(3) << r << "  (" << setprecision(1)
                 << chrono::duration<double>(chrono::steady_clock::now() - t).count() << " s)" << endl;
        }
        cout << split << ": geo-mean MASE / seasonal naive " << setprecision(3) << (k ? exp(lg / k) : NaN)
             << " over " << k << " datasets; forecasts in " << out << "/" << split << endl;
    }
    return 0;
}
