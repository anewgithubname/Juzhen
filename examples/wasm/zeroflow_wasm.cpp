/**
 * @file zeroflow_wasm.cpp
 * @brief Standalone (no-dependency) core for the single-file WebAssembly
 *        version of the zero-flow demo (see ../demo_zeroflow_web.cu and
 *        "Zero-Flow Encoders", arXiv:2602.00797).
 *
 * Implements a small velocity-field MLP (3 -> 128 -> 128 -> 128 -> 2, ReLU)
 * trained with Adam on the rectified-flow objective with independent
 * coupling, entirely in plain C++ so it compiles with Emscripten without
 * BLAS. Exports a tiny C API consumed by the JS shell:
 *
 *   zf_init(src, tgt, seed)   reset model + pick datasets
 *   zf_train(n)               run n training iterations, return last loss
 *   zf_animate()              integrate particles 0->1, fill frame buffers
 *   zf_pts()/zf_vec()/zf_meanv()/zf_grid()   float* accessors
 *   zf_M()/zf_K()/zf_G()      buffer dimensions
 *
 * Build: see build.sh (emcc -O3 -msimd128 -ffast-math, SINGLE_FILE).
 */

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

// ── network dimensions ──────────────────────────────────────────────────────
constexpr int D_IN = 3;    // (x, y, t)
constexpr int H = 128;
constexpr int D_OUT = 2;   // (vx, vy)
constexpr int BATCH = 256;

// ── animation dimensions ────────────────────────────────────────────────────
constexpr int M = 300;              // particles
constexpr int K = 50;               // Euler frames (t = 0..1 inclusive -> K+1)
constexpr int G = 12;               // G x G velocity-field grid
constexpr float LO = -3.5f, HI = 3.5f;
constexpr int NMAX = 300;           // max forward batch (M >= BATCH? no: max(BATCH,M,G*G))
static_assert(NMAX >= M && NMAX >= G * G, "NMAX too small");
constexpr int NBUF = (BATCH > NMAX) ? BATCH : NMAX;

std::mt19937 rng(7);

// ── one dense layer with Adam state ─────────────────────────────────────────
struct Linear {
    int in, out;
    std::vector<float> W, b, gW, gb, mW, vW, mb, vb;

    void init(int in_, int out_, float scale) {
        in = in_; out = out_;
        W.assign((size_t)in * out, 0.0f);
        b.assign(out, 0.0f);
        gW.assign(W.size(), 0.0f); gb.assign(out, 0.0f);
        mW.assign(W.size(), 0.0f); vW.assign(W.size(), 0.0f);
        mb.assign(out, 0.0f);      vb.assign(out, 0.0f);
        std::normal_distribution<float> g(0.0f, scale);
        for (auto& w : W) w = g(rng);
    }
};

Linear L1, L2, L3, L4;
int adam_t = 0;
constexpr float LR = 1e-3f, B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f;

// activations / gradients, sample-major: a[j*dim + i]
std::vector<float> a1(NBUF * H), a2(NBUF * H), a3(NBUF * H), aout(NBUF * D_OUT);
std::vector<float> d1(NBUF * H), d2(NBUF * H), d3(NBUF * H), dout(NBUF * D_OUT);

// y = W x + b for each sample (x: n x in, y: n x out, both sample-major)
void fwd_layer(const Linear& L, const float* x, float* y, int n, bool relu) {
    for (int j = 0; j < n; ++j) {
        const float* xj = x + (size_t)j * L.in;
        float* yj = y + (size_t)j * L.out;
        for (int i = 0; i < L.out; ++i) {
            const float* wi = &L.W[(size_t)i * L.in];
            float s = L.b[i];
            for (int k = 0; k < L.in; ++k) s += wi[k] * xj[k];
            yj[i] = (relu && s < 0.0f) ? 0.0f : s;
        }
    }
}

// backward through one layer: given dy (n x out) and forward input x (n x in),
// accumulate gW/gb and produce dx (n x in). If relu_out, dy is masked by the
// layer's own OUTPUT activation y (ReLU derivative) before use.
void bwd_layer(Linear& L, const float* x, const float* y, float* dy, float* dx,
               int n, bool relu_out) {
    if (relu_out) {
        for (size_t j = 0; j < (size_t)n * L.out; ++j)
            if (y[j] <= 0.0f) dy[j] = 0.0f;
    }
    for (int j = 0; j < n; ++j) {
        const float* xj = x + (size_t)j * L.in;
        const float* dyj = dy + (size_t)j * L.out;
        for (int i = 0; i < L.out; ++i) {
            const float d = dyj[i];
            if (d == 0.0f) continue;
            float* gwi = &L.gW[(size_t)i * L.in];
            for (int k = 0; k < L.in; ++k) gwi[k] += d * xj[k];
            L.gb[i] += d;
        }
        if (dx) {
            float* dxj = dx + (size_t)j * L.in;
            for (int k = 0; k < L.in; ++k) dxj[k] = 0.0f;
            for (int i = 0; i < L.out; ++i) {
                const float d = dyj[i];
                if (d == 0.0f) continue;
                const float* wi = &L.W[(size_t)i * L.in];
                for (int k = 0; k < L.in; ++k) dxj[k] += wi[k] * d;
            }
        }
    }
}

void adam_step_arr(std::vector<float>& w, std::vector<float>& g,
                   std::vector<float>& m, std::vector<float>& v,
                   float bc1, float bc2) {
    for (size_t i = 0; i < w.size(); ++i) {
        m[i] = B1 * m[i] + (1 - B1) * g[i];
        v[i] = B2 * v[i] + (1 - B2) * g[i] * g[i];
        w[i] -= LR * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + EPS);
        g[i] = 0.0f;
    }
}

void adam_step() {
    adam_t++;
    const float bc1 = 1.0f - std::pow(B1, (float)adam_t);
    const float bc2 = 1.0f - std::pow(B2, (float)adam_t);
    for (Linear* L : {&L1, &L2, &L3, &L4}) {
        adam_step_arr(L->W, L->gW, L->mW, L->vW, bc1, bc2);
        adam_step_arr(L->b, L->gb, L->mb, L->vb, bc1, bc2);
    }
}

// full network forward: X (n x 3) -> aout (n x 2)
void net_forward(const float* X, int n) {
    fwd_layer(L1, X, a1.data(), n, true);
    fwd_layer(L2, a1.data(), a2.data(), n, true);
    fwd_layer(L3, a2.data(), a3.data(), n, true);
    fwd_layer(L4, a3.data(), aout.data(), n, false);
}

// ── datasets (must match demo_zeroflow_web.cu) ──────────────────────────────
// ids: 0 gaussian, 1 mixture, 2 moons, 3 ring, 4 spiral
void sample_dataset(int id, float* buf, int n) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        float x = 0, y = 0;
        switch (id) {
            case 1: {
                const float c = (u(rng) < 0.5f) ? -1.8f : 1.8f;
                x = c + 0.5f * g(rng);
                y = c + 0.5f * g(rng);
                break;
            }
            case 2: {
                const float th = 3.14159265f * u(rng);
                if (u(rng) < 0.5f) { x =  2.0f * std::cos(th) - 1.0f; y =  2.0f * std::sin(th) - 0.6f; }
                else               { x = -2.0f * std::cos(th) + 1.0f; y = -2.0f * std::sin(th) + 0.6f; }
                x += 0.15f * g(rng);
                y += 0.15f * g(rng);
                break;
            }
            case 3: {
                const float th = 6.2831853f * u(rng);
                const float r = 2.2f + 0.15f * g(rng);
                x = r * std::cos(th);
                y = r * std::sin(th);
                break;
            }
            case 4: {
                const float s = u(rng);
                const float th = 4.0f * 3.14159265f * s;
                const float r = 0.4f + 2.6f * s;
                x = r * std::cos(th) + 0.1f * g(rng);
                y = r * std::sin(th) + 0.1f * g(rng);
                break;
            }
            default:
                x = 0.8f * g(rng);
                y = 0.8f * g(rng);
        }
        buf[2 * i] = x;
        buf[2 * i + 1] = y;
    }
}

int g_src = 1, g_tgt = 1;

// ── training scratch ────────────────────────────────────────────────────────
std::vector<float> x0v(2 * BATCH), x1v(2 * BATCH), tv_(BATCH);
std::vector<float> Xin(NBUF * D_IN), Ytgt(2 * BATCH);

// ── animation output buffers (read directly from JS) ───────────────────────
std::vector<float> frames_pts((K + 1) * M * 3);      // x, y, speed
std::vector<float> frames_vec((K + 1) * G * G * 2);  // grid u, v
std::vector<float> frames_meanv(K + 1);
std::vector<float> grid_xy(2 * G * G);               // grid x[], then y[]
std::vector<float> Z(2 * M), Vz(2 * M);

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int zf_M() { return M; }
EMSCRIPTEN_KEEPALIVE int zf_K() { return K; }
EMSCRIPTEN_KEEPALIVE int zf_G() { return G; }
EMSCRIPTEN_KEEPALIVE float* zf_pts()   { return frames_pts.data(); }
EMSCRIPTEN_KEEPALIVE float* zf_vec()   { return frames_vec.data(); }
EMSCRIPTEN_KEEPALIVE float* zf_meanv() { return frames_meanv.data(); }
EMSCRIPTEN_KEEPALIVE float* zf_grid()  { return grid_xy.data(); }

EMSCRIPTEN_KEEPALIVE void zf_init(int src, int tgt, unsigned seed) {
    g_src = src;
    g_tgt = tgt;
    rng.seed(seed);
    adam_t = 0;
    L1.init(D_IN, H, std::sqrt(2.0f / D_IN));
    L2.init(H, H, std::sqrt(2.0f / H));
    L3.init(H, H, std::sqrt(2.0f / H));
    L4.init(H, D_OUT, std::sqrt(1.0f / H));
    for (int a = 0; a < G; ++a)
        for (int b = 0; b < G; ++b) {
            const int j = a * G + b;
            grid_xy[j] = LO + (HI - LO) * (a + 0.5f) / G;
            grid_xy[G * G + j] = LO + (HI - LO) * (b + 0.5f) / G;
        }
}

// Run n training iterations of the rectified-flow objective with independent
// coupling; returns the last mini-batch loss (mean ||v - (X1-X0)||^2).
EMSCRIPTEN_KEEPALIVE float zf_train(int n) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float loss = 0.0f;
    for (int it = 0; it < n; ++it) {
        sample_dataset(g_src, x0v.data(), BATCH);
        sample_dataset(g_tgt, x1v.data(), BATCH);
        for (int j = 0; j < BATCH; ++j) {
            const float t = u(rng);
            tv_[j] = t;
            Xin[j * 3 + 0] = (1 - t) * x0v[2 * j] + t * x1v[2 * j];
            Xin[j * 3 + 1] = (1 - t) * x0v[2 * j + 1] + t * x1v[2 * j + 1];
            Xin[j * 3 + 2] = t;
            Ytgt[2 * j] = x1v[2 * j] - x0v[2 * j];
            Ytgt[2 * j + 1] = x1v[2 * j + 1] - x0v[2 * j + 1];
        }
        net_forward(Xin.data(), BATCH);
        loss = 0.0f;
        const float sc = 2.0f / BATCH;
        for (int j = 0; j < BATCH; ++j) {
            for (int i = 0; i < 2; ++i) {
                const float e = aout[j * 2 + i] - Ytgt[j * 2 + i];
                loss += e * e;
                dout[j * 2 + i] = sc * e;
            }
        }
        loss /= BATCH;
        bwd_layer(L4, a3.data(), aout.data(), dout.data(), d3.data(), BATCH, false);
        bwd_layer(L3, a2.data(), a3.data(), d3.data(), d2.data(), BATCH, true);
        bwd_layer(L2, a1.data(), a2.data(), d2.data(), d1.data(), BATCH, true);
        bwd_layer(L1, Xin.data(), a1.data(), d1.data(), nullptr, BATCH, true);
        adam_step();
    }
    return loss;
}

// Integrate M particles from t=0 to 1 (Euler, K steps) and evaluate the field
// on the grid at every frame; fills the frame buffers.
EMSCRIPTEN_KEEPALIVE void zf_animate() {
    sample_dataset(g_src, Z.data(), M);
    const float dt = 1.0f / K;
    for (int k = 0; k <= K; ++k) {
        const float t = (float)k / K;

        // particle velocities
        for (int j = 0; j < M; ++j) {
            Xin[j * 3 + 0] = Z[2 * j];
            Xin[j * 3 + 1] = Z[2 * j + 1];
            Xin[j * 3 + 2] = t;
        }
        net_forward(Xin.data(), M);
        float mv = 0.0f;
        for (int j = 0; j < M; ++j) {
            Vz[2 * j] = aout[j * 2];
            Vz[2 * j + 1] = aout[j * 2 + 1];
            const float sp = std::sqrt(Vz[2 * j] * Vz[2 * j] + Vz[2 * j + 1] * Vz[2 * j + 1]);
            mv += sp;
            float* p = &frames_pts[((size_t)k * M + j) * 3];
            p[0] = Z[2 * j];
            p[1] = Z[2 * j + 1];
            p[2] = sp;
        }
        frames_meanv[k] = mv / M;

        // grid field
        for (int j = 0; j < G * G; ++j) {
            Xin[j * 3 + 0] = grid_xy[j];
            Xin[j * 3 + 1] = grid_xy[G * G + j];
            Xin[j * 3 + 2] = t;
        }
        net_forward(Xin.data(), G * G);
        for (int j = 0; j < G * G; ++j) {
            frames_vec[((size_t)k * G * G + j) * 2] = aout[j * 2];
            frames_vec[((size_t)k * G * G + j) * 2 + 1] = aout[j * 2 + 1];
        }

        if (k < K)
            for (int j = 0; j < 2 * M; ++j) Z[j] += Vz[j] * dt;
    }
}

}  // extern "C"

#ifndef __EMSCRIPTEN__
// tiny native smoke test: train mixture->mixture, print mean_v profile
#include <cstdio>
int main() {
    zf_init(1, 1, 7);
    for (int c = 0; c < 20; ++c) printf("loss %.4f\n", zf_train(200));
    zf_animate();
    for (int k = 0; k <= K; k += 5)
        printf("t=%.2f  mean_v=%.3f\n", (float)k / K, frames_meanv[k]);
    return 0;
}
#endif
