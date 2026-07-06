/**
 * @file zeroflow_cond_wasm.cpp
 * @brief Standalone (no-dependency) core for the single-file WebAssembly
 *        visualization of the CONDITIONAL zero-flow condition (Theorem 3.4 of
 *        "Zero-Flow Encoders", arXiv:2602.00797; cf. Figure 2 of the paper).
 *
 * Data model (fixed):  Y ~ N(0,1),  X = 0.5 Y + eps,  eps ~ N(0,1).
 * A conditional velocity field u_t(x_t; f(Y'), Y) is trained on
 *
 *   min_u  int_0^1 E || X' - X - u_t(X_t, f(Y'), Y) ||^2 dt        (eq. 6)
 *
 * with (X', Y') an INDEPENDENT copy of (X, Y) and X_t = t X' + (1-t) X.
 * Its optimum v_t(z; eta, xi) = E[X' - X | X_t = z, f(Y') = eta, Y = xi]
 * transports p_{X|Y=xi} to p_{X|f(Y)=eta} (Theorem 3.3), and by Theorem 3.4
 * v_{t=0.5}(z; f(xi), xi) = 0 for all z, xi  <=>  p_{X|Y} = p_{X|f(Y)},
 * i.e. the statistic f is sufficient for predicting X.
 *
 * Exports a tiny C API consumed by the JS shell (see shell_cond.html):
 *
 *   zfc_init(fid, seed)       reset model + pick the statistic f
 *   zfc_train(n)              run n training iterations, return last loss
 *   zfc_animate()             integrate particles 0->1, fill frame buffers
 *   zfc_pts()/zfc_vec()/zfc_meanv()/zfc_grid()   float* accessors
 *   zfc_M()/zfc_K()/zfc_G()   buffer dimensions
 *
 * Statistic menu (fid): 0 sigmoid(-2y)  1 y  2 sin(2y)  3 |y|  4 sign(y)  5 0.
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
constexpr int D_IN = 4;    // (x_t, t, f(y'), y)
constexpr int H = 64;
constexpr int D_OUT = 1;   // velocity in x
constexpr int BATCH = 256;

// ── animation dimensions ────────────────────────────────────────────────────
constexpr int M = 300;              // particles (data points)
constexpr int K = 50;               // Euler frames (t = 0..1 inclusive -> K+1)
constexpr int G = 12;               // G x G midpoint-velocity grid (xi, z)
constexpr float LO = -3.5f, HI = 3.5f;
constexpr int NBUF = (BATCH > M) ? BATCH : ((M > G * G) ? M : G * G);

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

// full network forward: X (n x 4) -> aout (n x 1)
void net_forward(const float* X, int n) {
    fwd_layer(L1, X, a1.data(), n, true);
    fwd_layer(L2, a1.data(), a2.data(), n, true);
    fwd_layer(L3, a2.data(), a3.data(), n, true);
    fwd_layer(L4, a3.data(), aout.data(), n, false);
}

// ── the candidate statistic f(y) ────────────────────────────────────────────
// ids: 0 sigmoid(-2y) [sufficient], 1 y [sufficient], 2 sin(2y),
//      3 |y|, 4 sign(y), 5 constant 0 [all insufficient]
int g_fid = 0;

float fstat(float y) {
    switch (g_fid) {
        case 1: return y;
        case 2: return std::sin(2.0f * y);
        case 3: return std::fabs(y);
        case 4: return (y > 0.0f) ? 1.0f : -1.0f;
        case 5: return 0.0f;
        default: return 1.0f / (1.0f + std::exp(2.0f * y));   // sigmoid(-2y)
    }
}

// joint model: Y ~ N(0,1), X = 0.5 Y + N(0,1)
void sample_joint(float* xy, int n) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        const float y = g(rng);
        xy[2 * i] = 0.5f * y + g(rng);   // x
        xy[2 * i + 1] = y;               // y
    }
}

// ── training scratch ────────────────────────────────────────────────────────
std::vector<float> xy0(2 * BATCH), xy1(2 * BATCH);
std::vector<float> Xin(NBUF * D_IN), Ytgt(BATCH);

// ── animation output buffers (read directly from JS) ───────────────────────
std::vector<float> frames_pts((K + 1) * M * 3);      // y (=xi), x, speed
std::vector<float> frames_vec((K + 1) * G * G * 2);  // (0, v) per grid cell
std::vector<float> frames_meanv(K + 1);
std::vector<float> grid_xy(2 * G * G);               // grid xi[], then z[]
std::vector<float> Zx(M), Zy(M), Vz(M);

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int zfc_M() { return M; }
EMSCRIPTEN_KEEPALIVE int zfc_K() { return K; }
EMSCRIPTEN_KEEPALIVE int zfc_G() { return G; }
EMSCRIPTEN_KEEPALIVE float* zfc_pts()   { return frames_pts.data(); }
EMSCRIPTEN_KEEPALIVE float* zfc_vec()   { return frames_vec.data(); }
EMSCRIPTEN_KEEPALIVE float* zfc_meanv() { return frames_meanv.data(); }
EMSCRIPTEN_KEEPALIVE float* zfc_grid()  { return grid_xy.data(); }

EMSCRIPTEN_KEEPALIVE void zfc_init(int fid, unsigned seed) {
    g_fid = fid;
    rng.seed(seed);
    adam_t = 0;
    L1.init(D_IN, H, std::sqrt(2.0f / D_IN));
    L2.init(H, H, std::sqrt(2.0f / H));
    L3.init(H, H, std::sqrt(2.0f / H));
    L4.init(H, D_OUT, std::sqrt(1.0f / H));
    for (int a = 0; a < G; ++a)
        for (int b = 0; b < G; ++b) {
            const int j = a * G + b;
            grid_xy[j] = LO + (HI - LO) * (a + 0.5f) / G;          // xi
            grid_xy[G * G + j] = LO + (HI - LO) * (b + 0.5f) / G;  // z
        }
}

// Run n iterations of the conditional rectified-flow objective (eq. 6);
// returns the last mini-batch loss (mean |X' - X - u|^2).
EMSCRIPTEN_KEEPALIVE float zfc_train(int n) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float loss = 0.0f;
    for (int it = 0; it < n; ++it) {
        sample_joint(xy0.data(), BATCH);   // (X, Y)
        sample_joint(xy1.data(), BATCH);   // (X', Y') independent copy
        for (int j = 0; j < BATCH; ++j) {
            const float x = xy0[2 * j], y = xy0[2 * j + 1];
            const float x2 = xy1[2 * j], y2 = xy1[2 * j + 1];
            const float t = u(rng);
            Xin[j * 4 + 0] = t * x2 + (1 - t) * x;   // X_t
            Xin[j * 4 + 1] = t;
            Xin[j * 4 + 2] = fstat(y2);              // eta = f(Y')
            Xin[j * 4 + 3] = y;                      // xi = Y
            Ytgt[j] = x2 - x;
        }
        net_forward(Xin.data(), BATCH);
        loss = 0.0f;
        const float sc = 2.0f / BATCH;
        for (int j = 0; j < BATCH; ++j) {
            const float e = aout[j] - Ytgt[j];
            loss += e * e;
            dout[j] = sc * e;
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

// Transport M data points (Y_i, X_i) from t=0 to 1 along v_t(.; f(Y_i), Y_i)
// (Euler, K steps; Theorem 3.3) and evaluate the field on the (xi, z) grid at
// every frame; fills the frame buffers.
EMSCRIPTEN_KEEPALIVE void zfc_animate() {
    std::vector<float> xy(2 * M);
    sample_joint(xy.data(), M);
    for (int j = 0; j < M; ++j) { Zx[j] = xy[2 * j]; Zy[j] = xy[2 * j + 1]; }

    const float dt = 1.0f / K;
    for (int k = 0; k <= K; ++k) {
        const float t = (float)k / K;

        // particle velocities, conditioning on the CONSISTENT pair (f(Y_i), Y_i)
        for (int j = 0; j < M; ++j) {
            Xin[j * 4 + 0] = Zx[j];
            Xin[j * 4 + 1] = t;
            Xin[j * 4 + 2] = fstat(Zy[j]);
            Xin[j * 4 + 3] = Zy[j];
        }
        net_forward(Xin.data(), M);
        float mv = 0.0f;
        for (int j = 0; j < M; ++j) {
            Vz[j] = aout[j];
            const float sp = std::fabs(Vz[j]);
            mv += sp;
            float* p = &frames_pts[((size_t)k * M + j) * 3];
            p[0] = Zy[j];   // horizontal: conditioning variable
            p[1] = Zx[j];   // vertical: x
            p[2] = sp;
        }
        frames_meanv[k] = mv / M;

        // grid field v_t(z; f(xi), xi), emitted as (0, v) so the shell can
        // draw vertical arrows with the same code as the unconditional demo
        for (int j = 0; j < G * G; ++j) {
            const float xi = grid_xy[j], z = grid_xy[G * G + j];
            Xin[j * 4 + 0] = z;
            Xin[j * 4 + 1] = t;
            Xin[j * 4 + 2] = fstat(xi);
            Xin[j * 4 + 3] = xi;
        }
        net_forward(Xin.data(), G * G);
        for (int j = 0; j < G * G; ++j) {
            frames_vec[((size_t)k * G * G + j) * 2] = 0.0f;
            frames_vec[((size_t)k * G * G + j) * 2 + 1] = aout[j];
        }

        if (k < K)
            for (int j = 0; j < M; ++j) Zx[j] += Vz[j] * dt;
    }
}

}  // extern "C"

#ifndef __EMSCRIPTEN__
// tiny native smoke test: sufficient vs insufficient statistics
#include <cstdio>
int main() {
    const char* names[] = {"sigmoid(-2y)", "y", "sin(2y)", "|y|", "sign(y)", "0"};
    for (int fid : {0, 2, 5}) {
        zfc_init(fid, 7);
        float loss = 0;
        for (int c = 0; c < 20; ++c) loss = zfc_train(200);
        zfc_animate();
        printf("f = %-12s  loss %.4f  mean|v|: t=0 %.3f  t=0.5 %.3f  t=1 %.3f\n",
               names[fid], loss, frames_meanv[0], frames_meanv[K / 2], frames_meanv[K]);
    }
    return 0;
}
#endif
