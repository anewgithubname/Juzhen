/**
 * @file testTransformerRef.cu
 * @brief Correctness test for TransformerLayer against an INDEPENDENT reference.
 *
 * Unlike testTransformer.cu (a finite-difference self-consistency check, which
 * passes even if forward/backward agree on the *wrong* math) and
 * testTransformerParity.cu (CPU vs CUDA, which passes if both backends share a
 * bug), this test compares TransformerLayer against a from-scratch reference
 * implementation of the attention formula used by examples/demo_transformer.py:
 *
 *     scores[q,k] = (Q_q · K_k) / sqrt(d_k)     with causal mask (k > q → -inf)
 *     A          = softmax(scores, dim=keys)     (ROW-wise: each query row sums to 1)
 *     H[:,q]     = Σ_k A[q,k] · V[:,k]
 *     R          = x + (Wo·H + bo)
 *     out        = R + (W2·relu(W1·R + b1) + b2)
 *
 * The reference uses explicit loops (causal mask by construction, explicit
 * row-softmax) so it shares no code path with the layer under test.
 *
 * Forward is compared directly; the input gradient is compared against the
 * numerical gradient of the reference forward.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cmath>
#include <iostream>
#include <list>
#include <vector>

using namespace Juzhen;

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

struct TFWeights {
    Matrix<float> Wq, Wk, Wv, Wo, bo, W1, b1, W2, b2;
    Matrix<float> ln1_g, ln1_b, ln2_g, ln2_b;
};

struct TFConfig { int d_model, d_k, d_ff, seq, batch, num_heads; };

// Independent LayerNorm over rows (features) of each column/token.
static Matrix<float> layernorm(const Matrix<float>& x,
                               const Matrix<float>& g, const Matrix<float>& b) {
    const int D = (int)x.num_row(), N = (int)x.num_col();
    Matrix<float> y("ln_ref", D, N);
    for (int c = 0; c < N; ++c) {
        float mu = 0.0f;
        for (int r = 0; r < D; ++r) mu += x.elem(r, c);
        mu /= D;
        float var = 0.0f;
        for (int r = 0; r < D; ++r) { float d = x.elem(r, c) - mu; var += d * d; }
        var /= D;
        const float inv = 1.0f / std::sqrt(var + 1e-5f);
        for (int r = 0; r < D; ++r)
            y.elem(r, c) = g.elem(r, 0) * ((x.elem(r, c) - mu) * inv) + b.elem(r, 0);
    }
    return y;
}

// Independent reference forward. All matrices are host Matrix<float>.
// Multi-head: each head attends over its own d_h = d_k/num_heads rows of Q/K/V.
static Matrix<float> ref_forward(const TFWeights& w, const TFConfig& c,
                                 const Matrix<float>& x) {
    const int N = c.seq * c.batch;
    const int d_h = c.d_k / c.num_heads;
    const float scale = 1.0f / std::sqrt((float)d_h);

    auto x1 = layernorm(x, w.ln1_g, w.ln1_b);   // LN before attention
    auto Q = w.Wq * x1;   // (d_k, N)
    auto K = w.Wk * x1;   // (d_k, N)
    auto V = w.Wv * x1;   // (d_k, N)

    Matrix<float> H("Href", c.d_k, N);
    H.zeros();

    for (int hh = 0; hh < c.num_heads; ++hh) {
        const int r0 = hh * d_h;
        for (int b = 0; b < c.batch; ++b) {
            const int c0 = b * c.seq;
            for (int q = 0; q < c.seq; ++q) {
                // Causal: query q attends only to keys k <= q.
                std::vector<float> a(q + 1, 0.0f);
                float m = -1e30f;
                for (int k = 0; k <= q; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_h; ++d)
                        dot += Q.elem(r0 + d, c0 + q) * K.elem(r0 + d, c0 + k);
                    a[k] = dot * scale;
                    if (a[k] > m) m = a[k];
                }
                float s = 0.0f;
                for (int k = 0; k <= q; ++k) { a[k] = std::exp(a[k] - m); s += a[k]; }
                for (int k = 0; k <= q; ++k) a[k] /= s;   // row-softmax over keys

                for (int d = 0; d < d_h; ++d) {
                    float acc = 0.0f;
                    for (int k = 0; k <= q; ++k) acc += a[k] * V.elem(r0 + d, c0 + k);
                    H.elem(r0 + d, c0 + q) = acc;
                }
            }
        }
    }

    auto O = w.Wo * H;   // (d_model, N)
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_model; ++r)
            O.elem(r, col) += w.bo.elem(r, 0);

    auto R = x + O;      // (d_model, N) — residual 1

    auto x2 = layernorm(R, w.ln2_g, w.ln2_b);   // LN before FFN
    auto F1 = w.W1 * x2; // (d_ff, N)
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_ff; ++r) {
            float v = F1.elem(r, col) + w.b1.elem(r, 0);
            F1.elem(r, col) = v > 0.0f ? v : 0.0f;   // relu
        }

    auto F2 = w.W2 * F1; // (d_model, N)
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_model; ++r)
            F2.elem(r, col) += w.b2.elem(r, 0);

    return R + F2;
}

static float ref_loss(const TFWeights& w, const TFConfig& c,
                      const Matrix<float>& x, const Matrix<float>& g) {
    auto out = ref_forward(w, c, x);
    float s = 0.0f;
    for (size_t col = 0; col < out.num_col(); ++col)
        for (size_t r = 0; r < out.num_row(); ++r)
            s += out.elem(r, col) * g.elem(r, col);
    return s;
}

static void compare_matrix(const Matrix<float>& got, const Matrix<float>& ref,
                           const std::string& tag, float& max_abs, float& rel_l2) {
    max_abs = 0.0f;
    double num = 0, den = 0;
    for (size_t c = 0; c < ref.num_col(); ++c)
        for (size_t r = 0; r < ref.num_row(); ++r) {
            float d = std::fabs(got.elem(r, c) - ref.elem(r, c));
            if (d > max_abs) max_abs = d;
            num += (double)d * d;
            den += (double)ref.elem(r, c) * ref.elem(r, c);
        }
    rel_l2 = (float)std::sqrt(num / std::max(den, 1e-12));
    std::cout << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2 << "\n";
}

int compute() {
    global_rand_gen.seed(7);

#if defined(CUDA)
    GPUSampler sampler(7);
    using BackendT = CUDAfloat;
#elif defined(ROCM_HIP)
    using BackendT = ROCMfloat;
#elif defined(APPLE_SILICON)
    using BackendT = MPSfloat;
#else
    using BackendT = float;
#endif

    const TFConfig c{8, 8, 16, 5, 3, 4};   // d_k=8, num_heads=4 -> d_h=2
    const int N = c.seq * c.batch;

    TransformerLayer<BackendT> tf(c.d_model, c.d_k, c.d_ff, c.seq, c.batch, c.num_heads);
    std::cout << "num_heads=" << tf.heads() << " (d_h=" << c.d_k / c.num_heads << ")\n";

    // Pull the layer's (random) weights out to host for the reference.
    TFWeights w{
        as_host(tf.get_Wq()), as_host(tf.get_Wk()), as_host(tf.get_Wv()),
        as_host(tf.get_Wo()), as_host(tf.get_bo()),
        as_host(tf.get_W1()), as_host(tf.get_b1()),
        as_host(tf.get_W2()), as_host(tf.get_b2()),
        as_host(tf.get_ln1_gamma()), as_host(tf.get_ln1_beta()),
        as_host(tf.get_ln2_gamma()), as_host(tf.get_ln2_beta())
    };

    auto x_h = Matrix<float>::randn(c.d_model, N) * 0.5f;
    auto g_h = Matrix<float>::randn(c.d_model, N) * 0.5f;

    std::list<Layer<BackendT>*> layers = {&tf};
    freeze(layers);   // keep weights fixed across forward/backward

    int ret = 0;
    float max_abs, rel_l2;

    // ── Forward vs independent reference ────────────────────────────────
    tf.eval(Matrix<BackendT>(x_h));
    auto got_out = as_host(tf.value());
    auto ref_out = ref_forward(w, c, x_h);

    compare_matrix(got_out, ref_out, "forward vs reference", max_abs, rel_l2);
    if (max_abs > 1e-4f || rel_l2 > 1e-4f) {
        std::cout << "[FAIL] forward vs reference\n";
        ret = 1;
    } else {
        std::cout << "[PASS] forward vs reference\n";
    }

    // ── Backward (dx) vs numerical gradient of the reference ────────────
    tf.eval(Matrix<BackendT>(x_h));
    auto got_dx = as_host(tf.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(g_h)));

    const float eps = 1e-3f;
    Matrix<float> num_dx("num_dx", c.d_model, N);
    for (int col = 0; col < N; ++col) {
        for (int r = 0; r < c.d_model; ++r) {
            auto xp = Matrix<float>(x_h);
            auto xm = Matrix<float>(x_h);
            xp.elem(r, col) += eps;
            xm.elem(r, col) -= eps;
            float lp = ref_loss(w, c, xp, g_h);
            float lm = ref_loss(w, c, xm, g_h);
            num_dx.elem(r, col) = (lp - lm) / (2.0f * eps);
        }
    }

    compare_matrix(got_dx, num_dx, "backward (dx) vs reference", max_abs, rel_l2);
    if (max_abs > 1e-2f || rel_l2 > 1e-2f) {
        std::cout << "[FAIL] backward (dx) vs reference\n";
        ret = 1;
    } else {
        std::cout << "[PASS] backward (dx) vs reference\n";
    }

    unfreeze(layers);
    return ret;
}
