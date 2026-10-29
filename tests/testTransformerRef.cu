/**
 * @file testTransformerRef.cu
 * @brief Correctness test for TransformerLayer against an INDEPENDENT reference.
 *
 * Unlike testTransformer.cu (a finite-difference self-consistency check, which
 * passes even if forward/backward agree on the *wrong* math) and
 * testTransformerParity.cu (CPU vs CUDA, which passes if both backends share a
 * bug), this test compares TransformerLayer against a from-scratch reference
 * implementation of the pre-LN multi-head block computed by ml/layer.hpp:
 *
 *     x1 = LN1(x)                                 (pre-norm; gamma/beta, eps=1e-5)
 *     per head h over its own d_h = d_k/num_heads rows of Q,K,V = W{q,k,v}·x1:
 *       scores[q,k] = (Q_hq · K_hk) / sqrt(d_h)   with causal mask (k > q → -inf)
 *       A           = softmax(scores, dim=keys)   (ROW-wise: each query row sums to 1)
 *       H_h[:,q]    = Σ_k A[q,k] · V_h[:,k]
 *     R   = x + (Wo·H + bo)                       (residual 1)
 *     out = R + (W2·relu(W1·LN2(R) + b1) + b2)    (residual 2)
 *
 * The reference is written as explicit double-precision loops (causal mask by
 * construction, explicit row-softmax, per-head slices), so it shares no code
 * path with the layer under test. Double precision matters for the gradient
 * check: a float32 reference differentiated with a large eps is unreliable
 * whenever a ReLU pre-activation lies within eps of its kink, which made the
 * test flaky across random seeds/backends.
 *
 * Forward is compared directly; the input gradient is compared against the
 * numerical (central-difference) gradient of the double reference forward.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using namespace Juzhen;

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

// Minimal column-major double matrix for the reference computation.
struct DMat {
    int r = 0, c = 0;
    std::vector<double> v;
    DMat() = default;
    DMat(int r, int c) : r(r), c(c), v((size_t)r * c, 0.0) {}
    double& at(int i, int j) { return v[(size_t)j * r + i]; }
    double at(int i, int j) const { return v[(size_t)j * r + i]; }
};

static DMat to_d(const Matrix<float>& m) {
    DMat d((int)m.num_row(), (int)m.num_col());
    for (int j = 0; j < d.c; ++j)
        for (int i = 0; i < d.r; ++i)
            d.at(i, j) = (double)m.elem(i, j);
    return d;
}

static DMat mm(const DMat& A, const DMat& B) {
    DMat C(A.r, B.c);
    for (int j = 0; j < B.c; ++j)
        for (int k = 0; k < A.c; ++k) {
            const double b = B.at(k, j);
            for (int i = 0; i < A.r; ++i) C.at(i, j) += A.at(i, k) * b;
        }
    return C;
}

struct TFWeights {
    DMat Wq, Wk, Wv, Wo, bo, W1, b1, W2, b2;
    DMat ln1_g, ln1_b, ln2_g, ln2_b;
};

struct TFConfig { int d_model, d_k, d_ff, seq, batch, num_heads; };

// Independent LayerNorm over rows (features) of each column/token.
static DMat layernorm(const DMat& x, const DMat& g, const DMat& b) {
    DMat y(x.r, x.c);
    for (int c = 0; c < x.c; ++c) {
        double mu = 0.0;
        for (int r = 0; r < x.r; ++r) mu += x.at(r, c);
        mu /= x.r;
        double var = 0.0;
        for (int r = 0; r < x.r; ++r) { const double d = x.at(r, c) - mu; var += d * d; }
        var /= x.r;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int r = 0; r < x.r; ++r)
            y.at(r, c) = g.at(r, 0) * ((x.at(r, c) - mu) * inv) + b.at(r, 0);
    }
    return y;
}

// Independent reference forward in double precision.
// Multi-head: each head attends over its own d_h = d_k/num_heads rows of Q/K/V.
static DMat ref_forward(const TFWeights& w, const TFConfig& c, const DMat& x) {
    const int N = c.seq * c.batch;
    const int d_h = c.d_k / c.num_heads;
    const double scale = 1.0 / std::sqrt((double)d_h);

    auto x1 = layernorm(x, w.ln1_g, w.ln1_b);   // LN before attention
    auto Q = mm(w.Wq, x1);   // (d_k, N)
    auto K = mm(w.Wk, x1);   // (d_k, N)
    auto V = mm(w.Wv, x1);   // (d_k, N)

    DMat H(c.d_k, N);

    for (int hh = 0; hh < c.num_heads; ++hh) {
        const int r0 = hh * d_h;
        for (int b = 0; b < c.batch; ++b) {
            const int c0 = b * c.seq;
            for (int q = 0; q < c.seq; ++q) {
                // Causal: query q attends only to keys k <= q.
                std::vector<double> a(q + 1, 0.0);
                double m = -1e30;
                for (int k = 0; k <= q; ++k) {
                    double dot = 0.0;
                    for (int d = 0; d < d_h; ++d)
                        dot += Q.at(r0 + d, c0 + q) * K.at(r0 + d, c0 + k);
                    a[k] = dot * scale;
                    if (a[k] > m) m = a[k];
                }
                double s = 0.0;
                for (int k = 0; k <= q; ++k) { a[k] = std::exp(a[k] - m); s += a[k]; }
                for (int k = 0; k <= q; ++k) a[k] /= s;   // row-softmax over keys

                for (int d = 0; d < d_h; ++d) {
                    double acc = 0.0;
                    for (int k = 0; k <= q; ++k) acc += a[k] * V.at(r0 + d, c0 + k);
                    H.at(r0 + d, c0 + q) = acc;
                }
            }
        }
    }

    auto O = mm(w.Wo, H);    // (d_model, N)
    DMat R(c.d_model, N);
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_model; ++r)
            R.at(r, col) = x.at(r, col) + O.at(r, col) + w.bo.at(r, 0);  // residual 1

    auto x2 = layernorm(R, w.ln2_g, w.ln2_b);   // LN before FFN
    auto F1 = mm(w.W1, x2);  // (d_ff, N)
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_ff; ++r) {
            const double v = F1.at(r, col) + w.b1.at(r, 0);
            F1.at(r, col) = v > 0.0 ? v : 0.0;   // relu
        }

    auto F2 = mm(w.W2, F1);  // (d_model, N)
    DMat out(c.d_model, N);
    for (int col = 0; col < N; ++col)
        for (int r = 0; r < c.d_model; ++r)
            out.at(r, col) = R.at(r, col) + F2.at(r, col) + w.b2.at(r, 0);  // residual 2

    return out;
}

static double ref_loss(const TFWeights& w, const TFConfig& c,
                       const DMat& x, const DMat& g) {
    auto out = ref_forward(w, c, x);
    double s = 0.0;
    for (int col = 0; col < out.c; ++col)
        for (int r = 0; r < out.r; ++r)
            s += out.at(r, col) * g.at(r, col);
    return s;
}

static void compare_matrix(const Matrix<float>& got, const DMat& ref,
                           const std::string& tag, float& max_abs, float& rel_l2) {
    max_abs = 0.0f;
    double num = 0, den = 0;
    for (int c = 0; c < ref.c; ++c)
        for (int r = 0; r < ref.r; ++r) {
            const float d = std::fabs(got.elem(r, c) - (float)ref.at(r, c));
            if (d > max_abs) max_abs = d;
            num += (double)d * d;
            den += ref.at(r, c) * ref.at(r, c);
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
        to_d(as_host(tf.get_Wq())), to_d(as_host(tf.get_Wk())), to_d(as_host(tf.get_Wv())),
        to_d(as_host(tf.get_Wo())), to_d(as_host(tf.get_bo())),
        to_d(as_host(tf.get_W1())), to_d(as_host(tf.get_b1())),
        to_d(as_host(tf.get_W2())), to_d(as_host(tf.get_b2())),
        to_d(as_host(tf.get_ln1_gamma())), to_d(as_host(tf.get_ln1_beta())),
        to_d(as_host(tf.get_ln2_gamma())), to_d(as_host(tf.get_ln2_beta()))
    };

    auto x_h = Matrix<float>::randn(c.d_model, N) * 0.5f;
    auto g_h = Matrix<float>::randn(c.d_model, N) * 0.5f;
    const DMat x_d = to_d(x_h);
    const DMat g_d = to_d(g_h);

    std::list<Layer<BackendT>*> layers = {&tf};
    freeze(layers);   // keep weights fixed across forward/backward

    int ret = 0;
    float max_abs, rel_l2;

    // ── Forward vs independent reference ────────────────────────────────
    tf.eval(Matrix<BackendT>(x_h));
    auto got_out = as_host(tf.value());
    auto ref_out = ref_forward(w, c, x_d);

    compare_matrix(got_out, ref_out, "forward vs reference", max_abs, rel_l2);
    if (max_abs > 1e-4f || rel_l2 > 1e-4f) {
        std::cout << "[FAIL] forward vs reference\n";
        ret = 1;
    } else {
        std::cout << "[PASS] forward vs reference\n";
    }

    // ── Backward (dx) vs numerical gradient of the reference ────────────
    // eps can be small because the reference runs in double; this keeps the
    // check accurate even when a ReLU pre-activation is near its kink.
    tf.eval(Matrix<BackendT>(x_h));
    auto got_dx = as_host(tf.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(g_h)));

    const double eps = 1e-6;
    DMat num_dx(c.d_model, N);
    for (int col = 0; col < N; ++col) {
        for (int r = 0; r < c.d_model; ++r) {
            DMat xp = x_d, xm = x_d;
            xp.at(r, col) += eps;
            xm.at(r, col) -= eps;
            num_dx.at(r, col) = (ref_loss(w, c, xp, g_d) - ref_loss(w, c, xm, g_d))
                                / (2.0 * eps);
        }
    }

    compare_matrix(got_dx, num_dx, "backward (dx) vs reference", max_abs, rel_l2);
    if (max_abs > 1e-3f || rel_l2 > 1e-3f) {
        std::cout << "[FAIL] backward (dx) vs reference\n";
        ret = 1;
    } else {
        std::cout << "[PASS] backward (dx) vs reference\n";
    }

    unfreeze(layers);
    return ret;
}
