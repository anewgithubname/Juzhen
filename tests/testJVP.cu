/**
 * @file testJVP.cu
 * @brief Forward-mode (JVP) correctness on the CPU and CUDA backends.
 *
 * Two independent checks per network:
 *   (a) adjoint consistency against the existing backprop:
 *           <W, J u> == <J^T W, u>
 *       where J^T W comes from the free function grad() — both sides
 *       linearize at the same point, so agreement is at fp32 rounding level;
 *   (b) central finite differences along u (looser: fp32 + ReLU kinks).
 *
 * The templated networks (MLP: tanh -> relu -> linear, and the transformer)
 * run on the CPU backend in every build and additionally on CUDAfloat in
 * CUDA builds; the conv / conv-transpose networks use whichever ConvLayer
 * the build provides (im2col on CPU, cuDNN on CUDA). CUDA builds also check
 * CPU<->GPU parity of the JVP itself on networks with identical weights.
 */

#include "../ml/layer.hpp"
#include <cmath>
#include <iostream>

using namespace Juzhen;
using namespace std;

#if defined(APPLE_SILICON) || defined(ROCM_HIP)
int compute() {
    cout << "testJVP covers the CPU and CUDA backends only; skipping." << endl;
    return 77; // ctest SKIP_RETURN_CODE
}
#else

namespace {

template <class D>
float dotf(const Matrix<D>& A, const Matrix<D>& B) {
    return item(sum(sum(hadmd(A, B), 0), 1));
}

template <class D>
float fro(const Matrix<D>& A) { return std::sqrt(dotf(A, A)); }

template <class D>
int check_network(list<Layer<D>*> nn, int d_in, int nb, const string& name,
                  float eps = 1e-2f) {
    auto x = Matrix<D>::randn(d_in, nb);
    auto u = Matrix<D>::randn(d_in, nb); // input tangent direction

    auto Ju = jvp(nn, x, u);

    // (a) adjoint consistency with backprop.
    auto W = Matrix<D>::randn(Ju.num_row(), Ju.num_col());
    auto JTW = grad(nn, x, W);
    const float lhs = dotf(W, Ju), rhs = dotf(JTW, u);
    const float adj_err = std::abs(lhs - rhs) / (std::abs(lhs) + 1e-12f);

    // (b) central finite differences along u. Tried at shrinking steps and
    // scored by the best: when a ReLU pre-activation happens to fall within
    // eps of its kink, the central difference straddles the kink and is off
    // by O(1) at that entry no matter how correct the JVP is — but that
    // artifact disappears once eps drops below the kink distance, whereas a
    // genuine JVP bug stays at every step size.
    float fd_err = FLT_MAX;
    for (float e : { eps, eps * 0.25f, eps * 0.0625f }) {
        Matrix<D> fp = forward(nn, x + e * u);
        Matrix<D> fm = forward(nn, x - e * u);
        auto fd = (fp - fm) * (0.5f / e);
        fd_err = std::min(fd_err, fro(fd - Ju) / (fro(Ju) + 1e-12f));
    }

    cout << name << ": adjoint rel err = " << adj_err
         << ", finite-diff rel err = " << fd_err << endl;

    if (!(adj_err < 1e-4f)) {
        cout << name << " FAILED the adjoint consistency check." << endl;
        return 1;
    }
    if (!(fd_err < 5e-2f)) {
        cout << name << " FAILED the finite-difference check." << endl;
        return 1;
    }
    return 0;
}

// ── MLP: tanh -> relu -> linear ─────────────────────────────────────────
// Rescale weights: the default 1e-3 init makes outputs vanishingly small.
template <class D>
int run_mlp(const string& tag) {
    const int nb = 4;
    Layer<D> L0(8, 6, nb);
    ReluLayer<D> L1(7, 8, nb);
    LinearLayer<D> L2(3, 7, nb);
    L0.W() = Matrix<D>::randn(8, 6) * 0.7f; L0.b() = Matrix<D>::randn(8, 1) * 0.2f;
    L1.W() = Matrix<D>::randn(7, 8) * 0.7f; L1.b() = Matrix<D>::randn(7, 1) * 0.2f;
    L2.W() = Matrix<D>::randn(3, 7) * 0.7f; L2.b() = Matrix<D>::randn(3, 1) * 0.2f;
    list<Layer<D>*> mlp = { &L2, &L1, &L0 };
    return check_network<D>(mlp, 6, nb, tag + ":mlp");
}

// ── transformer (causal, multi-head), alone and under a linear head ─────
// Default 1/sqrt(dim) weight init is already well-scaled. Input width is
// d_model, "batch" is seq_len * batch sequences of tokens. Smaller FD step:
// the LN/softmax composition has enough curvature that eps=1e-2 second-order
// error swamps the tolerance (the adjoint check is exact regardless;
// central-diff error shrinks as O(eps^2)).
template <class D>
int run_transformer(const string& tag) {
    const int d_model = 6, d_kk = 8, d_ff = 10, seq_len = 5, batch = 3, heads = 2;
    TransformerLayer<D> tf(d_model, d_kk, d_ff, seq_len, batch, heads);
    list<Layer<D>*> tfnet = { &tf };
    if (check_network<D>(tfnet, d_model, seq_len * batch, tag + ":transformer", 2e-3f)) return 1;

    LinearLayer<D> head(2, d_model, seq_len * batch);
    head.W() = Matrix<D>::randn(2, d_model) * 0.5f;
    head.b() = Matrix<D>::randn(2, 1) * 0.2f;
    list<Layer<D>*> tfhead = { &head, &tf };
    return check_network<D>(tfhead, d_model, seq_len * batch, tag + ":transformer+linear", 2e-3f);
}

#if !defined(CUDA) || defined(CUDNN_AVAILABLE)
#define JVP_HAVE_CONV 1
#ifdef CUDA
using ConvD = CUDAfloat;
#else
using ConvD = float;
#endif

// ── conv nets: conv -> conv -> relu-fc -> linear, and conv-transpose ────
int run_conv(const string& tag) {
    using D = ConvD;
    const int nb = 4;

    ConvLayer conv1(nb, 2, 6, 6, 4, 3, 3, 1, 1, true);  // -> 4 x 6 x 6
    ConvLayer conv2(nb, 4, 6, 6, 3, 3, 3, 0, 1, true);  // -> 3 x 4 x 4
    ReluLayer<D> F1(10, 3 * 4 * 4, nb);
    LinearLayer<D> F2(2, 10, nb);
    conv1.W() = Matrix<D>::randn(conv1.W().num_row(), 1) * 0.4f;
    conv1.b() = Matrix<D>::randn(conv1.b().num_row(), 1) * 0.1f;
    conv2.W() = Matrix<D>::randn(conv2.W().num_row(), 1) * 0.4f;
    conv2.b() = Matrix<D>::randn(conv2.b().num_row(), 1) * 0.1f;
    F1.W() = Matrix<D>::randn(10, 48) * 0.5f; F1.b() = Matrix<D>::randn(10, 1) * 0.2f;
    F2.W() = Matrix<D>::randn(2, 10) * 0.5f;  F2.b() = Matrix<D>::randn(2, 1) * 0.2f;
    list<Layer<D>*> cnn = { &F2, &F1, &conv2, &conv1 };
    if (check_network<D>(cnn, 2 * 6 * 6, nb, tag + ":cnn+mlp")) return 1;

    convtransLayer ct(nb, 2, 4, 4, 3, 3, 3, 0, 1, true); // -> 3 x 6 x 6
    ct.W() = Matrix<D>::randn(ct.W().num_row(), 1) * 0.4f;
    ct.b() = Matrix<D>::randn(ct.b().num_row(), 1) * 0.1f;
    LinearLayer<D> G(2, 3 * 6 * 6, nb);
    G.W() = Matrix<D>::randn(2, 108) * 0.5f; G.b() = Matrix<D>::randn(2, 1) * 0.2f;
    list<Layer<D>*> tnet = { &G, &ct };
    return check_network<D>(tnet, 2 * 4 * 4, nb, tag + ":convtrans+linear");
}
#endif // conv available

#ifdef CUDA
// Copy every trainable tensor of a CPU layer into its GPU twin.
template <class LSrc, class LDst>
void copy_params(LSrc& src, LDst& dst) {
    auto ps = src.checkpoint_parameters();
    auto pd = dst.checkpoint_parameters();
    for (size_t i = 0; i < ps.size(); ++i)
        *pd[i].second = Matrix<CUDAfloat>(*ps[i].second);
}

int check_parity(list<Layer<float>*> cnn, list<Layer<CUDAfloat>*> gnn,
                 int d_in, int nb, const string& name) {
    auto x = Matrix<float>::randn(d_in, nb);
    auto u = Matrix<float>::randn(d_in, nb);
    auto Jc = jvp(cnn, x, u);
    auto Jg = jvp(gnn, Matrix<CUDAfloat>(x), Matrix<CUDAfloat>(u)).to_host();
    const float err = fro(Jc - Jg) / (fro(Jc) + 1e-12f);
    cout << name << ": cpu-vs-cuda rel err = " << err << endl;
    if (!(err < 1e-4f)) {
        cout << name << " FAILED the CPU/CUDA parity check." << endl;
        return 1;
    }
    return 0;
}

// ── same weights on both backends: JVP outputs must agree ───────────────
int run_parity() {
    const int nb = 4;

    Layer<float> c0(8, 6, nb);
    ReluLayer<float> c1(7, 8, nb);
    LinearLayer<float> c2(3, 7, nb);
    c0.W() = Matrix<float>::randn(8, 6) * 0.7f; c0.b() = Matrix<float>::randn(8, 1) * 0.2f;
    c1.W() = Matrix<float>::randn(7, 8) * 0.7f; c1.b() = Matrix<float>::randn(7, 1) * 0.2f;
    c2.W() = Matrix<float>::randn(3, 7) * 0.7f; c2.b() = Matrix<float>::randn(3, 1) * 0.2f;
    Layer<CUDAfloat> g0(8, 6, nb);
    ReluLayer<CUDAfloat> g1(7, 8, nb);
    LinearLayer<CUDAfloat> g2(3, 7, nb);
    copy_params(c0, g0); copy_params(c1, g1); copy_params(c2, g2);
    if (check_parity({ &c2, &c1, &c0 }, { &g2, &g1, &g0 }, 6, nb, "parity:mlp")) return 1;

    const int d_model = 6, d_kk = 8, d_ff = 10, seq_len = 5, batch = 3, heads = 2;
    TransformerLayer<float> tc(d_model, d_kk, d_ff, seq_len, batch, heads);
    TransformerLayer<CUDAfloat> tg(d_model, d_kk, d_ff, seq_len, batch, heads);
    copy_params(tc, tg);
    return check_parity({ &tc }, { &tg }, d_model, seq_len * batch, "parity:transformer");
}
#endif // CUDA

} // namespace

int compute() {
    global_rand_gen.seed(123);
#ifdef CUDA
    GPUSampler sampler(123); // initializes the cuRAND generator behind Matrix<CUDAfloat>::randn
#endif
    int rc = 0;

    // Templated layers always have a CPU (float) instantiation, even in
    // CUDA builds — so every build exercises the generic JVP path.
    rc |= run_mlp<float>("cpu");
    rc |= run_transformer<float>("cpu");
#if defined(JVP_HAVE_CONV) && !defined(CUDA)
    rc |= run_conv("cpu");
#endif

#ifdef CUDA
    rc |= run_mlp<CUDAfloat>("cuda");
    rc |= run_transformer<CUDAfloat>("cuda");
#ifdef JVP_HAVE_CONV
    rc |= run_conv("cuda");
#endif
    rc |= run_parity();
#endif

    if (rc == 0) cout << "All JVP tests passed." << endl;
    return rc;
}

#endif
