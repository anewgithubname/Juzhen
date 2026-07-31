/**
 * @file testJVP.cu
 * @brief Forward-mode (JVP) correctness on the CPU backend.
 *
 * Two independent checks per network:
 *   (a) adjoint consistency against the existing backprop:
 *           <W, J u> == <J^T W, u>
 *       where J^T W comes from the free function grad() — both sides
 *       linearize at the same point, so agreement is at fp32 rounding level;
 *   (b) central finite differences along u (looser: fp32 + ReLU kinks).
 *
 * Covers an MLP (tanh -> relu -> linear), a mixed CNN+MLP, and a
 * conv-transpose network, exercising the base-class rule, the conv
 * overrides, and the network-level driver.
 */

#include "../ml/layer.hpp"
#include <cmath>
#include <iostream>

using namespace Juzhen;
using namespace std;

#if defined(CUDA) || defined(APPLE_SILICON) || defined(ROCM_HIP)
int compute() {
    cout << "testJVP requires a CPU-only build; skipping." << endl;
    return 77; // ctest SKIP_RETURN_CODE
}
#else

namespace {

float dot(const Matrix<float>& A, const Matrix<float>& B) {
    return sum(sum(hadmd(A, B), 0), 1).elem(0, 0);
}

float fro(const Matrix<float>& A) { return std::sqrt(dot(A, A)); }

int check_network(list<Layer<float>*> nn, int d_in, int nb, const char* name,
                  float eps = 1e-2f) {
    auto x = Matrix<float>::randn(d_in, nb);
    auto u = Matrix<float>::randn(d_in, nb); // input tangent direction

    auto Ju = jvp(nn, x, u);

    // (a) adjoint consistency with backprop.
    auto W = Matrix<float>::randn(Ju.num_row(), Ju.num_col());
    auto JTW = grad(nn, x, W);
    const float lhs = dot(W, Ju), rhs = dot(JTW, u);
    const float adj_err = std::abs(lhs - rhs) / (std::abs(lhs) + 1e-12f);

    // (b) central finite differences along u.
    Matrix<float> fp = forward(nn, x + eps * u);
    Matrix<float> fm = forward(nn, x - eps * u);
    auto fd = (fp - fm) / (2.0 * eps);
    const float fd_err = fro(fd - Ju) / (fro(Ju) + 1e-12f);

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

} // namespace

int compute() {
    const int nb = 4;

    // ── MLP: tanh -> relu -> linear ─────────────────────────────────────
    // Rescale weights: the default 1e-3 init makes outputs vanishingly small.
    Layer<float> L0(8, 6, nb);
    ReluLayer<float> L1(7, 8, nb);
    LinearLayer<float> L2(3, 7, nb);
    L0.W() = Matrix<float>::randn(8, 6) * 0.7f; L0.b() = Matrix<float>::randn(8, 1) * 0.2f;
    L1.W() = Matrix<float>::randn(7, 8) * 0.7f; L1.b() = Matrix<float>::randn(7, 1) * 0.2f;
    L2.W() = Matrix<float>::randn(3, 7) * 0.7f; L2.b() = Matrix<float>::randn(3, 1) * 0.2f;
    list<Layer<float>*> mlp = { &L2, &L1, &L0 };
    if (check_network(mlp, 6, nb, "mlp")) return 1;

    // ── CNN + MLP: conv -> conv -> relu-fc -> linear ────────────────────
    ConvLayer conv1(nb, 2, 6, 6, 4, 3, 3, 1, 1, true);  // -> 4 x 6 x 6
    ConvLayer conv2(nb, 4, 6, 6, 3, 3, 3, 0, 1, true);  // -> 3 x 4 x 4
    ReluLayer<float> F1(10, 3 * 4 * 4, nb);
    LinearLayer<float> F2(2, 10, nb);
    conv1.W() = Matrix<float>::randn(conv1.W().num_row(), 1) * 0.4f;
    conv1.b() = Matrix<float>::randn(conv1.b().num_row(), 1) * 0.1f;
    conv2.W() = Matrix<float>::randn(conv2.W().num_row(), 1) * 0.4f;
    conv2.b() = Matrix<float>::randn(conv2.b().num_row(), 1) * 0.1f;
    F1.W() = Matrix<float>::randn(10, 48) * 0.5f; F1.b() = Matrix<float>::randn(10, 1) * 0.2f;
    F2.W() = Matrix<float>::randn(2, 10) * 0.5f;  F2.b() = Matrix<float>::randn(2, 1) * 0.2f;
    list<Layer<float>*> cnn = { &F2, &F1, &conv2, &conv1 };
    if (check_network(cnn, 2 * 6 * 6, nb, "cnn+mlp")) return 1;

    // ── conv-transpose -> linear ────────────────────────────────────────
    convtransLayer ct(nb, 2, 4, 4, 3, 3, 3, 0, 1, true); // -> 3 x 6 x 6
    ct.W() = Matrix<float>::randn(ct.W().num_row(), 1) * 0.4f;
    ct.b() = Matrix<float>::randn(ct.b().num_row(), 1) * 0.1f;
    LinearLayer<float> G(2, 3 * 6 * 6, nb);
    G.W() = Matrix<float>::randn(2, 108) * 0.5f; G.b() = Matrix<float>::randn(2, 1) * 0.2f;
    list<Layer<float>*> tnet = { &G, &ct };
    if (check_network(tnet, 2 * 4 * 4, nb, "convtrans+linear")) return 1;

    // ── transformer (causal, multi-head) ────────────────────────────────
    // Default 1/sqrt(dim) weight init is already well-scaled. Input width
    // is d_model, "batch" is seq_len * batch sequences of tokens.
    const int d_model = 6, d_kk = 8, d_ff = 10, seq_len = 5, batch = 3, heads = 2;
    TransformerLayer<float> tf(d_model, d_kk, d_ff, seq_len, batch, heads);
    // Smaller FD step: the LN/softmax composition has enough curvature that
    // eps=1e-2 second-order error swamps the tolerance (the adjoint check
    // is exact regardless; central-diff error shrinks as O(eps^2)).
    list<Layer<float>*> tfnet = { &tf };
    if (check_network(tfnet, d_model, seq_len * batch, "transformer", 2e-3f)) return 1;

    // ── transformer -> linear head (mixed composition) ──────────────────
    LinearLayer<float> head(2, d_model, seq_len * batch);
    head.W() = Matrix<float>::randn(2, d_model) * 0.5f;
    head.b() = Matrix<float>::randn(2, 1) * 0.2f;
    list<Layer<float>*> tfhead = { &head, &tf };
    if (check_network(tfhead, d_model, seq_len * batch, "transformer+linear", 2e-3f)) return 1;

    cout << "All JVP tests passed." << endl;
    return 0;
}

#endif
