/**
 * @file testTransformer.cu
 * @brief Unit tests for TransformerLayer: forward shape, finite-difference
 *        gradient check for the input gradient.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cmath>
#include <iostream>
#include <list>

using namespace Juzhen;

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

static void compare_matrix(const Matrix<float>& got,
                           const Matrix<float>& ref,
                           const std::string& tag,
                           float& max_abs, float& rel_l2) {
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

static float scalar_loss(const Matrix<float>& y, const Matrix<float>& g) {
    return item(sum(sum(hadmd(y, g), 0), 1));
}

template <class BackendT>
static float forward_loss_host(TransformerLayer<BackendT>& layer,
                               const Matrix<float>& x_h,
                               const Matrix<float>& upstream_h) {
    layer.eval(Matrix<BackendT>(x_h));
    return scalar_loss(as_host(layer.value()), upstream_h);
}

template <class BackendT>
static int check_input_gradient_fd(const std::string& name,
                                   TransformerLayer<BackendT>& layer,
                                   const Matrix<float>& x_h,
                                   const Matrix<float>& upstream_h,
                                   float eps,
                                   float max_abs_tol,
                                   float rel_l2_tol) {
    std::list<Layer<BackendT>*> layers;
    layers.push_back(&layer);
    freeze(layers);

    layer.eval(Matrix<BackendT>(x_h));
    auto dx_h = as_host(
        layer.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(upstream_h)));

    Matrix<float> num("num_dx", x_h.num_row(), x_h.num_col());
    for (size_t c = 0; c < x_h.num_col(); ++c) {
        for (size_t r = 0; r < x_h.num_row(); ++r) {
            auto xp = Matrix<float>(x_h);
            auto xm = Matrix<float>(x_h);
            xp.elem(r, c) += eps;
            xm.elem(r, c) -= eps;
            float lp = forward_loss_host<BackendT>(layer, xp, upstream_h);
            float lm = forward_loss_host<BackendT>(layer, xm, upstream_h);
            num.elem(r, c) = (lp - lm) / (2.0f * eps);
        }
    }

    unfreeze(layers);

    float max_abs = 0, rel_l2 = 0;
    compare_matrix(dx_h, num, name, max_abs, rel_l2);
    if (max_abs > max_abs_tol || rel_l2 > rel_l2_tol) {
        std::cout << "[FAIL] " << name << "\n";
        return 1;
    }
    std::cout << "[PASS] " << name << "\n";
    return 0;
}

int compute() {
    global_rand_gen.seed(42);

#if defined(CUDA)
    GPUSampler sampler(42);
    using BackendT = CUDAfloat;
#elif defined(ROCM_HIP)
    using BackendT = ROCMfloat;
#elif defined(APPLE_SILICON)
    using BackendT = MPSfloat;
#else
    using BackendT = float;
#endif

    int ret = 0;

    // ── Test 1: forward shape ──────────────────────────────────────────
    {
        const int d_model = 4, d_k = 3, d_ff = 6, seq = 3, batch = 2;
        TransformerLayer<BackendT> tf(d_model, d_k, d_ff, seq, batch);

        auto x = Matrix<BackendT>(Matrix<float>::randn(d_model, seq * batch));
        tf.eval(x);

        auto out = as_host(tf.value());
        if (out.num_row() != (size_t)d_model ||
            out.num_col() != (size_t)(seq * batch)) {
            std::cout << "[FAIL] forward shape: got " << out.num_row()
                      << "x" << out.num_col() << "\n";
            ret = 1;
        } else {
            std::cout << "[PASS] forward shape: " << out.num_row()
                      << "x" << out.num_col() << "\n";
        }
    }

    // ── Test 2: finite-difference gradient check (small net) ──────────
    {
        const int d_model = 4, d_k = 3, d_ff = 6, seq = 3, batch = 2;
        TransformerLayer<BackendT> tf(d_model, d_k, d_ff, seq, batch);

        auto x_h = Matrix<float>::randn(d_model, seq * batch) * 0.5f;
        auto g_h = Matrix<float>::randn(d_model, seq * batch) * 0.5f;

        ret += check_input_gradient_fd<BackendT>(
            "TransformerLayer dx", tf, x_h, g_h,
            1e-3f, 5e-2f, 5e-2f);
    }

    // ── Test 3: repeated forward gives same result (determinism) ──────
    {
        const int d_model = 4, d_k = 3, d_ff = 6, seq = 3, batch = 2;
        TransformerLayer<BackendT> tf(d_model, d_k, d_ff, seq, batch);

        auto x = Matrix<BackendT>(Matrix<float>::randn(d_model, seq * batch));
        tf.eval(x);
        auto y1 = as_host(tf.value());
        tf.eval(x);
        auto y2 = as_host(tf.value());

        float max_abs = 0, rel_l2 = 0;
        compare_matrix(y1, y2, "determinism", max_abs, rel_l2);
        if (max_abs > 1e-6f) {
            std::cout << "[FAIL] determinism\n";
            ret = 1;
        } else {
            std::cout << "[PASS] determinism\n";
        }
    }

    return ret;
}
