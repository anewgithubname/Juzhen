/**
 * @file testTransformerParity.cu
 * @brief CPU–CUDA parity test for TransformerLayer.
 *
 * Builds two TransformerLayers (one CPU, one CUDA) with identical weights,
 * runs eval() and backward() on the same inputs, and compares results.
 * Only compiled when CUDA is available; otherwise a trivial pass.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cmath>
#include <iostream>
#include <list>

using namespace Juzhen;

#if defined(CUDA)

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

// Copy weights from CPU layer to GPU layer.
static void sync_weights(const TransformerLayer<float>& cpu,
                         TransformerLayer<CUDAfloat>& gpu) {
    gpu.set_Wq(Matrix<CUDAfloat>(cpu.get_Wq()));
    gpu.set_Wk(Matrix<CUDAfloat>(cpu.get_Wk()));
    gpu.set_Wv(Matrix<CUDAfloat>(cpu.get_Wv()));
    gpu.set_Wo(Matrix<CUDAfloat>(cpu.get_Wo()));
    gpu.set_bo(Matrix<CUDAfloat>(cpu.get_bo()));
    gpu.set_W1(Matrix<CUDAfloat>(cpu.get_W1()));
    gpu.set_b1(Matrix<CUDAfloat>(cpu.get_b1()));
    gpu.set_W2(Matrix<CUDAfloat>(cpu.get_W2()));
    gpu.set_b2(Matrix<CUDAfloat>(cpu.get_b2()));
}

int compute() {
    global_rand_gen.seed(123);
    GPUSampler sampler(123);

    const int d_model = 8, d_k = 6, d_ff = 12, seq = 4, batch = 2;
    const int N = seq * batch;

    TransformerLayer<float>      cpu_tf(d_model, d_k, d_ff, seq, batch);
    TransformerLayer<CUDAfloat>  gpu_tf(d_model, d_k, d_ff, seq, batch);
    sync_weights(cpu_tf, gpu_tf);

    auto x_h = Matrix<float>::randn(d_model, N) * 0.5f;
    auto g_h = Matrix<float>::randn(d_model, N) * 0.5f;

    // Freeze to avoid weight updates changing state between forward/backward.
    std::list<Layer<float>*>       cpu_layers = {&cpu_tf};
    std::list<Layer<CUDAfloat>*>   gpu_layers = {&gpu_tf};
    freeze(cpu_layers);
    freeze(gpu_layers);

    int ret = 0;
    float max_abs, rel_l2;

    // ── Forward parity ──────────────────────────────────────────────────
    cpu_tf.eval(x_h);
    gpu_tf.eval(Matrix<CUDAfloat>(x_h));

    auto cpu_out = cpu_tf.value();
    auto gpu_out = gpu_tf.value().to_host();

    compare_matrix(gpu_out, cpu_out, "forward parity", max_abs, rel_l2);
    if (max_abs > 1e-4f || rel_l2 > 1e-4f) {
        std::cout << "[FAIL] forward parity\n";
        ret = 1;
    } else {
        std::cout << "[PASS] forward parity\n";
    }

    // ── Backward parity ─────────────────────────────────────────────────
    cpu_tf.eval(x_h);
    gpu_tf.eval(Matrix<CUDAfloat>(x_h));

    auto cpu_dx = cpu_tf.backward(x_h, Matrix<float>(g_h));
    auto gpu_dx = gpu_tf.backward(Matrix<CUDAfloat>(x_h),
                                  Matrix<CUDAfloat>(g_h)).to_host();

    compare_matrix(gpu_dx, cpu_dx, "backward parity (dx)", max_abs, rel_l2);
    if (max_abs > 1e-3f || rel_l2 > 1e-3f) {
        std::cout << "[FAIL] backward parity (dx)\n";
        ret = 1;
    } else {
        std::cout << "[PASS] backward parity (dx)\n";
    }

    unfreeze(cpu_layers);
    unfreeze(gpu_layers);

    return ret;
}

#else // Not CUDA

int compute() {
    std::cout << "[SKIP] TransformerParity: CUDA not available\n";
    return 0;
}

#endif
