/**
 * @file testTransformerTorchDump.cu
 * @brief Dump a TransformerLayer's weights, input, forward output, and input
 *        gradient so tests/testTransformerTorch.py can cross-check them
 *        against a PyTorch implementation of the same block.
 *
 * testTransformerRef.cu already compares against an independent C++ reference,
 * but both live in this repo, so a shared misunderstanding of the intended
 * architecture would pass. This pair closes that gap: PyTorch is an external
 * oracle. Run this first, then:
 *
 *     python3 tests/testTransformerTorch.py
 *
 * File format (little-endian), written to res/transformer_torch_dump.bin:
 *   char[8]   magic "JZTFDMP2"
 *   int32[7]  d_model, d_k, d_ff, seq_len, batch, num_heads, causal
 *   17 matrices, each: int32 rows, int32 cols, float32[rows*cols] col-major:
 *     Wq Wk Wv Wo bo W1 b1 W2 b2 ln1_g ln1_b ln2_g ln2_b x g out dx
 *   char[8] "JZTRAIN1", int32 step_count; per step, 13 parameter records:
 *     float32[4] lr/beta1/beta2/eps, then parameter, Adam m, Adam v matrices.
 * JUZHEN_TRANSFORMER_CASE selects default/single/bidirectional/long. Each
 * non-default case uses a separate output filename for parallel CTest runs.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <string>

using namespace Juzhen;

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

static void write_mat(FILE* fp, const Matrix<float>& m) {
    const int32_t r = (int32_t)m.num_row(), c = (int32_t)m.num_col();
    fwrite(&r, sizeof(int32_t), 1, fp);
    fwrite(&c, sizeof(int32_t), 1, fp);
    for (int32_t j = 0; j < c; ++j)
        for (int32_t i = 0; i < r; ++i) {
            const float v = m.elem(i, j);
            fwrite(&v, sizeof(float), 1, fp);
        }
}

int compute() {
    global_rand_gen.seed(11);

#if defined(CUDA)
    GPUSampler sampler(11);
    using BackendT = CUDAfloat;
#elif defined(ROCM_HIP)
    using BackendT = ROCMfloat;
#elif defined(APPLE_SILICON)
    using BackendT = MPSfloat;
#else
    using BackendT = float;
#endif

    int d_model = 16, d_k = 16, d_ff = 32, seq = 7, batch = 2, heads = 4;
    const char* selected = std::getenv("JUZHEN_TRANSFORMER_CASE");
    const std::string test_case = selected ? selected : "default";
    bool causal = true;
    if (test_case == "single") { seq = 1; batch = 3; }
    else if (test_case == "bidirectional") {
        d_model = 12; d_k = 15; d_ff = 25; seq = 17; batch = 3; heads = 3;
        causal = false;
    } else if (test_case == "long") { seq = 257; }
    else if (test_case != "default") return 1;
    const int N = seq * batch;

    TransformerLayer<BackendT> tf(d_model, d_k, d_ff, seq, batch, heads, causal);

    auto x_h = Matrix<float>::randn(d_model, N) * 0.5f;
    auto g_h = Matrix<float>::randn(d_model, N) * 0.5f;

    std::list<Layer<BackendT>*> layers = {&tf};
    freeze(layers);   // keep weights fixed: dump must match what ran
    tf.eval(Matrix<BackendT>(x_h));
    auto out_h = as_host(tf.value());
    auto dx_h  = as_host(tf.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(g_h)));
    unfreeze(layers);

    const std::string path =
        std::string(PROJECT_DIR) + "/res/transformer_torch_dump" +
        (test_case == "default" ? "" : "_" + test_case) + ".bin";
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cout << "[FAIL] cannot write " << path << "\n";
        return 1;
    }

    const int32_t cfg[7] = {d_model, d_k, d_ff, seq, batch, heads, causal};
    fwrite("JZTFDMP2", 1, 8, fp);
    fwrite(cfg, sizeof(int32_t), 7, fp);

    write_mat(fp, as_host(tf.get_Wq()));
    write_mat(fp, as_host(tf.get_Wk()));
    write_mat(fp, as_host(tf.get_Wv()));
    write_mat(fp, as_host(tf.get_Wo()));
    write_mat(fp, as_host(tf.get_bo()));
    write_mat(fp, as_host(tf.get_W1()));
    write_mat(fp, as_host(tf.get_b1()));
    write_mat(fp, as_host(tf.get_W2()));
    write_mat(fp, as_host(tf.get_b2()));
    write_mat(fp, as_host(tf.get_ln1_gamma()));
    write_mat(fp, as_host(tf.get_ln1_beta()));
    write_mat(fp, as_host(tf.get_ln2_gamma()));
    write_mat(fp, as_host(tf.get_ln2_beta()));
    write_mat(fp, x_h);
    write_mat(fp, g_h);
    write_mat(fp, out_h);
    write_mat(fp, dx_h);
    // Versioned training extension: all parameters and Adam states
    // after each step, allowing an independent gradient and update oracle.
    fwrite("JZTRAIN1", 1, 8, fp);
    const int32_t steps = 3;
    fwrite(&steps, sizeof(steps), 1, fp);
    tf.set_lr(1e-4f);
    for (int step = 0; step < steps; ++step) {
        tf.eval(Matrix<BackendT>(x_h));
        tf.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(g_h));
        const auto params = tf.checkpoint_parameters();
        const auto opts = tf.checkpoint_optimizers();
        for (size_t i = 0; i < params.size(); ++i) {
            auto& state = *opts[i].second;
            const float hyper[] = {state.alpha, state.beta1, state.beta2, state.eps};
            fwrite(hyper, sizeof(float), 4, fp);
            write_mat(fp, as_host(*params[i].second));
            write_mat(fp, as_host(state.m));
            write_mat(fp, as_host(state.v));
        }
    }
    fclose(fp);

    std::cout << "Wrote " << path << "\n"
              << "Now run: python3 tests/testTransformerTorch.py\n";
    return 0;
}
