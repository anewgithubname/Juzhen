/**
 * @file testRectifiedPathParity.cu
 * @brief CPU vs MPS parity check for rectified-flow data path.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>

using namespace Juzhen;

#if !defined(APPLE_SILICON)
int compute() {
    std::cout << "testRectifiedPathParity requires APPLE_SILICON. Skipping.\n";
    return 0;
}
#else

static void compare_matrix(const Matrix<float>& a,
                           const Matrix<float>& b,
                           const std::string& tag,
                           float& max_abs,
                           float& rel_l2) {
    max_abs = 0.0f;
    double num = 0.0;
    double den = 0.0;

    for (size_t c = 0; c < a.num_col(); ++c) {
        for (size_t r = 0; r < a.num_row(); ++r) {
            const float da = a.elem(r, c);
            const float db = b.elem(r, c);
            const float d = da - db;
            max_abs = std::max(max_abs, std::fabs(d));
            num += (double)d * (double)d;
            den += (double)da * (double)da;
        }
    }

    rel_l2 = (float)std::sqrt(num / std::max(den, 1e-12));
    std::cout << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2 << std::endl;
}

int compute() {
    global_rand_gen.seed(2026);

    constexpr int batch = 32;
    constexpr int d = 2;

    auto x0_h = Matrix<float>::randn(d, batch);
    auto x1_h = Matrix<float>::randn(d, batch);
    auto t_h  = Matrix<float>::rand(1, batch);

    auto x0_m = Matrix<MPSfloat>(x0_h);
    auto x1_m = Matrix<MPSfloat>(x1_h);
    auto t_m  = Matrix<MPSfloat>(t_h);

    auto one_h = Matrix<float>::ones(d, 1);
    auto one_m = Matrix<MPSfloat>::ones(d, 1);

    auto Xt_h = hadmd(x0_h, one_h * (1 - t_h)) + hadmd(x1_h, one_h * t_h);
    auto Xt_m = hadmd(x0_m, one_m * (1 - t_m)) + hadmd(x1_m, one_m * t_m);

    auto target_h = x1_h - x0_h;
    auto target_m = x1_m - x0_m;

    auto inp_h = vstack<float>({Xt_h, t_h});
    auto inp_m = vstack({Xt_m, t_m});

    std::cout << "Xt_h shape=" << Xt_h.num_row() << "x" << Xt_h.num_col()
              << " trans=" << Xt_h.get_transpose() << std::endl;
    std::cout << "Xt_m shape=" << Xt_m.num_row() << "x" << Xt_m.num_col()
              << " trans=" << Xt_m.get_transpose() << std::endl;
    std::cout << "t_h  shape=" << t_h.num_row()  << "x" << t_h.num_col()
              << " trans=" << t_h.get_transpose() << std::endl;
    std::cout << "t_m  shape=" << t_m.num_row()  << "x" << t_m.num_col()
              << " trans=" << t_m.get_transpose() << std::endl;
    std::cout << "inp_h shape=" << inp_h.num_row() << "x" << inp_h.num_col()
              << " trans=" << inp_h.get_transpose() << std::endl;
    std::cout << "inp_m shape=" << inp_m.num_row() << "x" << inp_m.num_col()
              << " trans=" << inp_m.get_transpose() << std::endl;

    float max_abs = 0.0f, rel_l2 = 0.0f;

    compare_matrix(Xt_h, Xt_m.to_host(), "Xt", max_abs, rel_l2);
    if (max_abs > 2e-5f || rel_l2 > 2e-5f) {
        std::cout << "[FAIL] Xt mismatch\n";
        return 1;
    }

    compare_matrix(target_h, target_m.to_host(), "target", max_abs, rel_l2);
    if (max_abs > 2e-6f || rel_l2 > 2e-6f) {
        std::cout << "[FAIL] target mismatch\n";
        return 1;
    }

    compare_matrix(inp_h, inp_m.to_host(), "inp=vstack(Xt,t)", max_abs, rel_l2);
    if (max_abs > 2e-5f || rel_l2 > 2e-5f) {
        std::cout << "[FAIL] inp mismatch\n";
        return 1;
    }

    ReluLayer<float> L0_h(64, d + 1, batch), L1_h(64, 64, batch), L2_h(64, 64, batch);
    LinearLayer<float> L3_h(d, 64, batch);

    ReluLayer<MPSfloat> L0_m(64, d + 1, batch), L1_m(64, 64, batch), L2_m(64, 64, batch);
    LinearLayer<MPSfloat> L3_m(d, 64, batch);

    // Ensure identical parameters between CPU and MPS paths.
    L0_m.W() = Matrix<MPSfloat>(L0_h.W());
    L0_m.b() = Matrix<MPSfloat>(L0_h.b());
    L1_m.W() = Matrix<MPSfloat>(L1_h.W());
    L1_m.b() = Matrix<MPSfloat>(L1_h.b());
    L2_m.W() = Matrix<MPSfloat>(L2_h.W());
    L2_m.b() = Matrix<MPSfloat>(L2_h.b());
    L3_m.W() = Matrix<MPSfloat>(L3_h.W());
    L3_m.b() = Matrix<MPSfloat>(L3_h.b());

    std::list<Layer<float>*> net_h = {&L3_h, &L2_h, &L1_h, &L0_h};
    std::list<Layer<MPSfloat>*> net_m = {&L3_m, &L2_m, &L1_m, &L0_m};

    auto out_h = Matrix<float>(forward(net_h, inp_h));
    auto out_m = forward(net_m, inp_m).to_host();

    compare_matrix(out_h, out_m, "rectified forward output", max_abs, rel_l2);
    if (max_abs > 3e-4f || rel_l2 > 3e-4f) {
        std::cout << "[FAIL] model output mismatch\n";
        return 1;
    }

    LossLayer<float> loss_h(batch, target_h);
    LossLayer<MPSfloat> loss_m(batch, Matrix<MPSfloat>(target_h));

    std::list<Layer<float>*> eval_h = {&loss_h, &L3_h, &L2_h, &L1_h, &L0_h};
    std::list<Layer<MPSfloat>*> eval_m = {&loss_m, &L3_m, &L2_m, &L1_m, &L0_m};

    const float scalar_h = item(forward(eval_h, inp_h));
    const float scalar_m = item(forward(eval_m, inp_m));
    const float loss_abs = std::fabs(scalar_h - scalar_m);
    const float loss_rel = loss_abs / std::max(std::fabs(scalar_h), 1e-12f);

    std::cout << "loss scalar abs=" << loss_abs << " rel=" << loss_rel << std::endl;
    if (loss_abs > 3e-4f && loss_rel > 3e-4f) {
        std::cout << "[FAIL] loss scalar mismatch\n";
        return 1;
    }

    std::cout << "[PASS] testRectifiedPathParity\n";
    return 0;
}
#endif
