/**
 * @file testMPSParity.cu
 * @brief CPU vs MPS parity check for one MLP training step.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <list>
#include <vector>

using namespace Juzhen;

#if !defined(APPLE_SILICON)
int compute() {
    std::cout << "testMPSParity requires APPLE_SILICON. Skipping.\n";
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

struct RandStats {
    float min_v = 0.0f;
    float max_v = 0.0f;
    double mean = 0.0;
    double var = 0.0;
    std::array<size_t, 10> hist{};
};

static RandStats summarize_rand(const Matrix<float>& m) {
    RandStats s;
    s.min_v = m.elem(0, 0);
    s.max_v = m.elem(0, 0);

    double sum = 0.0;
    double sum_sq = 0.0;
    const size_t n = m.num_row() * m.num_col();

    for (size_t c = 0; c < m.num_col(); ++c) {
        for (size_t r = 0; r < m.num_row(); ++r) {
            const float v = m.elem(r, c);
            s.min_v = std::min(s.min_v, v);
            s.max_v = std::max(s.max_v, v);
            sum += v;
            sum_sq += (double)v * (double)v;

            int bin = (int)(v * 10.0f);
            if (bin < 0) bin = 0;
            if (bin > 9) bin = 9;
            s.hist[(size_t)bin]++;
        }
    }

    s.mean = sum / (double)n;
    s.var = sum_sq / (double)n - s.mean * s.mean;
    return s;
}

static int check_mps_rand() {
    constexpr size_t rows = 1024;
    constexpr size_t cols = 64;
    constexpr size_t n = rows * cols;

    global_rand_gen.seed(2026);
    auto cpu = Matrix<float>::rand(rows, cols);
    auto mps = Matrix<MPSfloat>::rand(rows, cols).to_host();

    const RandStats cpu_stats = summarize_rand(cpu);
    const RandStats mps_stats = summarize_rand(mps);

    std::cout << "cpu rand stats: min=" << cpu_stats.min_v
              << " max=" << cpu_stats.max_v
              << " mean=" << cpu_stats.mean
              << " var=" << cpu_stats.var << std::endl;

    std::cout << "mps rand stats: min=" << mps_stats.min_v
              << " max=" << mps_stats.max_v
              << " mean=" << mps_stats.mean
              << " var=" << mps_stats.var << std::endl;

    // Uniform [0, 1): E[x]=0.5, Var[x]=1/12 ~ 0.0833.
    if (cpu_stats.min_v < 0.0f || cpu_stats.max_v >= 1.0f ||
        mps_stats.min_v < 0.0f || mps_stats.max_v >= 1.0f) {
        std::cout << "[FAIL] rand range check\n";
        return 1;
    }

    if (std::fabs(cpu_stats.mean - 0.5) > 0.03 ||
        std::fabs(cpu_stats.var - (1.0 / 12.0)) > 0.02 ||
        std::fabs(mps_stats.mean - 0.5) > 0.03 ||
        std::fabs(mps_stats.var - (1.0 / 12.0)) > 0.02) {
        std::cout << "[FAIL] rand distribution sanity check\n";
        return 1;
    }

    const double mean_diff = std::fabs(cpu_stats.mean - mps_stats.mean);
    const double var_diff = std::fabs(cpu_stats.var - mps_stats.var);
    double hist_l1 = 0.0;
    for (size_t i = 0; i < cpu_stats.hist.size(); ++i) {
        const double p = (double)cpu_stats.hist[i] / (double)n;
        const double q = (double)mps_stats.hist[i] / (double)n;
        hist_l1 += std::fabs(p - q);
    }

    std::cout << "rand parity: mean_diff=" << mean_diff
              << " var_diff=" << var_diff
              << " hist_l1=" << hist_l1 << std::endl;

    if (mean_diff > 0.015 || var_diff > 0.015 || hist_l1 > 0.10) {
        std::cout << "[FAIL] cpu vs mps rand parity check\n";
        return 1;
    }

    std::cout << "[PASS] mps rand parity\n";
    return 0;
}

static int check_mps_stack_parity() {
    float max_abs = 0.0f, rel_l2 = 0.0f;

    auto a_h = Matrix<float>::randn(5, 3);
    auto b_h = Matrix<float>::randn(5, 2);
    auto c_h = Matrix<float>::randn(5, 4);
    auto a_m = Matrix<MPSfloat>(a_h);
    auto b_m = Matrix<MPSfloat>(b_h);
    auto c_m = Matrix<MPSfloat>(c_h);

    auto h_cpu = hstack(std::vector<MatrixView<float>>{a_h, b_h, c_h});
    auto h_mps = hstack(std::vector<MatrixView<MPSfloat>>{a_m, b_m, c_m}).to_host();
    compare_matrix(h_cpu, h_mps, "hstack", max_abs, rel_l2);
    if (max_abs > 1e-6f || rel_l2 > 1e-6f) {
        std::cout << "[FAIL] hstack mismatch\n";
        return 1;
    }

    auto p_h = Matrix<float>::randn(4, 2);
    auto q_h = Matrix<float>::randn(4, 5);
    auto p_t_h = p_h.T();
    auto q_t_h = q_h.T();
    auto p_t_m = Matrix<MPSfloat>(p_h).T();
    auto q_t_m = Matrix<MPSfloat>(q_h).T();

    auto v_cpu_t = vstack(std::vector<MatrixView<float>>{p_t_h, q_t_h});
    auto v_mps_t = vstack(std::vector<MatrixView<MPSfloat>>{p_t_m, q_t_m}).to_host();
    compare_matrix(v_cpu_t, v_mps_t, "vstack(transposed)", max_abs, rel_l2);
    if (max_abs > 1e-6f || rel_l2 > 1e-6f) {
        std::cout << "[FAIL] vstack(transposed) mismatch\n";
        return 1;
    }

    auto d_h = Matrix<float>::randn(2, 4);
    auto e_h = Matrix<float>::randn(3, 4);
    auto d_m = Matrix<MPSfloat>(d_h);
    auto e_m = Matrix<MPSfloat>(e_h);

    auto v_cpu = vstack(std::vector<MatrixView<float>>{d_h, e_h});
    auto v_mps = vstack(std::vector<MatrixView<MPSfloat>>{d_m, e_m}).to_host();
    compare_matrix(v_cpu, v_mps, "vstack", max_abs, rel_l2);
    if (max_abs > 1e-6f || rel_l2 > 1e-6f) {
        std::cout << "[FAIL] vstack mismatch\n";
        return 1;
    }

    std::cout << "[PASS] mps stack parity\n";
    return 0;
}

int compute() {
    global_rand_gen.seed(1234);

    if (check_mps_rand() != 0) {
        return 1;
    }
    if (check_mps_stack_parity() != 0) {
        return 1;
    }

    constexpr int batch = 16;
    constexpr int in_dim = 4;
    constexpr int hidden = 12;
    constexpr int out_dim = 3;
    constexpr float lr = 1e-3f;

    ReluLayer<float> l0_cpu(hidden, in_dim, batch);
    LinearLayer<float> l1_cpu(out_dim, hidden, batch);

    ReluLayer<MPSfloat> l0_mps(hidden, in_dim, batch);
    LinearLayer<MPSfloat> l1_mps(out_dim, hidden, batch);

    // Start from identical parameters.
    l0_mps.W() = Matrix<MPSfloat>(l0_cpu.W());
    l0_mps.b() = Matrix<MPSfloat>(l0_cpu.b());
    l1_mps.W() = Matrix<MPSfloat>(l1_cpu.W());
    l1_mps.b() = Matrix<MPSfloat>(l1_cpu.b());

    for (auto* l : std::list<Layer<float>*>{&l1_cpu, &l0_cpu}) {
        l->adamWstate().alpha = lr;
        l->adambstate().alpha = lr;
    }
    for (auto* l : std::list<Layer<MPSfloat>*>{&l1_mps, &l0_mps}) {
        l->adamWstate().alpha = lr;
        l->adambstate().alpha = lr;
    }

    auto x_h = Matrix<float>::randn(in_dim, batch);
    auto y_h = Matrix<float>::randn(out_dim, batch);

    auto x_m = Matrix<MPSfloat>(x_h);
    auto y_m = Matrix<MPSfloat>(y_h);

    std::list<Layer<float>*> net_cpu = {&l1_cpu, &l0_cpu};
    std::list<Layer<MPSfloat>*> net_mps = {&l1_mps, &l0_mps};

    auto out_cpu = Matrix<float>(forward(net_cpu, x_h));
    auto out_mps = forward(net_mps, x_m).to_host();

    float max_abs = 0.0f, rel_l2 = 0.0f;
    compare_matrix(out_cpu, out_mps, "forward", max_abs, rel_l2);
    if (max_abs > 2e-3f || rel_l2 > 2e-3f) {
        std::cout << "[FAIL] forward mismatch\n";
        return 1;
    }

    LossLayer<float> loss_cpu(batch, y_h);
    net_cpu.push_front(&loss_cpu);
    (void)backprop(net_cpu, x_h);
    net_cpu.pop_front();

    LossLayer<MPSfloat> loss_mps(batch, y_m);
    net_mps.push_front(&loss_mps);
    (void)backprop(net_mps, x_m);
    net_mps.pop_front();

    compare_matrix(l0_cpu.W(), l0_mps.W().to_host(), "l0.W after step", max_abs, rel_l2);
    if (max_abs > 5e-3f || rel_l2 > 5e-3f) {
        std::cout << "[FAIL] l0.W mismatch\n";
        return 1;
    }

    compare_matrix(l0_cpu.b(), l0_mps.b().to_host(), "l0.b after step", max_abs, rel_l2);
    if (max_abs > 5e-3f || rel_l2 > 5e-3f) {
        std::cout << "[FAIL] l0.b mismatch\n";
        return 1;
    }

    compare_matrix(l1_cpu.W(), l1_mps.W().to_host(), "l1.W after step", max_abs, rel_l2);
    if (max_abs > 5e-3f || rel_l2 > 5e-3f) {
        std::cout << "[FAIL] l1.W mismatch\n";
        return 1;
    }

    compare_matrix(l1_cpu.b(), l1_mps.b().to_host(), "l1.b after step", max_abs, rel_l2);
    if (max_abs > 5e-3f || rel_l2 > 5e-3f) {
        std::cout << "[FAIL] l1.b mismatch\n";
        return 1;
    }

    std::cout << "[PASS] testMPSParity\n";
    return 0;
}
#endif
