/**
 * @file testDiffusionScore.cu
 * @brief End-to-end correctness test for the diffusion score objective with a tiny U-Net.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <vector>

using namespace Juzhen;

#if !defined(CUDA) || !defined(CUDNN_AVAILABLE)
int compute() {
    std::cout << "testDiffusionScore requires CUDA + cuDNN. Skipping.\n";
    return 0;
}
#else

inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> vs(std::vector<MatrixView<CUDAfloat>> ms) { return vstack(ms); }

struct VPSchedule {
    std::vector<float> beta, alpha, abar, sqrt_abar, sqrt_one_minus_abar;

    explicit VPSchedule(int T) {
        beta.resize(T + 1);
        alpha.resize(T + 1);
        abar.resize(T + 1);
        sqrt_abar.resize(T + 1);
        sqrt_one_minus_abar.resize(T + 1);
        abar[0] = 1.f;
        for (int t = 1; t <= T; ++t) {
            float s = (float)(t - 1) / (float)(T - 1);
            beta[t] = 1e-4f + s * (0.02f - 1e-4f);
            alpha[t] = 1.f - beta[t];
            abar[t] = abar[t - 1] * alpha[t];
            sqrt_abar[t] = std::sqrt(abar[t]);
            sqrt_one_minus_abar[t] = std::sqrt(std::max(1e-12f, 1.f - abar[t]));
        }
    }
};

class TinyScoreUNet {
public:
    static constexpr int H = 16;
    static constexpr int W = 16;
    static constexpr int d = H * W;

    ConvLayer enc1;
    ConvLayer enc2;
    convtransLayer up1;
    ConvLayer dec1;
    ConvLayer head;

    Matrix<CUDAfloat> e1;
    Matrix<CUDAfloat> e2;
    Matrix<CUDAfloat> c_cat;

    explicit TinyScoreUNet(int bs)
        : enc1(bs,   2, 16, 16,  8, 3, 3, 1, 1, true),
          enc2(bs,   8, 16, 16, 16, 3, 3, 1, 2, true),
          up1 (bs,  16,  8,  8,  8, 4, 4, 1, 2, true),
          dec1(bs,  16, 16, 16,  8, 3, 3, 1, 1, true),
          head(bs,   8, 16, 16,  1, 3, 3, 1, 1, false),
          e1("e1", 8 * d, bs),
          e2("e2", 16 * 8 * 8, bs),
          c_cat("cc", 16 * d, bs) {}

    std::list<Layer<CUDAfloat>*> layers() {
        return {&head, &dec1, &up1, &enc2, &enc1};
    }

    void set_lr(float alpha) {
        for (auto* l : layers()) {
            l->adamWstate().alpha = alpha;
            l->adambstate().alpha = alpha;
        }
    }

    const Matrix<CUDAfloat>& fwd(const Matrix<CUDAfloat>& inp) {
        enc1.eval(inp);
        e1 = enc1.value();
        enc2.eval(e1);
        e2 = enc2.value();
        up1.eval(e2);
        c_cat = vs({up1.value(), e1});
        dec1.eval(c_cat);
        head.eval(dec1.value());
        return head.value();
    }

    void bwd(const Matrix<CUDAfloat>& inp, Matrix<CUDAfloat>&& g) {
        auto g2 = head.backward(dec1.value(), std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = 8ULL * d;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        enc1.backward(inp, std::move(g_inp));
    }
};

static float max_abs(const Matrix<float>& m) {
    float ma = 0.f;
    for (size_t c = 0; c < m.num_col(); ++c) {
        for (size_t r = 0; r < m.num_row(); ++r) {
            ma = std::max(ma, std::fabs(m.elem(r, c)));
        }
    }
    return ma;
}

int compute() {
    GPUSampler sampler(2026);

    constexpr int batchsize = 4;
    constexpr int T = 100;
    constexpr int n_iters = 1200;
    constexpr float lr = 3e-4f;

    VPSchedule sched(T);
    TinyScoreUNet net(batchsize);
    net.set_lr(lr);

    auto x0 = Matrix<CUDAfloat>::rand(TinyScoreUNet::d, batchsize);
    auto eps = randn(TinyScoreUNet::d, batchsize);

    Matrix<float> t_row("t", 1, batchsize);
    Matrix<float> c1_row("c1", 1, batchsize);
    Matrix<float> c2_row("c2", 1, batchsize);
    Matrix<float> invsig_row("is", 1, batchsize);

    const int fixed_t[batchsize] = {10, 35, 70, 95};
    for (int b = 0; b < batchsize; ++b) {
        int t_idx = fixed_t[b];
        t_row.elem(0, b) = (float)t_idx / (float)T;
        c1_row.elem(0, b) = sched.sqrt_abar[t_idx];
        c2_row.elem(0, b) = sched.sqrt_one_minus_abar[t_idx];
        invsig_row.elem(0, b) = 1.f / std::max(1e-6f, sched.sqrt_one_minus_abar[t_idx]);
    }

    auto t_cu = Matrix<CUDAfloat>(t_row);
    auto c1 = Matrix<CUDAfloat>(c1_row);
    auto c2 = Matrix<CUDAfloat>(c2_row);
    auto invsig = Matrix<CUDAfloat>(invsig_row);

    auto xt = hadmd(x0, ones(TinyScoreUNet::d, 1) * c1) + hadmd(eps, ones(TinyScoreUNet::d, 1) * c2);
    auto t_map = ones(TinyScoreUNet::d, 1) * t_cu;
    auto inp = vs({xt, t_map});
    auto score_target = -hadmd(eps, ones(TinyScoreUNet::d, 1) * invsig);

    // Consistency check: score target should reconstruct epsilon via eps = -score * sigma_t.
    auto eps_recon = -hadmd(score_target, ones(TinyScoreUNet::d, 1) * c2);
    auto recon_err = max_abs((eps_recon - eps).to_host());
    std::cout << "reconstruction_max_abs_err=" << recon_err << std::endl;
    if (recon_err > 5e-4f) {
        LOG_ERROR("Score target reconstruction check failed: {}", recon_err);
        return 1;
    }

    float initial_loss = 0.f;
    float best_loss = 0.f;
    float final_loss = 0.f;
    for (int it = 0; it < n_iters; ++it) {
        const auto& pred = net.fwd(inp);
        auto diff = pred - score_target;
        float loss = item(sum(sum(square(diff), 0), 1) / (float)batchsize);
        if (it == 0) {
            initial_loss = loss;
            best_loss = loss;
        }
        best_loss = std::min(best_loss, loss);
        final_loss = loss;
        if (it % 200 == 0 || it + 1 == n_iters) {
            std::cout << "iter=" << it << " loss=" << loss << std::endl;
        }
        net.bwd(inp, 2.f * diff / (float)batchsize);
    }

    std::cout << "initial_loss=" << initial_loss
              << " best_loss=" << best_loss
              << " final_loss=" << final_loss << std::endl;

    const bool improved = best_loss < initial_loss * 0.50f;
    const bool low_abs = best_loss < 5000.f;
    if (!improved || !low_abs) {
        LOG_ERROR("testDiffusionScore failed: initial_loss={}, final_loss={}", initial_loss, final_loss);
        return 1;
    }

    LOG_INFO("testDiffusionScore passed: initial_loss={}, final_loss={}", initial_loss, final_loss);
    return 0;
}

#endif