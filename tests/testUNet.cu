/**
 * @file testUNet.cu
 * @brief Forward and backward correctness checks for a tiny U-Net built from ConvLayer/convtransLayer.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using namespace Juzhen;

#if !defined(CUDA) || !defined(CUDNN_AVAILABLE)
int compute() {
    std::cout << "testUNet requires CUDA + cuDNN. Skipping.\n";
    return 0;
}
#else

static float scalar_loss(const Matrix<CUDAfloat>& y, const Matrix<CUDAfloat>& g) {
    return item(sum(sum(hadmd(y, g), 0), 1));
}

static void compare_vectors(const std::string& tag,
                            const std::vector<float>& lhs,
                            const std::vector<float>& rhs,
                            float& max_abs_err,
                            float& rel_l2_err) {
    max_abs_err = 0.0f;
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        float diff = lhs[i] - rhs[i];
        max_abs_err = std::max(max_abs_err, std::fabs(diff));
        num += (double)diff * (double)diff;
        den += (double)lhs[i] * (double)lhs[i];
    }
    rel_l2_err = (float)std::sqrt(num / std::max(den, 1e-12));
    std::cout << tag << " max_abs_err=" << max_abs_err
              << " rel_l2_err=" << rel_l2_err << std::endl;
}

class TinyUNet {
public:
    static constexpr int N = 2;
    static constexpr int H = 8;
    static constexpr int W = 8;
    static constexpr int D = H * W;
    static constexpr int C0 = 2;
    static constexpr int C1 = 4;
    static constexpr int C2 = 6;
    static constexpr int H2 = 4;
    static constexpr int W2 = 4;

    ConvLayer enc1;
    ConvLayer enc2;
    convtransLayer up1;
    ConvLayer dec1;
    ConvLayer head;

    Matrix<CUDAfloat> e1;
    Matrix<CUDAfloat> e2;
    Matrix<CUDAfloat> c_cat;

    TinyUNet()
        : enc1(N, C0, H, W, C1, 3, 3, 1, 1, true),
          enc2(N, C1, H, W, C2, 3, 3, 1, 2, true),
          up1 (N, C2, H2, W2, C1, 4, 4, 1, 2, true),
          dec1(N, C1 + C1, H, W, C1, 3, 3, 1, 1, true),
          head(N, C1, H, W, 1, 3, 3, 1, 1, false),
          e1("e1", C1 * D, N),
          e2("e2", C2 * H2 * W2, N),
          c_cat("cc", (C1 + C1) * D, N) {}

    std::list<Layer<CUDAfloat>*> layers() {
        return {&head, &dec1, &up1, &enc2, &enc1};
    }

    void zero_all_params() {
        for (auto* layer : layers()) {
            layer->W() = Matrix<CUDAfloat>::zeros(layer->W().num_row(), layer->W().num_col());
            layer->b() = Matrix<CUDAfloat>::zeros(layer->b().num_row(), layer->b().num_col());
        }
    }

    const Matrix<CUDAfloat>& fwd(const Matrix<CUDAfloat>& inp) {
        enc1.eval(inp);
        e1 = enc1.value();
        enc2.eval(e1);
        e2 = enc2.value();
        up1.eval(e2);
        c_cat = vstack(std::vector<MatrixView<CUDAfloat>>{up1.value(), e1});
        dec1.eval(c_cat);
        head.eval(dec1.value());
        return head.value();
    }

    Matrix<CUDAfloat> backward_input(const Matrix<CUDAfloat>& inp, Matrix<CUDAfloat>&& g) {
        auto g2 = head.backward(dec1.value(), std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = (size_t)C1 * (size_t)D;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        return enc1.backward(inp, std::move(g_inp));
    }
};

template <class NetT>
static float forward_loss(NetT& net,
                          const Matrix<CUDAfloat>& x,
                          const Matrix<CUDAfloat>& upstream) {
    return scalar_loss(net.fwd(x), upstream);
}

static int check_zero_output() {
    TinyUNet net;
    net.zero_all_params();
    auto x = Matrix<CUDAfloat>::rand(TinyUNet::C0 * TinyUNet::D, TinyUNet::N);
    auto y = net.fwd(x).to_host();

    float max_abs = 0.0f;
    for (size_t c = 0; c < y.num_col(); ++c) {
        for (size_t r = 0; r < y.num_row(); ++r) {
            max_abs = std::max(max_abs, std::fabs(y.elem(r, c)));
        }
    }
    std::cout << "TinyUNet zero-output max_abs_err=" << max_abs << std::endl;
    if (max_abs > 1e-7f) {
        std::cout << "[FAIL] TinyUNet zero-output" << std::endl;
        return 1;
    }
    std::cout << "[PASS] TinyUNet zero-output" << std::endl;
    return 0;
}

static int check_head_bias_output() {
    TinyUNet net;
    net.zero_all_params();
    Matrix<float> head_b("hb", 1, 1);
    head_b.elem(0, 0) = -0.375f;
    net.head.b() = Matrix<CUDAfloat>(head_b);

    auto x = Matrix<CUDAfloat>::rand(TinyUNet::C0 * TinyUNet::D, TinyUNet::N);
    auto y = net.fwd(x).to_host();

    float max_abs = 0.0f;
    for (size_t c = 0; c < y.num_col(); ++c) {
        for (size_t r = 0; r < y.num_row(); ++r) {
            max_abs = std::max(max_abs, std::fabs(y.elem(r, c) + 0.375f));
        }
    }
    std::cout << "TinyUNet head-bias-only max_abs_err=" << max_abs << std::endl;
    if (max_abs > 1e-7f) {
        std::cout << "[FAIL] TinyUNet head-bias-only" << std::endl;
        return 1;
    }
    std::cout << "[PASS] TinyUNet head-bias-only" << std::endl;
    return 0;
}

static int check_input_gradient() {
    TinyUNet net;
    auto layers = net.layers();
    freeze(layers);

    auto x = Matrix<CUDAfloat>::rand(TinyUNet::C0 * TinyUNet::D, TinyUNet::N) - 0.5f;
    auto g = Matrix<CUDAfloat>::rand(TinyUNet::D, TinyUNet::N) - 0.5f;

    net.fwd(x);
    auto dx = net.backward_input(x, Matrix<CUDAfloat>(g)).to_host();
    auto x_h = x.to_host();

    std::vector<float> analytic((size_t)dx.num_row() * (size_t)dx.num_col());
    std::vector<float> numeric((size_t)x_h.num_row() * (size_t)x_h.num_col(), 0.0f);
    for (size_t i = 0; i < analytic.size(); ++i) {
        analytic[i] = dx.data()[i];
    }

    const float eps = 1e-3f;
    for (size_t idx = 0; idx < numeric.size(); ++idx) {
        auto xp = Matrix<float>(x_h);
        auto xm = Matrix<float>(x_h);
        size_t r = idx % x_h.num_row();
        size_t c = idx / x_h.num_row();
        xp.elem(r, c) += eps;
        xm.elem(r, c) -= eps;
        float lp = forward_loss(net, Matrix<CUDAfloat>(xp), g);
        float lm = forward_loss(net, Matrix<CUDAfloat>(xm), g);
        numeric[idx] = (lp - lm) / (2.0f * eps);
    }

    float max_abs_err = 0.0f;
    float rel_l2_err = 0.0f;
    compare_vectors("TinyUNet dL/dx", analytic, numeric, max_abs_err, rel_l2_err);

    unfreeze(layers);

    if (max_abs_err > 6e-2f && rel_l2_err > 4e-2f) {
        std::cout << "[FAIL] TinyUNet dL/dx" << std::endl;
        return 1;
    }
    std::cout << "[PASS] TinyUNet dL/dx" << std::endl;
    return 0;
}

static int check_directional_derivative() {
    TinyUNet net;
    auto layers = net.layers();
    freeze(layers);

    auto x = Matrix<CUDAfloat>::rand(TinyUNet::C0 * TinyUNet::D, TinyUNet::N) - 0.5f;
    auto g = Matrix<CUDAfloat>::rand(TinyUNet::D, TinyUNet::N) - 0.5f;
    auto v = Matrix<CUDAfloat>::rand(TinyUNet::C0 * TinyUNet::D, TinyUNet::N) - 0.5f;

    net.fwd(x);
    auto dx = net.backward_input(x, Matrix<CUDAfloat>(g));
    float lhs = scalar_loss(dx, v);

    const float eps = 1e-3f;
    float lp = forward_loss(net, x + v * eps, g);
    float lm = forward_loss(net, x - v * eps, g);
    float rhs = (lp - lm) / (2.0f * eps);

    unfreeze(layers);

    float abs_err = std::fabs(lhs - rhs);
    float rel_err = abs_err / std::max(std::fabs(lhs), 1e-6f);
    std::cout << "TinyUNet directional-derivative abs_err=" << abs_err
              << " rel_err=" << rel_err << std::endl;
    if (abs_err > 8e-3f && rel_err > 5e-3f) {
        std::cout << "[FAIL] TinyUNet directional-derivative" << std::endl;
        return 1;
    }
    std::cout << "[PASS] TinyUNet directional-derivative" << std::endl;
    return 0;
}

int compute() {
    GPUSampler sampler(7);

    int ret = 0;
    ret += check_zero_output();
    ret += check_head_bias_output();
    ret += check_input_gradient();
    ret += check_directional_derivative();

    if (ret == 0) {
        LOG_INFO("testUNet passed!");
    } else {
        LOG_ERROR("testUNet failed!");
    }
    return ret;
}

#endif
