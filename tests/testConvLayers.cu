/**
 * @file testConvLayers.cu
 * @brief Numerical gradient checks for ConvLayer and convtransLayer input gradients.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <vector>

using namespace Juzhen;

#if !defined(CUDA) || !defined(CUDNN_AVAILABLE)
int compute() {
    std::cout << "testConvLayers requires CUDA + cuDNN. Skipping.\n";
    return 0;
}
#else

static float scalar_loss(const Matrix<CUDAfloat>& y, const Matrix<CUDAfloat>& g) {
    return item(sum(sum(hadmd(y, g), 0), 1));
}

static void compare_vectors(const std::string& tag,
                            const std::vector<float>& ana,
                            const std::vector<float>& num,
                            float& max_abs_err,
                            float& rel_l2_err);

static inline size_t idx_chw(int c, int h, int w, int H, int W) {
    return (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
}

static inline size_t idx_w_conv(int co, int ci, int kh, int kw,
                                int Cin, int kH, int kW) {
    return ((size_t)co * (size_t)Cin * (size_t)kH * (size_t)kW)
         + ((size_t)ci * (size_t)kH * (size_t)kW)
         + ((size_t)kh * (size_t)kW)
         + (size_t)kw;
}

static inline size_t idx_w_deconv(int ci, int co, int kh, int kw,
                                  int Cout, int kH, int kW) {
    return ((size_t)ci * (size_t)Cout * (size_t)kH * (size_t)kW)
         + ((size_t)co * (size_t)kH * (size_t)kW)
         + ((size_t)kh * (size_t)kW)
         + (size_t)kw;
}

static Matrix<float> relu_ref(const Matrix<float>& x) {
    auto y = Matrix<float>(x);
    for (size_t c = 0; c < y.num_col(); ++c) {
        for (size_t r = 0; r < y.num_row(); ++r) {
            y.elem(r, c) = std::max(0.0f, y.elem(r, c));
        }
    }
    return y;
}

static Matrix<float> conv2d_ref(const Matrix<float>& x,
                                const Matrix<float>& w,
                                const Matrix<float>& b,
                                int N, int Cin, int H, int W,
                                int Cout, int kH, int kW,
                                int pad, int stride) {
    const int Hout = (H + 2 * pad - kH) / stride + 1;
    const int Wout = (W + 2 * pad - kW) / stride + 1;
    Matrix<float> y("conv_ref", Cout * Hout * Wout, N);
    y.zeros();

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float acc = b.elem(co, 0);
                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int ih = oh * stride - pad + kh;
                                const int iw = ow * stride - pad + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                const float xv = x.elem(idx_chw(ci, ih, iw, H, W), n);
                                const float wv = w.elem(idx_w_conv(co, ci, kh, kw, Cin, kH, kW), 0);
                                acc += xv * wv;
                            }
                        }
                    }
                    y.elem(idx_chw(co, oh, ow, Hout, Wout), n) = acc;
                }
            }
        }
    }
    return y;
}

static Matrix<float> convtrans2d_ref(const Matrix<float>& x,
                                     const Matrix<float>& w,
                                     const Matrix<float>& b,
                                     int N, int Cin, int H, int W,
                                     int Cout, int kH, int kW,
                                     int pad, int stride) {
    const int Hout = (H - 1) * stride - 2 * pad + kH;
    const int Wout = (W - 1) * stride - 2 * pad + kW;
    Matrix<float> y("deconv_ref", Cout * Hout * Wout, N);
    y.zeros();

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    y.elem(idx_chw(co, oh, ow, Hout, Wout), n) = b.elem(co, 0);
                }
            }
        }
        for (int ci = 0; ci < Cin; ++ci) {
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    const float xv = x.elem(idx_chw(ci, ih, iw, H, W), n);
                    for (int co = 0; co < Cout; ++co) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                const int oh = ih * stride - pad + kh;
                                const int ow = iw * stride - pad + kw;
                                if (oh < 0 || oh >= Hout || ow < 0 || ow >= Wout) continue;
                                const float wv = w.elem(idx_w_deconv(ci, co, kh, kw, Cout, kH, kW), 0);
                                y.elem(idx_chw(co, oh, ow, Hout, Wout), n) += xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    return y;
}

static int check_forward_conv() {
    constexpr int N = 2;
    constexpr int Cin = 2;
    constexpr int H = 5;
    constexpr int W = 5;
    constexpr int Cout = 3;
    constexpr int k = 3;
    constexpr int pad = 1;
    constexpr int stride = 1;

    ConvLayer conv(N, Cin, H, W, Cout, k, k, pad, stride, false);
    auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);

    // Deterministic parameters for robust CPU/GPU forward comparison.
    conv.W() = Matrix<CUDAfloat>::randn(Cout * Cin * k * k, 1) * 0.1f;
    conv.b() = Matrix<CUDAfloat>::randn(Cout, 1) * 0.1f;

    conv.eval(x);
    auto y_gpu = conv.value().to_host();

    auto y_ref = conv2d_ref(x.to_host(), conv.W().to_host(), conv.b().to_host(),
                            N, Cin, H, W, Cout, k, k, pad, stride);

    std::vector<float> a((size_t)y_gpu.num_row() * (size_t)y_gpu.num_col());
    std::vector<float> b((size_t)y_ref.num_row() * (size_t)y_ref.num_col());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = y_gpu.data()[i];
        b[i] = y_ref.data()[i];
    }

    float max_abs_err = 0.0f, rel_l2_err = 0.0f;
    compare_vectors("ConvLayer forward", a, b, max_abs_err, rel_l2_err);
    if (max_abs_err > 2e-4f || rel_l2_err > 2e-4f) {
        std::cout << "[FAIL] ConvLayer forward" << std::endl;
        return 1;
    }
    std::cout << "[PASS] ConvLayer forward" << std::endl;
    return 0;
}

static int check_forward_convtrans() {
    constexpr int N = 2;
    constexpr int Cin = 3;
    constexpr int H = 4;
    constexpr int W = 4;
    constexpr int Cout = 2;
    constexpr int k = 3;
    constexpr int pad = 1;
    constexpr int stride = 2;

    convtransLayer deconv(N, Cin, H, W, Cout, k, k, pad, stride, false);
    auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);

    deconv.W() = Matrix<CUDAfloat>::randn(Cin * Cout * k * k, 1) * 0.1f;
    deconv.b() = Matrix<CUDAfloat>::randn(Cout, 1) * 0.1f;

    deconv.eval(x);
    auto y_gpu = deconv.value().to_host();

    auto y_ref = convtrans2d_ref(x.to_host(), deconv.W().to_host(), deconv.b().to_host(),
                                 N, Cin, H, W, Cout, k, k, pad, stride);

    std::vector<float> a((size_t)y_gpu.num_row() * (size_t)y_gpu.num_col());
    std::vector<float> b((size_t)y_ref.num_row() * (size_t)y_ref.num_col());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = y_gpu.data()[i];
        b[i] = y_ref.data()[i];
    }

    float max_abs_err = 0.0f, rel_l2_err = 0.0f;
    compare_vectors("convtransLayer forward", a, b, max_abs_err, rel_l2_err);
    if (max_abs_err > 3e-4f || rel_l2_err > 3e-4f) {
        std::cout << "[FAIL] convtransLayer forward" << std::endl;
        return 1;
    }
    std::cout << "[PASS] convtransLayer forward" << std::endl;
    return 0;
}

static int check_bias_broadcast_conv() {
    constexpr int N = 2;
    constexpr int Cin = 2;
    constexpr int H = 5;
    constexpr int W = 5;
    constexpr int Cout = 3;
    constexpr int k = 3;

    int ret = 0;
    for (bool relu : {false, true}) {
        ConvLayer conv(N, Cin, H, W, Cout, k, k, 1, 1, relu);
        conv.W() = Matrix<CUDAfloat>::zeros(Cout * Cin * k * k, 1);

        Matrix<float> b_h("b", Cout, 1);
        b_h.elem(0, 0) = -0.3f;
        b_h.elem(1, 0) = 0.2f;
        b_h.elem(2, 0) = 0.7f;
        conv.b() = Matrix<CUDAfloat>(b_h);

        auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);
        conv.eval(x);
        auto y = conv.value().to_host();

        const int Hout = H;
        const int Wout = W;
        float max_err = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                float ref = b_h.elem(co, 0);
                if (relu) ref = std::max(0.0f, ref);
                for (int h = 0; h < Hout; ++h) {
                    for (int w = 0; w < Wout; ++w) {
                        float v = y.elem(idx_chw(co, h, w, Hout, Wout), n);
                        max_err = std::max(max_err, std::fabs(v - ref));
                    }
                }
            }
        }

        std::cout << "ConvLayer bias-only (relu=" << relu << ") max_abs_err=" << max_err << std::endl;
        if (max_err > 2e-6f) {
            std::cout << "[FAIL] ConvLayer bias-only" << std::endl;
            ret += 1;
        }
    }
    if (ret == 0) std::cout << "[PASS] ConvLayer bias-only" << std::endl;
    return ret;
}

static int check_bias_broadcast_convtrans() {
    constexpr int N = 2;
    constexpr int Cin = 3;
    constexpr int H = 4;
    constexpr int W = 4;
    constexpr int Cout = 2;
    constexpr int k = 3;
    constexpr int pad = 1;
    constexpr int stride = 2;
    constexpr int Hout = (H - 1) * stride - 2 * pad + k;
    constexpr int Wout = (W - 1) * stride - 2 * pad + k;

    int ret = 0;
    for (bool relu : {false, true}) {
        convtransLayer deconv(N, Cin, H, W, Cout, k, k, pad, stride, relu);
        deconv.W() = Matrix<CUDAfloat>::zeros(Cin * Cout * k * k, 1);

        Matrix<float> b_h("b", Cout, 1);
        b_h.elem(0, 0) = -0.25f;
        b_h.elem(1, 0) = 0.4f;
        deconv.b() = Matrix<CUDAfloat>(b_h);

        auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);
        deconv.eval(x);
        auto y = deconv.value().to_host();

        float max_err = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                float ref = b_h.elem(co, 0);
                if (relu) ref = std::max(0.0f, ref);
                for (int h = 0; h < Hout; ++h) {
                    for (int w = 0; w < Wout; ++w) {
                        float v = y.elem(idx_chw(co, h, w, Hout, Wout), n);
                        max_err = std::max(max_err, std::fabs(v - ref));
                    }
                }
            }
        }

        std::cout << "convtransLayer bias-only (relu=" << relu << ") max_abs_err=" << max_err << std::endl;
        if (max_err > 2e-6f) {
            std::cout << "[FAIL] convtransLayer bias-only" << std::endl;
            ret += 1;
        }
    }
    if (ret == 0) std::cout << "[PASS] convtransLayer bias-only" << std::endl;
    return ret;
}

template <class LayerT>
static int check_directional_derivative(const std::string& name,
                                        LayerT& layer,
                                        const Matrix<CUDAfloat>& x,
                                        const Matrix<CUDAfloat>& upstream,
                                        const Matrix<CUDAfloat>& direction,
                                        float eps,
                                        float abs_tol,
                                        float rel_tol) {
    std::list<Layer<CUDAfloat>*> layers = {&layer};
    freeze(layers);

    layer.eval(x);
    auto dx = layer.backward(x, Matrix<CUDAfloat>(upstream));
    const float lhs = scalar_loss(dx, direction);

    auto lp = forward_loss(layer, x + direction * eps, upstream);
    auto lm = forward_loss(layer, x - direction * eps, upstream);
    const float rhs = (lp - lm) / (2.0f * eps);

    unfreeze(layers);

    const float abs_err = std::fabs(lhs - rhs);
    const float rel_err = abs_err / std::max(std::fabs(lhs), 1e-6f);
    std::cout << name << " directional-derivative abs_err=" << abs_err
              << " rel_err=" << rel_err << std::endl;

    if (abs_err > abs_tol && rel_err > rel_tol) {
        std::cout << "[FAIL] " << name << " directional-derivative" << std::endl;
        return 1;
    }
    std::cout << "[PASS] " << name << " directional-derivative" << std::endl;
    return 0;
}

static int check_forward_sweep_conv() {
    struct Case { int N, Cin, H, W, Cout, k, pad, stride; bool relu; };
    const std::vector<Case> cases = {
        {2, 2, 5, 5, 3, 3, 1, 1, false},
        {1, 1, 6, 4, 2, 3, 0, 1, true},
        {2, 3, 6, 5, 2, 2, 1, 2, true},
    };

    int ret = 0;
    for (const auto& c : cases) {
        ConvLayer conv(c.N, c.Cin, c.H, c.W, c.Cout, c.k, c.k, c.pad, c.stride, c.relu);
        conv.W() = Matrix<CUDAfloat>::randn(c.Cout * c.Cin * c.k * c.k, 1) * 0.1f;
        conv.b() = Matrix<CUDAfloat>::randn(c.Cout, 1) * 0.1f;
        auto x = Matrix<CUDAfloat>::rand(c.Cin * c.H * c.W, c.N);

        conv.eval(x);
        auto y_gpu = conv.value().to_host();
        auto y_ref = conv2d_ref(x.to_host(), conv.W().to_host(), conv.b().to_host(),
                                c.N, c.Cin, c.H, c.W, c.Cout, c.k, c.k, c.pad, c.stride);
        if (c.relu) y_ref = relu_ref(y_ref);

        std::vector<float> a((size_t)y_gpu.num_row() * (size_t)y_gpu.num_col());
        std::vector<float> b((size_t)y_ref.num_row() * (size_t)y_ref.num_col());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = y_gpu.data()[i];
            b[i] = y_ref.data()[i];
        }

        float max_abs_err = 0.0f, rel_l2_err = 0.0f;
        compare_vectors("ConvLayer forward sweep", a, b, max_abs_err, rel_l2_err);
        if (max_abs_err > 3e-4f || rel_l2_err > 3e-4f) {
            std::cout << "[FAIL] ConvLayer forward sweep case" << std::endl;
            ret += 1;
        }
    }
    if (ret == 0) std::cout << "[PASS] ConvLayer forward sweep" << std::endl;
    return ret;
}

static int check_forward_sweep_convtrans() {
    struct Case { int N, Cin, H, W, Cout, k, pad, stride; bool relu; };
    const std::vector<Case> cases = {
        {2, 3, 4, 4, 2, 3, 1, 2, false},
        {1, 1, 5, 3, 2, 2, 0, 1, true},
        {2, 2, 3, 5, 3, 3, 1, 1, true},
    };

    int ret = 0;
    for (const auto& c : cases) {
        convtransLayer deconv(c.N, c.Cin, c.H, c.W, c.Cout, c.k, c.k, c.pad, c.stride, c.relu);
        deconv.W() = Matrix<CUDAfloat>::randn(c.Cin * c.Cout * c.k * c.k, 1) * 0.1f;
        deconv.b() = Matrix<CUDAfloat>::randn(c.Cout, 1) * 0.1f;
        auto x = Matrix<CUDAfloat>::rand(c.Cin * c.H * c.W, c.N);

        deconv.eval(x);
        auto y_gpu = deconv.value().to_host();
        auto y_ref = convtrans2d_ref(x.to_host(), deconv.W().to_host(), deconv.b().to_host(),
                                     c.N, c.Cin, c.H, c.W, c.Cout, c.k, c.k, c.pad, c.stride);
        if (c.relu) y_ref = relu_ref(y_ref);

        std::vector<float> a((size_t)y_gpu.num_row() * (size_t)y_gpu.num_col());
        std::vector<float> b((size_t)y_ref.num_row() * (size_t)y_ref.num_col());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = y_gpu.data()[i];
            b[i] = y_ref.data()[i];
        }

        float max_abs_err = 0.0f, rel_l2_err = 0.0f;
        compare_vectors("convtransLayer forward sweep", a, b, max_abs_err, rel_l2_err);
        if (max_abs_err > 4e-4f || rel_l2_err > 4e-4f) {
            std::cout << "[FAIL] convtransLayer forward sweep case" << std::endl;
            ret += 1;
        }
    }
    if (ret == 0) std::cout << "[PASS] convtransLayer forward sweep" << std::endl;
    return ret;
}

template <class LayerT>
static float forward_loss(LayerT& layer,
                          const Matrix<CUDAfloat>& x,
                          const Matrix<CUDAfloat>& upstream) {
    layer.eval(x);
    return scalar_loss(layer.value(), upstream);
}

static void compare_vectors(const std::string& tag,
                            const std::vector<float>& ana,
                            const std::vector<float>& num,
                            float& max_abs_err,
                            float& rel_l2_err) {
    max_abs_err = 0.0f;
    double num_norm = 0.0;
    double den_norm = 0.0;
    for (size_t i = 0; i < ana.size(); ++i) {
        float diff = ana[i] - num[i];
        max_abs_err = std::max(max_abs_err, std::fabs(diff));
        num_norm += (double)diff * (double)diff;
        den_norm += (double)ana[i] * (double)ana[i];
    }
    rel_l2_err = (float)std::sqrt(num_norm / std::max(den_norm, 1e-12));

    std::cout << tag << " max_abs_err=" << max_abs_err
              << " rel_l2_err=" << rel_l2_err << std::endl;
}

template <class LayerT>
static int check_input_gradient(const std::string& name,
                                LayerT& layer,
                                const Matrix<CUDAfloat>& x,
                                const Matrix<CUDAfloat>& upstream,
                                float eps,
                                float max_abs_tol,
                                float rel_l2_tol) {
    std::list<Layer<CUDAfloat>*> layers = {&layer};
    freeze(layers);

    layer.eval(x);
    auto dx = layer.backward(x, Matrix<CUDAfloat>(upstream));
    auto dx_h = dx.to_host();

    auto x_h = x.to_host();
    std::vector<float> numerical((size_t)x_h.num_row() * (size_t)x_h.num_col(), 0.0f);

    for (size_t idx = 0; idx < numerical.size(); ++idx) {
        auto xp = Matrix<float>(x_h);
        auto xm = Matrix<float>(x_h);
        const size_t r = idx % x_h.num_row();
        const size_t c = idx / x_h.num_row();
        xp.elem(r, c) += eps;
        xm.elem(r, c) -= eps;

        float lp = forward_loss(layer, Matrix<CUDAfloat>(xp), upstream);
        float lm = forward_loss(layer, Matrix<CUDAfloat>(xm), upstream);
        numerical[idx] = (lp - lm) / (2.0f * eps);
    }

    std::vector<float> analytic((size_t)dx_h.num_row() * (size_t)dx_h.num_col(), 0.0f);
    for (size_t i = 0; i < analytic.size(); ++i) {
        analytic[i] = dx_h.data()[i];
    }

    float max_abs_err = 0.0f, rel_l2_err = 0.0f;
    compare_vectors(name, analytic, numerical, max_abs_err, rel_l2_err);

    unfreeze(layers);

    if (max_abs_err > max_abs_tol || rel_l2_err > rel_l2_tol) {
        std::cout << "[FAIL] " << name
                  << " thresholds: max_abs<=" << max_abs_tol
                  << ", rel_l2<=" << rel_l2_tol << std::endl;
        return 1;
    }

    std::cout << "[PASS] " << name << std::endl;
    return 0;
}

int compute() {
    GPUSampler sampler(123);

    int ret = 0;

    ret += check_forward_conv();
    ret += check_forward_convtrans();
    ret += check_forward_sweep_conv();
    ret += check_forward_sweep_convtrans();
    ret += check_bias_broadcast_conv();
    ret += check_bias_broadcast_convtrans();

    {
        // Conv2D: N=2, Cin=2, H=W=5, Cout=3, k=3, pad=1, stride=1
        constexpr int N = 2;
        constexpr int Cin = 2;
        constexpr int H = 5;
        constexpr int W = 5;
        constexpr int Cout = 3;
        constexpr int k = 3;

        ConvLayer conv(N, Cin, H, W, Cout, k, k, 1, 1, false);

        auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);
        auto g = Matrix<CUDAfloat>::rand(Cout * H * W, N);

        ret += check_input_gradient("ConvLayer dL/dx", conv, x, g,
                                    1e-3f, 3e-2f, 2e-2f);

        auto v = Matrix<CUDAfloat>::rand(Cin * H * W, N) - 0.5f;
        ret += check_directional_derivative("ConvLayer", conv, x, g, v,
                            1e-3f, 3e-3f, 2e-3f);
    }

    {
        // ConvTranspose2D: N=2, Cin=3, H=W=4, Cout=2, k=3, pad=1, stride=2
        // Output spatial size: (4-1)*2 - 2*1 + 3 = 7
        constexpr int N = 2;
        constexpr int Cin = 3;
        constexpr int H = 4;
        constexpr int W = 4;
        constexpr int Cout = 2;
        constexpr int k = 3;
        constexpr int Hout = 7;
        constexpr int Wout = 7;

        convtransLayer deconv(N, Cin, H, W, Cout, k, k, 1, 2, false);

        auto x = Matrix<CUDAfloat>::rand(Cin * H * W, N);
        auto g = Matrix<CUDAfloat>::rand(Cout * Hout * Wout, N);

        ret += check_input_gradient("convtransLayer dL/dx", deconv, x, g,
                                    1e-3f, 5e-2f, 3e-2f);

        auto v = Matrix<CUDAfloat>::rand(Cin * H * W, N) - 0.5f;
        ret += check_directional_derivative("convtransLayer", deconv, x, g, v,
                            1e-3f, 5e-3f, 3e-3f);
    }

    if (ret == 0) {
        LOG_INFO("testConvLayers passed!");
    } else {
        LOG_ERROR("testConvLayers failed!");
    }

    return ret;
}

#endif
