/**
 * @file testConvBackendParity.cu
 * @brief Backend parity test for ConvLayer and ConvTransLayer against CPU reference.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Juzhen;

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
                                int pad, int stride,
                                bool use_relu) {
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
                                acc += x.elem(idx_chw(ci, ih, iw, H, W), n)
                                    * w.elem(idx_w_conv(co, ci, kh, kw, Cin, kH, kW), 0);
                            }
                        }
                    }
                    y.elem(idx_chw(co, oh, ow, Hout, Wout), n) = acc;
                }
            }
        }
    }

    return use_relu ? relu_ref(y) : y;
}

static Matrix<float> convtrans2d_ref(const Matrix<float>& x,
                                     const Matrix<float>& w,
                                     const Matrix<float>& b,
                                     int N, int Cin, int H, int W,
                                     int Cout, int kH, int kW,
                                     int pad, int stride,
                                     bool use_relu) {
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
                                y.elem(idx_chw(co, oh, ow, Hout, Wout), n) +=
                                    xv * w.elem(idx_w_deconv(ci, co, kh, kw, Cout, kH, kW), 0);
                            }
                        }
                    }
                }
            }
        }
    }

    return use_relu ? relu_ref(y) : y;
}

static void compare_matrix(const Matrix<float>& got,
                           const Matrix<float>& ref,
                           const std::string& tag,
                           float& max_abs,
                           float& rel_l2) {
    max_abs = 0.0f;
    double num = 0.0;
    double den = 0.0;

    for (size_t c = 0; c < got.num_col(); ++c) {
        for (size_t r = 0; r < got.num_row(); ++r) {
            const float d = got.elem(r, c) - ref.elem(r, c);
            max_abs = std::max(max_abs, std::fabs(d));
            num += (double)d * (double)d;
            den += (double)ref.elem(r, c) * (double)ref.elem(r, c);
        }
    }

    rel_l2 = (float)std::sqrt(num / std::max(den, 1e-12));
    std::cout << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2 << std::endl;
}

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) {
    return m.to_host();
}

template <>
Matrix<float> as_host<float>(const Matrix<float>& m) {
    return m;
}

static float scalar_loss(const Matrix<float>& y, const Matrix<float>& g) {
    return item(sum(sum(hadmd(y, g), 0), 1));
}

template <class BackendT, class LayerT>
static float forward_loss_host(LayerT& layer,
                               const Matrix<float>& x_h,
                               const Matrix<float>& upstream_h) {
    layer.eval(Matrix<BackendT>(x_h));
    return scalar_loss(as_host(layer.value()), upstream_h);
}

template <class BackendT, class LayerT>
static int check_input_gradient_fd(const std::string& name,
                                   LayerT& layer,
                                   const Matrix<float>& x_h,
                                   const Matrix<float>& upstream_h,
                                   float eps,
                                   float max_abs_tol,
                                   float rel_l2_tol) {
    std::list<Layer<BackendT>*> layers = {&layer};
    freeze(layers);

    layer.eval(Matrix<BackendT>(x_h));
    auto dx_h = as_host(layer.backward(Matrix<BackendT>(x_h), Matrix<BackendT>(upstream_h)));

    Matrix<float> num("num_dx", x_h.num_row(), x_h.num_col());
    for (size_t c = 0; c < x_h.num_col(); ++c) {
        for (size_t r = 0; r < x_h.num_row(); ++r) {
            auto xp = Matrix<float>(x_h);
            auto xm = Matrix<float>(x_h);
            xp.elem(r, c) += eps;
            xm.elem(r, c) -= eps;

            const float lp = forward_loss_host<BackendT>(layer, xp, upstream_h);
            const float lm = forward_loss_host<BackendT>(layer, xm, upstream_h);
            num.elem(r, c) = (lp - lm) / (2.0f * eps);
        }
    }

    unfreeze(layers);

    float max_abs = 0.0f, rel_l2 = 0.0f;
    compare_matrix(dx_h, num, name, max_abs, rel_l2);
    if (max_abs > max_abs_tol || rel_l2 > rel_l2_tol) {
        std::cout << "[FAIL] " << name << "\n";
        return 1;
    }
    return 0;
}

int compute() {
    global_rand_gen.seed(777);

#if defined(CUDA)
    GPUSampler sampler(777);
    using BackendT = CUDAfloat;
#elif defined(ROCM_HIP)
    using BackendT = ROCMfloat;
#elif defined(APPLE_SILICON)
    using BackendT = MPSfloat;
#else
    using BackendT = float;
#endif

    constexpr int N = 2;
    constexpr int Cin = 2;
    constexpr int H = 5;
    constexpr int W = 4;
    constexpr int Cout = 3;
    constexpr int kH = 3;
    constexpr int kW = 3;

    int ret = 0;

    for (bool relu : {false, true}) {
        const int pad = 1;
        const int stride = 1;

        auto x_h = Matrix<float>::randn(Cin * H * W, N);
        auto w_h = Matrix<float>::randn(Cout * Cin * kH * kW, 1) * 0.1f;
        auto b_h = Matrix<float>::randn(Cout, 1) * 0.1f;

        ConvLayer conv(N, Cin, H, W, Cout, kH, kW, pad, stride, relu);
        conv.W() = Matrix<BackendT>(w_h);
        conv.b() = Matrix<BackendT>(b_h);
        conv.eval(Matrix<BackendT>(x_h));

        auto got = as_host(conv.value());
        auto ref = conv2d_ref(x_h, w_h, b_h, N, Cin, H, W, Cout, kH, kW, pad, stride, relu);

        float max_abs = 0.0f, rel_l2 = 0.0f;
        compare_matrix(got, ref, std::string("ConvLayer relu=") + (relu ? "true" : "false"), max_abs, rel_l2);
        if (max_abs > 6e-4f || rel_l2 > 6e-4f) {
            std::cout << "[FAIL] ConvLayer parity\n";
            ret += 1;
        }
    }

    for (bool relu : {false, true}) {
        const int pad = 1;
        const int stride = 2;

        auto x_h = Matrix<float>::randn(Cin * H * W, N);
        auto w_h = Matrix<float>::randn(Cin * Cout * kH * kW, 1) * 0.1f;
        auto b_h = Matrix<float>::randn(Cout, 1) * 0.1f;

        ConvTransLayer deconv(N, Cin, H, W, Cout, kH, kW, pad, stride, relu);
        deconv.W() = Matrix<BackendT>(w_h);
        deconv.b() = Matrix<BackendT>(b_h);
        deconv.eval(Matrix<BackendT>(x_h));

        auto got = as_host(deconv.value());
        auto ref = convtrans2d_ref(x_h, w_h, b_h, N, Cin, H, W, Cout, kH, kW, pad, stride, relu);

        float max_abs = 0.0f, rel_l2 = 0.0f;
        compare_matrix(got, ref, std::string("ConvTransLayer relu=") + (relu ? "true" : "false"), max_abs, rel_l2);
        if (max_abs > 8e-4f || rel_l2 > 8e-4f) {
            std::cout << "[FAIL] ConvTransLayer parity\n";
            ret += 1;
        }
    }

    {
        const int pad = 1;
        const int stride = 1;
        const bool relu = false;

        auto x_h = Matrix<float>::randn(Cin * H * W, N);
        auto w_h = Matrix<float>::randn(Cout * Cin * kH * kW, 1) * 0.1f;
        auto b_h = Matrix<float>::randn(Cout, 1) * 0.1f;
        const int Hout = (H + 2 * pad - kH) / stride + 1;
        const int Wout = (W + 2 * pad - kW) / stride + 1;
        auto upstream_h = Matrix<float>::randn(Cout * Hout * Wout, N);

        ConvLayer conv(N, Cin, H, W, Cout, kH, kW, pad, stride, relu);
        conv.W() = Matrix<BackendT>(w_h);
        conv.b() = Matrix<BackendT>(b_h);

        ret += check_input_gradient_fd<BackendT>(
            "ConvLayer dL/dx finite-diff", conv, x_h, upstream_h,
            1e-3f, 4e-2f, 3e-2f);
    }

    {
        const int pad = 1;
        const int stride = 2;
        const bool relu = false;

        auto x_h = Matrix<float>::randn(Cin * H * W, N);
        auto w_h = Matrix<float>::randn(Cin * Cout * kH * kW, 1) * 0.1f;
        auto b_h = Matrix<float>::randn(Cout, 1) * 0.1f;
        const int Hout = (H - 1) * stride - 2 * pad + kH;
        const int Wout = (W - 1) * stride - 2 * pad + kW;
        auto upstream_h = Matrix<float>::randn(Cout * Hout * Wout, N);

        ConvTransLayer deconv(N, Cin, H, W, Cout, kH, kW, pad, stride, relu);
        deconv.W() = Matrix<BackendT>(w_h);
        deconv.b() = Matrix<BackendT>(b_h);

        ret += check_input_gradient_fd<BackendT>(
            "ConvTransLayer dL/dx finite-diff", deconv, x_h, upstream_h,
            1e-3f, 6e-2f, 4e-2f);
    }

    if (ret == 0) {
        std::cout << "[PASS] testConvBackendParity\n";
    }
    return ret;
}
