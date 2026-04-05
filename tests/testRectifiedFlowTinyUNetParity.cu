/**
 * @file testRectifiedFlowTinyUNetParity.cu
 * @brief CPU-reference vs MPS parity check for a few rectified-flow TinyUNet steps on a toy dataset.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace Juzhen;

#if !defined(APPLE_SILICON)
int compute() {
    std::cout << "testRectifiedFlowTinyUNetParity requires APPLE_SILICON. Skipping.\n";
    return 0;
}
#else

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

static void diff_stats(const Matrix<float>& a,
                       const Matrix<float>& b,
                       float& max_abs,
                       float& rel_l2) {
    max_abs = 0.0f;
    double num = 0.0;
    double den = 0.0;
    for (size_t c = 0; c < a.num_col(); ++c) {
        for (size_t r = 0; r < a.num_row(); ++r) {
            const float d = a.elem(r, c) - b.elem(r, c);
            max_abs = std::max(max_abs, std::fabs(d));
            num += (double)d * (double)d;
            den += (double)a.elem(r, c) * (double)a.elem(r, c);
        }
    }
    rel_l2 = (float)std::sqrt(num / std::max(den, 1e-12));
}

static void compare_matrix(const Matrix<float>& a,
                           const Matrix<float>& b,
                           const std::string& tag,
                           float& max_abs,
                           float& rel_l2) {
    diff_stats(a, b, max_abs, rel_l2);
    std::cout << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2 << std::endl;
}

static float mse_loss_host(const Matrix<float>& pred, const Matrix<float>& target) {
    double acc = 0.0;
    const double n = (double)pred.num_row() * (double)pred.num_col();
    for (size_t c = 0; c < pred.num_col(); ++c) {
        for (size_t r = 0; r < pred.num_row(); ++r) {
            const double d = (double)pred.elem(r, c) - (double)target.elem(r, c);
            acc += d * d;
        }
    }
    return (float)(acc / std::max(n, 1.0));
}

static void draw_rect(Matrix<float>& image,
                      int x0,
                      int y0,
                      int x1,
                      int y1,
                      float value,
                      int width,
                      int height) {
    for (int y = std::max(0, y0); y < std::min(height, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(width, x1); ++x) {
            image.elem((size_t)y * (size_t)width + (size_t)x, 0) = value;
        }
    }
}

static Matrix<float> make_clean_batch(int batch_size, int iter, int height, int width) {
    Matrix<float> batch("clean", (size_t)height * (size_t)width, batch_size);
    batch.zeros();
    for (int n = 0; n < batch_size; ++n) {
        Matrix<float> img("img", (size_t)height * (size_t)width, 1);
        img.zeros();

        const int shift = (iter + 2 * n) % 4;
        draw_rect(img, 2 + shift, 3 + (n % 3), 8 + shift, 8 + (n % 3), 1.0f, width, height);
        draw_rect(img, 9 + (n % 4), 5 + shift, 14 + (n % 4), 10 + shift, 0.7f, width, height);
        draw_rect(img, 4 + (n % 5), 10 - shift, 11 + (n % 5), 15 - shift, 0.45f, width, height);

        for (size_t r = 0; r < img.num_row(); ++r) {
            batch.elem(r, (size_t)n) = img.elem(r, 0);
        }
    }
    return batch;
}

static Matrix<float> make_noise_batch(int batch_size, int iter, int height, int width) {
    const int d = height * width;
    Matrix<float> noise("noise", d, batch_size);
    for (int n = 0; n < batch_size; ++n) {
        for (int r = 0; r < d; ++r) {
            const float phase = (float)((iter + 1) * 29 + (n + 1) * 17 + (r + 1) * 11);
            noise.elem((size_t)r, (size_t)n) = 0.8f * std::sin(phase * 0.019f);
        }
    }
    return noise;
}

static Matrix<float> make_time_row(int batch_size, int iter, int total_iters) {
    Matrix<float> t("time", 1, batch_size);
    for (int n = 0; n < batch_size; ++n) {
        const int idx = 1 + ((iter * 7 + n * 3) % total_iters);
        t.elem(0, (size_t)n) = (float)idx / (float)(total_iters + 1);
    }
    return t;
}

struct CpuConvLayer {
    int N, Cin, H, W, Cout, kH, kW, pad, stride, Hout, Wout;
    bool use_relu;
    Matrix<float> Wm, bm, val;
    adam_state<float> adamW, adamb;

    CpuConvLayer(int N, int Cin, int H, int W, int Cout, int kH, int kW, int pad, int stride, bool relu)
        : N(N), Cin(Cin), H(H), W(W), Cout(Cout), kH(kH), kW(kW), pad(pad), stride(stride),
          Hout((H + 2 * pad - kH) / stride + 1), Wout((W + 2 * pad - kW) / stride + 1),
          use_relu(relu),
          Wm(Matrix<float>::randn(Cout * Cin * kH * kW, 1) * 0.001f),
          bm(Matrix<float>::zeros(Cout, 1)),
          val("conv_val", Cout * Hout * Wout, N),
          adamW(0.0001, Cout * Cin * kH * kW, 1),
          adamb(0.0001, Cout, 1) {}

    void set_lr(float alpha) {
        adamW.alpha = alpha;
        adamb.alpha = alpha;
    }

    const Matrix<float>& eval(const Matrix<float>& x) {
        val.zeros();
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        float acc = bm.elem(co, 0);
                        for (int ci = 0; ci < Cin; ++ci) {
                            for (int kh = 0; kh < kH; ++kh) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    const int ih = oh * stride - pad + kh;
                                    const int iw = ow * stride - pad + kw;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    acc += x.elem(idx_chw(ci, ih, iw, H, W), n)
                                        * Wm.elem(idx_w_conv(co, ci, kh, kw, Cin, kH, kW), 0);
                                }
                            }
                        }
                        if (use_relu && acc < 0.0f) acc = 0.0f;
                        val.elem(idx_chw(co, oh, ow, Hout, Wout), n) = acc;
                    }
                }
            }
        }
        return val;
    }

    Matrix<float> backward(const Matrix<float>& x, Matrix<float>&& upstream) {
        Matrix<float> t(upstream);
        if (use_relu) {
            for (size_t c = 0; c < t.num_col(); ++c) {
                for (size_t r = 0; r < t.num_row(); ++r) {
                    if (val.elem(r, c) <= 0.0f) t.elem(r, c) = 0.0f;
                }
            }
        }

        Matrix<float> dW("dW", Wm.num_row(), 1);
        Matrix<float> db("db", bm.num_row(), 1);
        Matrix<float> dx("dx", Cin * H * W, N);
        dW.zeros();
        db.zeros();
        dx.zeros();

        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        const float gv = t.elem(idx_chw(co, oh, ow, Hout, Wout), n);
                        db.elem(co, 0) += gv;
                        for (int ci = 0; ci < Cin; ++ci) {
                            for (int kh = 0; kh < kH; ++kh) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    const int ih = oh * stride - pad + kh;
                                    const int iw = ow * stride - pad + kw;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    const size_t widx = idx_w_conv(co, ci, kh, kw, Cin, kH, kW);
                                    dW.elem(widx, 0) += x.elem(idx_chw(ci, ih, iw, H, W), n) * gv;
                                    dx.elem(idx_chw(ci, ih, iw, H, W), n) += Wm.elem(widx, 0) * gv;
                                }
                            }
                        }
                    }
                }
            }
        }

        Wm -= adam_update(std::move(dW), adamW);
        bm -= adam_update(std::move(db), adamb);
        return dx;
    }
};

struct CpuConvTransLayer {
    int N, Cin, H, W, Cout, kH, kW, pad, stride, Hout, Wout;
    bool use_relu;
    Matrix<float> Wm, bm, val;
    adam_state<float> adamW, adamb;

    CpuConvTransLayer(int N, int Cin, int H, int W, int Cout, int kH, int kW, int pad, int stride, bool relu)
        : N(N), Cin(Cin), H(H), W(W), Cout(Cout), kH(kH), kW(kW), pad(pad), stride(stride),
          Hout((H - 1) * stride - 2 * pad + kH), Wout((W - 1) * stride - 2 * pad + kW),
          use_relu(relu),
          Wm(Matrix<float>::randn(Cin * Cout * kH * kW, 1) * 0.001f),
          bm(Matrix<float>::zeros(Cout, 1)),
          val("deconv_val", Cout * Hout * Wout, N),
          adamW(0.0001, Cin * Cout * kH * kW, 1),
          adamb(0.0001, Cout, 1) {}

    void set_lr(float alpha) {
        adamW.alpha = alpha;
        adamb.alpha = alpha;
    }

    const Matrix<float>& eval(const Matrix<float>& x) {
        val.zeros();
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        val.elem(idx_chw(co, oh, ow, Hout, Wout), n) = bm.elem(co, 0);
                    }
                }
            }
        }

        for (int n = 0; n < N; ++n) {
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
                                    val.elem(idx_chw(co, oh, ow, Hout, Wout), n) +=
                                        xv * Wm.elem(idx_w_deconv(ci, co, kh, kw, Cout, kH, kW), 0);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (use_relu) {
            for (size_t c = 0; c < val.num_col(); ++c) {
                for (size_t r = 0; r < val.num_row(); ++r) {
                    if (val.elem(r, c) < 0.0f) val.elem(r, c) = 0.0f;
                }
            }
        }
        return val;
    }

    Matrix<float> backward(const Matrix<float>& x, Matrix<float>&& upstream) {
        Matrix<float> t(upstream);
        if (use_relu) {
            for (size_t c = 0; c < t.num_col(); ++c) {
                for (size_t r = 0; r < t.num_row(); ++r) {
                    if (val.elem(r, c) <= 0.0f) t.elem(r, c) = 0.0f;
                }
            }
        }

        Matrix<float> dW("dW", Wm.num_row(), 1);
        Matrix<float> db("db", bm.num_row(), 1);
        Matrix<float> dx("dx", Cin * H * W, N);
        dW.zeros();
        db.zeros();
        dx.zeros();

        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                float sb = 0.0f;
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        sb += t.elem(idx_chw(co, oh, ow, Hout, Wout), n);
                    }
                }
                db.elem(co, 0) += sb;
            }
        }

        for (int n = 0; n < N; ++n) {
            for (int ci = 0; ci < Cin; ++ci) {
                for (int ih = 0; ih < H; ++ih) {
                    for (int iw = 0; iw < W; ++iw) {
                        float acc_dx = 0.0f;
                        for (int co = 0; co < Cout; ++co) {
                            for (int kh = 0; kh < kH; ++kh) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    const int oh = ih * stride - pad + kh;
                                    const int ow = iw * stride - pad + kw;
                                    if (oh < 0 || oh >= Hout || ow < 0 || ow >= Wout) continue;
                                    const float gv = t.elem(idx_chw(co, oh, ow, Hout, Wout), n);
                                    const size_t widx = idx_w_deconv(ci, co, kh, kw, Cout, kH, kW);
                                    acc_dx += gv * Wm.elem(widx, 0);
                                    dW.elem(widx, 0) += x.elem(idx_chw(ci, ih, iw, H, W), n) * gv;
                                }
                            }
                        }
                        dx.elem(idx_chw(ci, ih, iw, H, W), n) = acc_dx;
                    }
                }
            }
        }

        Wm -= adam_update(std::move(dW), adamW);
        bm -= adam_update(std::move(db), adamb);
        return dx;
    }
};

class CpuTinyRectifiedUNet {
public:
    static constexpr int H = 16;
    static constexpr int W = 16;
    static constexpr int D = H * W;
    static constexpr int C0 = 2;
    static constexpr int C1 = 8;
    static constexpr int C2 = 12;

    CpuConvLayer enc1;
    CpuConvLayer enc2;
    CpuConvTransLayer up1;
    CpuConvLayer dec1;
    CpuConvLayer head;

    Matrix<float> e1;
    Matrix<float> e2;
    Matrix<float> c_cat;

        explicit CpuTinyRectifiedUNet(int batch_size, int head_k = 3, int head_pad = 1)
        : enc1(batch_size, C0, H, W, C1, 3, 3, 1, 1, true),
          enc2(batch_size, C1, H, W, C2, 3, 3, 1, 2, true),
          up1(batch_size, C2, H / 2, W / 2, C1, 4, 4, 1, 2, true),
          dec1(batch_size, C1 + C1, H, W, C1, 3, 3, 1, 1, true),
                    head(batch_size, C1, H, W, 1, head_k, head_k, head_pad, 1, false),
          e1("e1", C1 * D, batch_size),
          e2("e2", C2 * (H / 2) * (W / 2), batch_size),
          c_cat("cc", (C1 + C1) * D, batch_size) {}

    void set_lr(float alpha) {
        enc1.set_lr(alpha);
        enc2.set_lr(alpha);
        up1.set_lr(alpha);
        dec1.set_lr(alpha);
        head.set_lr(alpha);
    }

    const Matrix<float>& fwd(const Matrix<float>& x) {
        e1 = enc1.eval(x);
        e2 = enc2.eval(e1);
        const auto up = up1.eval(e2);
        c_cat = vstack<float>({up, e1});
        const auto dec = dec1.eval(c_cat);
        return head.eval(dec);
    }

    void bwd(const Matrix<float>& x, Matrix<float>&& g) {
        auto g2 = head.backward(dec1.val, std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = (size_t)C1 * (size_t)D;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        enc1.backward(x, std::move(g_inp));
    }
};

class MpsTinyRectifiedUNet {
public:
    static constexpr int H = 16;
    static constexpr int W = 16;
    static constexpr int D = H * W;
    static constexpr int C0 = 2;
    static constexpr int C1 = 8;
    static constexpr int C2 = 12;

    ConvLayer enc1;
    ConvLayer enc2;
    convtransLayer up1;
    ConvLayer dec1;
    ConvLayer head;

    Matrix<MPSfloat> e1;
    Matrix<MPSfloat> e2;
    Matrix<MPSfloat> c_cat;

        explicit MpsTinyRectifiedUNet(int batch_size, int head_k = 3, int head_pad = 1)
        : enc1(batch_size, C0, H, W, C1, 3, 3, 1, 1, true),
          enc2(batch_size, C1, H, W, C2, 3, 3, 1, 2, true),
          up1(batch_size, C2, H / 2, W / 2, C1, 4, 4, 1, 2, true),
          dec1(batch_size, C1 + C1, H, W, C1, 3, 3, 1, 1, true),
                    head(batch_size, C1, H, W, 1, head_k, head_k, head_pad, 1, false),
          e1("e1", C1 * D, batch_size),
          e2("e2", C2 * (H / 2) * (W / 2), batch_size),
          c_cat("cc", (C1 + C1) * D, batch_size) {}

    void set_lr(float alpha) {
        enc1.adamWstate().alpha = alpha; enc1.adambstate().alpha = alpha;
        enc2.adamWstate().alpha = alpha; enc2.adambstate().alpha = alpha;
        up1.adamWstate().alpha = alpha;  up1.adambstate().alpha = alpha;
        dec1.adamWstate().alpha = alpha; dec1.adambstate().alpha = alpha;
        head.adamWstate().alpha = alpha; head.adambstate().alpha = alpha;
    }

    void copy_params_from(CpuTinyRectifiedUNet& src) {
        enc1.W() = Matrix<MPSfloat>(src.enc1.Wm); enc1.b() = Matrix<MPSfloat>(src.enc1.bm);
        enc2.W() = Matrix<MPSfloat>(src.enc2.Wm); enc2.b() = Matrix<MPSfloat>(src.enc2.bm);
        up1.W()  = Matrix<MPSfloat>(src.up1.Wm);  up1.b()  = Matrix<MPSfloat>(src.up1.bm);
        dec1.W() = Matrix<MPSfloat>(src.dec1.Wm); dec1.b() = Matrix<MPSfloat>(src.dec1.bm);
        head.W() = Matrix<MPSfloat>(src.head.Wm); head.b() = Matrix<MPSfloat>(src.head.bm);
    }

    Matrix<float> W_host_enc1() { return enc1.W().to_host(); }
    Matrix<float> W_host_enc2() { return enc2.W().to_host(); }
    Matrix<float> W_host_up1()  { return up1.W().to_host(); }
    Matrix<float> W_host_dec1() { return dec1.W().to_host(); }
    Matrix<float> W_host_head() { return head.W().to_host(); }
    Matrix<float> b_host_enc1() { return enc1.b().to_host(); }
    Matrix<float> b_host_enc2() { return enc2.b().to_host(); }
    Matrix<float> b_host_up1()  { return up1.b().to_host(); }
    Matrix<float> b_host_dec1() { return dec1.b().to_host(); }
    Matrix<float> b_host_head() { return head.b().to_host(); }

    const Matrix<MPSfloat>& fwd(const Matrix<MPSfloat>& x) {
        enc1.eval(x);
        e1 = enc1.value();
        enc2.eval(e1);
        e2 = enc2.value();
        up1.eval(e2);
        c_cat = vstack(std::vector<MatrixView<MPSfloat>>{up1.value(), e1});
        dec1.eval(c_cat);
        head.eval(dec1.value());
        return head.value();
    }

    void bwd(const Matrix<MPSfloat>& x, Matrix<MPSfloat>&& g) {
        auto g2 = head.backward(dec1.value(), std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = (size_t)C1 * (size_t)D;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        enc1.backward(x, std::move(g_inp));
    }
};

static void accumulate_diff(const Matrix<float>& a,
                            const Matrix<float>& b,
                            float& max_abs_all,
                            float& rel_l2_all) {
    float max_abs = 0.0f;
    float rel_l2 = 0.0f;
    diff_stats(a, b, max_abs, rel_l2);
    max_abs_all = std::max(max_abs_all, max_abs);
    rel_l2_all = std::max(rel_l2_all, rel_l2);
}

static void compare_params(CpuTinyRectifiedUNet& cpu,
                           MpsTinyRectifiedUNet& mps,
                           int iter,
                           float& max_abs,
                           float& rel_l2) {
    max_abs = 0.0f;
    rel_l2 = 0.0f;
    accumulate_diff(cpu.enc1.Wm, mps.W_host_enc1(), max_abs, rel_l2);
    accumulate_diff(cpu.enc2.Wm, mps.W_host_enc2(), max_abs, rel_l2);
    accumulate_diff(cpu.up1.Wm,  mps.W_host_up1(),  max_abs, rel_l2);
    accumulate_diff(cpu.dec1.Wm, mps.W_host_dec1(), max_abs, rel_l2);
    accumulate_diff(cpu.head.Wm, mps.W_host_head(), max_abs, rel_l2);
    accumulate_diff(cpu.enc1.bm, mps.b_host_enc1(), max_abs, rel_l2);
    accumulate_diff(cpu.enc2.bm, mps.b_host_enc2(), max_abs, rel_l2);
    accumulate_diff(cpu.up1.bm,  mps.b_host_up1(),  max_abs, rel_l2);
    accumulate_diff(cpu.dec1.bm, mps.b_host_dec1(), max_abs, rel_l2);
    accumulate_diff(cpu.head.bm, mps.b_host_head(), max_abs, rel_l2);
    std::cout << "iter=" << iter << " params max_abs=" << max_abs << " rel_l2=" << rel_l2 << std::endl;
}

int compute() {
    global_rand_gen.seed(2026);

    constexpr int batch_size = 4;
    constexpr int train_iters = 6;
    constexpr float lr = 3e-3f;
    constexpr float fwd_tol_abs = 8e-4f;
    constexpr float fwd_tol_rel = 8e-4f;
    constexpr float loss_tol_abs = 8e-4f;
    constexpr float loss_tol_rel = 8e-4f;
    constexpr float param_tol_abs = 2.5e-3f;
    constexpr float param_tol_rel = 2.5e-3f;

    auto run_case = [&](int head_k, int head_pad, const std::string& case_name) -> int {
        CpuTinyRectifiedUNet net_h(batch_size, head_k, head_pad);
        MpsTinyRectifiedUNet net_m(batch_size, head_k, head_pad);
        net_m.copy_params_from(net_h);
        net_h.set_lr(lr);
        net_m.set_lr(lr);

        float initial_loss_h = 0.0f;
        float initial_loss_m = 0.0f;
        float final_loss_h = 0.0f;
        float final_loss_m = 0.0f;

        for (int it = 0; it < train_iters; ++it) {
            const auto clean_h = make_clean_batch(batch_size, it, CpuTinyRectifiedUNet::H, CpuTinyRectifiedUNet::W);
            const auto noise_h = make_noise_batch(batch_size, it, CpuTinyRectifiedUNet::H, CpuTinyRectifiedUNet::W);
            const auto t_h = make_time_row(batch_size, it, train_iters);

            const auto ones_d = Matrix<float>::ones(CpuTinyRectifiedUNet::D, 1);
            const auto xt_h = hadmd(clean_h, ones_d * (1.0 - t_h)) + hadmd(noise_h, ones_d * t_h);
            const auto inp_h = vstack<float>({xt_h, ones_d * t_h});
            const auto target_h = noise_h - clean_h;

            const auto inp_m = Matrix<MPSfloat>(inp_h);

            const auto pred_h = Matrix<float>(net_h.fwd(inp_h));
            const auto pred_m = net_m.fwd(inp_m).to_host();

            float max_abs = 0.0f;
            float rel_l2 = 0.0f;
            compare_matrix(pred_h, pred_m, case_name + " iter=" + std::to_string(it) + " pred", max_abs, rel_l2);
            if (max_abs > fwd_tol_abs || rel_l2 > fwd_tol_rel) {
                std::cout << "[FAIL] " << case_name << " forward parity mismatch at iter=" << it << "\n";
                return 1;
            }

            const float loss_h = mse_loss_host(pred_h, target_h);
            const float loss_m = mse_loss_host(pred_m, target_h);
            const float loss_abs = std::fabs(loss_h - loss_m);
            const float loss_rel = loss_abs / std::max(std::fabs(loss_h), 1e-12f);
            std::cout << case_name << " iter=" << it
                      << " loss_cpu=" << loss_h
                      << " loss_mps=" << loss_m
                      << " loss_abs=" << loss_abs
                      << " loss_rel=" << loss_rel
                      << std::endl;
            if (loss_abs > loss_tol_abs && loss_rel > loss_tol_rel) {
                std::cout << "[FAIL] " << case_name << " loss parity mismatch at iter=" << it << "\n";
                return 1;
            }

            if (it == 0) {
                initial_loss_h = loss_h;
                initial_loss_m = loss_m;
            }
            final_loss_h = loss_h;
            final_loss_m = loss_m;

            auto upstream_h = 2.f * (pred_h - target_h) / (float)(pred_h.num_row() * pred_h.num_col());
            net_h.bwd(inp_h, Matrix<float>(upstream_h));
            net_m.bwd(inp_m, Matrix<MPSfloat>(upstream_h));

            compare_params(net_h, net_m, it, max_abs, rel_l2);
            if (max_abs > param_tol_abs || rel_l2 > param_tol_rel) {
                std::cout << "[FAIL] " << case_name << " parameter parity mismatch at iter=" << it << "\n";
                return 1;
            }
        }

        std::cout << case_name
                  << " initial_loss_cpu=" << initial_loss_h
                  << " final_loss_cpu=" << final_loss_h
                  << " initial_loss_mps=" << initial_loss_m
                  << " final_loss_mps=" << final_loss_m
                  << std::endl;

        if (!(final_loss_h < initial_loss_h && final_loss_m < initial_loss_m)) {
            std::cout << "[FAIL] " << case_name << " training did not improve on both backends\n";
            return 1;
        }

        return 0;
    };

    if (run_case(3, 1, "head3x3") != 0) return 1;
    if (run_case(1, 0, "head1x1") != 0) return 1;

    std::cout << "[PASS] testRectifiedFlowTinyUNetParity\n";
    return 0;
}
#endif