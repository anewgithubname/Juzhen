/**
 * @file testDiffusionTrajectoryCompare.cu
 * @brief Deterministic diffusion training trajectory harness for CPU/CUDA comparison.
 */

#include "../ml/layer.hpp"
#include "../ml/dataloader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using namespace Juzhen;
using namespace std;

#if defined(APPLE_SILICON)
int compute() {
    cout << "testDiffusionTrajectoryCompare is intended for CPU/CUDA comparison. Skipping on Apple Silicon.\n";
    return 0;
}
#else

#if defined(CUDA)
using BackendT = CUDAfloat;
static const char* kBackendName = "cuda";
#else
using BackendT = float;
static const char* kBackendName = "cpu";
#endif

static Matrix<BackendT> vs(std::vector<MatrixView<BackendT>> ms) {
    return vstack(ms);
}

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) {
    return m.to_host();
}

template <>
Matrix<float> as_host<float>(const Matrix<float>& m) {
    return m;
}

template <class D>
static float matrix_rms_host(const Matrix<D>& m) {
    const auto h = as_host(m);
    const float n = (float)(h.num_row() * h.num_col());
    return std::sqrt(item(sum(sum(square(h), 0), 1)) / std::max(n, 1.f));
}

struct BackwardProbe {
    float g_out_rms = 0.0f;
    float g2_rms = 0.0f;
    float g_cat_rms = 0.0f;
    float g_up_rms = 0.0f;
    float g_skip_rms = 0.0f;
    float g_e2_rms = 0.0f;
    float g_e1_rms = 0.0f;
    float g_inp_rms = 0.0f;
};

static Matrix<float> resize_batch_bilinear(const Matrix<float>& src,
                                           int src_h,
                                           int src_w,
                                           int dst_h,
                                           int dst_w) {
    Matrix<float> dst("mnist_resized", dst_h * dst_w, src.num_col());
    const float scale_y = (float)src_h / (float)dst_h;
    const float scale_x = (float)src_w / (float)dst_w;
    for (size_t n = 0; n < src.num_col(); ++n) {
        for (int dy = 0; dy < dst_h; ++dy) {
            float sy = ((float)dy + 0.5f) * scale_y - 0.5f;
            sy = std::max(0.0f, std::min(sy, (float)(src_h - 1)));
            int y0 = (int)std::floor(sy);
            int y1 = std::min(y0 + 1, src_h - 1);
            float wy = sy - (float)y0;
            for (int dx = 0; dx < dst_w; ++dx) {
                float sx = ((float)dx + 0.5f) * scale_x - 0.5f;
                sx = std::max(0.0f, std::min(sx, (float)(src_w - 1)));
                int x0 = (int)std::floor(sx);
                int x1 = std::min(x0 + 1, src_w - 1);
                float wx = sx - (float)x0;
                auto at = [&](int y, int x) {
                    return src.elem((size_t)y * (size_t)src_w + (size_t)x, n);
                };
                float top = (1.f - wx) * at(y0, x0) + wx * at(y0, x1);
                float bot = (1.f - wx) * at(y1, x0) + wx * at(y1, x1);
                dst.elem((size_t)dy * (size_t)dst_w + (size_t)dx, n) = (1.f - wy) * top + wy * bot;
            }
        }
    }
    return dst;
}

struct VPSchedule {
    std::vector<float> sqrt_abar;
    std::vector<float> sqrt_one_minus_abar;

    explicit VPSchedule(int T)
        : sqrt_abar(T + 1), sqrt_one_minus_abar(T + 1) {
        std::vector<float> abar(T + 1, 1.f);
        for (int t = 1; t <= T; ++t) {
            const float s = (float)(t - 1) / (float)(T - 1);
            const float beta = 1e-4f + s * (0.02f - 1e-4f);
            const float alpha = 1.f - beta;
            abar[t] = abar[t - 1] * alpha;
            sqrt_abar[t] = std::sqrt(abar[t]);
            sqrt_one_minus_abar[t] = std::sqrt(std::max(1e-12f, 1.f - abar[t]));
        }
    }
};

class UNetScore {
public:
    static constexpr int H = 32;
    static constexpr int W = 32;
    static constexpr int d = H * W;

    ConvLayer enc1;
    ConvLayer enc2;
    convtransLayer up1;
    ConvLayer dec1;
    ConvLayer head;

    Matrix<BackendT> e1;
    Matrix<BackendT> e2;
    Matrix<BackendT> c_cat;

    explicit UNetScore(int bs)
        : enc1(bs, 3, 32, 32, 16, 3, 3, 1, 1, true),
          enc2(bs, 16, 32, 32, 32, 3, 3, 1, 2, true),
          up1(bs, 32, 16, 16, 16, 4, 4, 1, 2, true),
          dec1(bs, 32, 32, 32, 16, 3, 3, 1, 1, true),
          head(bs, 16, 32, 32, 1, 3, 3, 1, 1, false),
          e1("e1", 16 * d, bs),
          e2("e2", 32 * 16 * 16, bs),
          c_cat("cc", 32 * d, bs) {}

    std::list<Layer<BackendT>*> layers() {
        return {&head, &dec1, &up1, &enc2, &enc1};
    }

    void set_lr(float alpha) {
        for (auto* l : layers()) {
            l->adamWstate().alpha = alpha;
            l->adambstate().alpha = alpha;
        }
    }

    void reinit_from_host(float wstd) {
        for (auto* l : layers()) {
            Matrix<float> w_h("w_init", l->W().num_row(), l->W().num_col());
            for (size_t r = 0; r < w_h.num_row(); ++r) {
                for (size_t c = 0; c < w_h.num_col(); ++c) {
                    const float phase = (float)((r + 1) * 131 + (c + 1) * 17);
                    w_h.elem(r, c) = std::sin(phase * 0.001f) * wstd;
                }
            }
            const auto b_h = Matrix<float>::zeros(l->b().num_row(), l->b().num_col());
            l->W() = Matrix<BackendT>(w_h);
            l->b() = Matrix<BackendT>(b_h);
        }
    }

    const Matrix<BackendT>& fwd(const Matrix<BackendT>& inp) {
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

    void bwd(const Matrix<BackendT>& inp, Matrix<BackendT>&& g) {
        auto g2 = head.backward(dec1.value(), std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = 16ULL * d;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        enc1.backward(inp, std::move(g_inp));
    }

    BackwardProbe bwd_probe(const Matrix<BackendT>& inp, Matrix<BackendT>&& g) {
        BackwardProbe p;
        p.g_out_rms = matrix_rms_host(g);

        auto g2 = head.backward(dec1.value(), std::move(g));
        p.g2_rms = matrix_rms_host(g2);

        auto g_cat = dec1.backward(c_cat, std::move(g2));
        p.g_cat_rms = matrix_rms_host(g_cat);

        const size_t up_sz = 16ULL * d;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        p.g_up_rms = matrix_rms_host(g_up);
        p.g_skip_rms = matrix_rms_host(g_skip);

        auto g_e2 = up1.backward(e2, std::move(g_up));
        p.g_e2_rms = matrix_rms_host(g_e2);

        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        p.g_e1_rms = matrix_rms_host(g_e1);

        auto g_inp = g_e1 + g_skip;
        p.g_inp_rms = matrix_rms_host(g_inp);

        enc1.backward(inp, std::move(g_inp));
        return p;
    }
};

static float scalar_loss(const Matrix<float>& diff) {
    return item(sum(sum(square(diff), 0), 1) / (float)diff.num_col());
}

static float rms(const Matrix<float>& m) {
    const float n = (float)(m.num_row() * m.num_col());
    return std::sqrt(item(sum(sum(square(m), 0), 1)) / std::max(n, 1.f));
}

int compute() {
    constexpr int srcH = 28;
    constexpr int srcW = 28;
    constexpr int H = 32;
    constexpr int W = 32;
    constexpr int d = H * W;
    constexpr int batchsize = 32;
    constexpr int T = 50;
    constexpr int n_iters = 1000;
    constexpr float lr = 1e-3f;
    constexpr float wstd = 0.02f;

#if defined(CUDA)
    // Required by internal CUDA randn paths during layer setup.
    GPUSampler sampler(12345);
#endif

    const std::string mnist_dir = std::string(PROJECT_DIR) + "/datasets/MNIST";
    DataLoader<float, int> loader(mnist_dir, "train", batchsize);
    VPSchedule sched(T);
    UNetScore net(batchsize);
    net.reinit_from_host(wstd);
    net.set_lr(lr);

    std::cout << "backend=" << kBackendName << " iterations=" << n_iters << " batchsize=" << batchsize << std::endl;

    for (int it = 0; it < n_iters; ++it) {
        auto [x_cpu_raw, lbl] = loader.next_batch();
        while ((int)x_cpu_raw.num_col() != batchsize) {
            auto [xtmp, ltmp] = loader.next_batch();
            x_cpu_raw = std::move(xtmp);
        }

        const auto x0_img = resize_batch_bilinear(x_cpu_raw, srcH, srcW, H, W);
        const auto x0_h = Matrix<float>(2.f * (x0_img / 255.f) - 1.f);
        Matrix<float> eps_h("eps_det", d, batchsize);

        Matrix<float> t_sin_row("t_sin", 1, batchsize);
        Matrix<float> t_cos_row("t_cos", 1, batchsize);
        Matrix<float> c1_row("c1", 1, batchsize);
        Matrix<float> c2_row("c2", 1, batchsize);
        for (int b = 0; b < batchsize; ++b) {
            const int t_idx = 1 + ((it * 37 + b * 17) % T);
            const float tn = (float)t_idx / (float)T;
            t_sin_row.elem(0, b) = std::sin(2.f * 3.14159265358979323846f * tn);
            t_cos_row.elem(0, b) = std::cos(2.f * 3.14159265358979323846f * tn);
            c1_row.elem(0, b) = sched.sqrt_abar[t_idx];
            c2_row.elem(0, b) = sched.sqrt_one_minus_abar[t_idx];

            for (int r = 0; r < d; ++r) {
                const float phase = (float)((it + 1) * 29 + (b + 1) * 13 + (r + 1) * 7);
                eps_h.elem((size_t)r, (size_t)b) = std::sin(phase * 0.011f);
            }
        }

        const auto ones_d = Matrix<float>::ones(d, 1);
        const auto xt_h = hadmd(x0_h, ones_d * c1_row) + hadmd(eps_h, ones_d * c2_row);
        const auto inp_h = vstack<float>({
            MatrixView<float>(xt_h),
            MatrixView<float>(ones_d * t_sin_row),
            MatrixView<float>(ones_d * t_cos_row)
        });

        const auto inp = Matrix<BackendT>(inp_h);
        const auto eps = Matrix<BackendT>(eps_h);

        const auto& pred = net.fwd(inp);
        const auto diff_h = as_host(pred) - eps_h;
        const float loss = scalar_loss(diff_h);
        const float pred_rms = rms(as_host(pred));
        const float target_rms = rms(eps_h);
        const float first_pred = as_host(pred).elem(0, 0);

        std::cout << "iter=" << it
                  << " loss=" << loss
                  << " pred_rms=" << pred_rms
                  << " target_rms=" << target_rms
                  << " first_pred=" << first_pred
                  << std::endl;

        auto diff = pred - eps;
        auto upstream = 2.f * diff / (float)batchsize;
        if (it == 0) {
            auto probe = net.bwd_probe(inp, std::move(upstream));
            std::cout << "bwdprobe backend=" << kBackendName
                      << " g_out=" << probe.g_out_rms
                      << " g2=" << probe.g2_rms
                      << " g_cat=" << probe.g_cat_rms
                      << " g_up=" << probe.g_up_rms
                      << " g_skip=" << probe.g_skip_rms
                      << " g_e2=" << probe.g_e2_rms
                      << " g_e1=" << probe.g_e1_rms
                      << " g_inp=" << probe.g_inp_rms
                      << std::endl;
        } else {
            net.bwd(inp, std::move(upstream));
        }


    }

    std::cout << "summary backend=" << kBackendName
              << " enc1_w_rms=" << rms(as_host(net.enc1.W()))
              << " enc2_w_rms=" << rms(as_host(net.enc2.W()))
              << " up1_w_rms=" << rms(as_host(net.up1.W()))
              << " dec1_w_rms=" << rms(as_host(net.dec1.W()))
              << " head_w_rms=" << rms(as_host(net.head.W()))
              << std::endl;

    return 0;
}

#endif