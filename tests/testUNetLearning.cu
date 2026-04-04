/**
 * @file testUNetLearning.cu
 * @brief End-to-end learning test for a tiny U-Net on a fixed synthetic denoising task.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>

using namespace Juzhen;

#if !defined(CUDA) || !defined(CUDNN_AVAILABLE)
int compute() {
    std::cout << "testUNetLearning requires CUDA + cuDNN. Skipping.\n";
    return 0;
}
#else

class TinyUNetLearner {
public:
    static constexpr int H = 32;
    static constexpr int W = 32;
    static constexpr int D = H * W;

    ConvLayer      enc1;
    ConvLayer      enc2;
    convtransLayer up1;
    ConvLayer      dec1;
    ConvLayer      head;

    Matrix<CUDAfloat> e1;
    Matrix<CUDAfloat> e2;
    Matrix<CUDAfloat> c_cat;

    explicit TinyUNetLearner(int batch_size)
        : enc1(batch_size, 1, H, W, 8, 3, 3, 1, 1, true),
          enc2(batch_size, 8, H, W, 16, 3, 3, 1, 2, true),
          up1 (batch_size, 16, H / 2, W / 2, 8, 4, 4, 1, 2, true),
          dec1(batch_size, 16, H, W, 8, 3, 3, 1, 1, true),
          head(batch_size, 8, H, W, 1, 3, 3, 1, 1, false),
          e1("e1", 8 * D, batch_size),
          e2("e2", 16 * (H / 2) * (W / 2), batch_size),
          c_cat("cc", 16 * D, batch_size) {}

    std::list<Layer<CUDAfloat>*> layers() {
        return {&head, &dec1, &up1, &enc2, &enc1};
    }

    void set_lr(float alpha) {
        for (auto* layer : layers()) {
            layer->adamWstate().alpha = alpha;
            layer->adambstate().alpha = alpha;
        }
    }

    const Matrix<CUDAfloat>& fwd(const Matrix<CUDAfloat>& x) {
        enc1.eval(x);
        e1 = enc1.value();
        enc2.eval(e1);
        e2 = enc2.value();
        up1.eval(e2);
        c_cat = vstack(std::vector<MatrixView<CUDAfloat>>{up1.value(), e1});
        dec1.eval(c_cat);
        head.eval(dec1.value());
        return head.value();
    }

    void bwd(const Matrix<CUDAfloat>& x, Matrix<CUDAfloat>&& g) {
        auto g2 = head.backward(dec1.value(), std::move(g));
        auto g_cat = dec1.backward(c_cat, std::move(g2));
        const size_t up_sz = 8ULL * D;
        auto g_up = g_cat.rows(0, up_sz);
        auto g_skip = g_cat.rows(up_sz, g_cat.num_row());
        auto g_e2 = up1.backward(e2, std::move(g_up));
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        auto g_inp = g_e1 + g_skip;
        enc1.backward(x, std::move(g_inp));
    }
};

static void draw_filled_rect(Matrix<float>& image,
                             int x0,
                             int y0,
                             int x1,
                             int y1,
                             float value) {
    for (int y = std::max(0, y0); y < std::min((int)TinyUNetLearner::H, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min((int)TinyUNetLearner::W, x1); ++x) {
            image.elem((size_t)y * TinyUNetLearner::W + (size_t)x, 0) = value;
        }
    }
}

static Matrix<float> make_clean_batch(int batch_size) {
    Matrix<float> batch("clean", TinyUNetLearner::D, batch_size);
    batch.zeros();

    for (int n = 0; n < batch_size; ++n) {
        Matrix<float> img("img", TinyUNetLearner::D, 1);
        img.zeros();

        int margin = 2 + (n % 3);
        draw_filled_rect(img, 4 + n, 5 + (n % 4), 14 + n, 13 + (n % 4), 1.0f);
        draw_filled_rect(img, 18 - (n % 3), 16 + (n % 5), 28 - (n % 3), 26 + (n % 2), 0.7f);
        draw_filled_rect(img, margin, 24 - (n % 5), margin + 6, 30 - (n % 3), 0.5f);

        for (int i = 0; i < TinyUNetLearner::D; ++i) {
            batch.elem(i, n) = img.elem(i, 0);
        }
    }
    return batch;
}

int compute() {
    GPUSampler sampler(11);

    constexpr int batch_size = 1;
    constexpr int train_iters = 4000;
    constexpr int log_every = 400;
    constexpr float lr = 5e-3f;

    auto clean_cpu = make_clean_batch(batch_size);

    auto clean = Matrix<CUDAfloat>(clean_cpu);
    auto inp = Matrix<CUDAfloat>(clean_cpu);

    TinyUNetLearner net(batch_size);
    net.set_lr(lr);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < train_iters; ++it) {
        const auto& pred = net.fwd(inp);
        auto diff = pred - clean;
        float loss = item(sum(sum(square(diff), 0), 1) / (float)batch_size);

        if (it == 0) initial_loss = loss;
        final_loss = loss;

        if (it % log_every == 0 || it + 1 == train_iters) {
            std::cout << "iter=" << it
                      << " loss=" << loss << std::endl;
        }

        net.bwd(inp, 2.f * diff / (float)batch_size);
    }

    auto elapsed_ms = time_in_ms(t0, std::chrono::high_resolution_clock::now());
    std::cout << "initial_loss=" << initial_loss
              << " final_loss=" << final_loss
              << " elapsed_ms=" << elapsed_ms << std::endl;

    const bool improved_ratio = final_loss < initial_loss * 0.70f;
    const bool low_absolute = final_loss < 80.0f;

    if (!improved_ratio || !low_absolute) {
        LOG_ERROR("testUNetLearning failed: initial_loss={}, final_loss={}", initial_loss, final_loss);
        return 1;
    }

    LOG_INFO("testUNetLearning passed: initial_loss={}, final_loss={}", initial_loss, final_loss);
    return 0;
}

#endif