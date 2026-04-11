/**
 * @file testDiffusionTrainIterCPU.cu
 * @brief Test one training iteration of diffusion model on CPU
 */

#include "../ml/layer.hpp"
#include "../ml/dataloader.hpp"
#include <iostream>
#include <cmath>

using namespace Juzhen;
using namespace std;

#if defined(CUDA) || defined(APPLE_SILICON) || defined(ROCM_HIP)
int compute() {
    cout << "testDiffusionTrainIterCPU requires a CPU-only build." << endl;
    return 0;
}
#else

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

    Matrix<float> e1;
    Matrix<float> e2;
    Matrix<float> c_cat;

    explicit UNetScore(int bs)
                : enc1(bs,   3, 32, 32,  16, 3, 3, 1, 1, true),
                    enc2(bs,  16, 32, 32,  32, 3, 3, 1, 2, true),
                    up1 (bs,  32, 16, 16,  16, 4, 4, 1, 2, true),
                    dec1(bs,  32, 32, 32,  16, 3, 3, 1, 1, true),
                    head(bs,  16, 32, 32,   1, 3, 3, 1, 1, false),
                    e1("e1", 16 * d, bs),
                    e2("e2", 32 * 16 * 16, bs),
                    c_cat("cc", 32 * d, bs) {}

    void set_lr(float alpha) {
        enc1.adamWstate().alpha = alpha;
        enc1.adambstate().alpha = alpha;
        enc2.adamWstate().alpha = alpha;
        enc2.adambstate().alpha = alpha;
        up1.adamWstate().alpha = alpha;
        up1.adambstate().alpha = alpha;
        dec1.adamWstate().alpha = alpha;
        dec1.adambstate().alpha = alpha;
        head.adamWstate().alpha = alpha;
        head.adambstate().alpha = alpha;
    }

    const Matrix<float>& fwd(const Matrix<float>& inp) {
        enc1.eval(inp);
        e1 = enc1.value();
        enc2.eval(e1);
        e2 = enc2.value();
        up1.eval(e2);
        c_cat = vstack<float>({MatrixView<float>(up1.value()), MatrixView<float>(e1)});
        dec1.eval(c_cat);
        head.eval(dec1.value());
        return head.value();
    }

    void bwd(const Matrix<float>& inp, Matrix<float>&& g) {
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
};

int compute() {
    global_rand_gen.seed(42);
    cout << "Testing one diffusion training iteration on CPU..." << endl;
    
    const int srcH = 28, srcW = 28;
    const int H = 32, W = 32, d = H * W;
    const int batchsize = 32;
    const int T = 50;
    
    cout << "Loading data..." << endl;
    const string mnist_dir = string(PROJECT_DIR) + "/datasets/MNIST";
    DataLoader<float, int> loader(mnist_dir, "train", batchsize);
    
    cout << "Creating network..." << endl;
    UNetScore net(batchsize);
    net.set_lr(1e-3f);
    
    cout << "Loading batch..." << endl;
    auto [x_cpu_raw, lbl] = loader.next_batch();
    cout << "Batch loaded: " << x_cpu_raw.num_row() << " x " << x_cpu_raw.num_col() << endl;
    
    cout << "Resizing..." << endl;
    auto x0_host = resize_batch_bilinear(x_cpu_raw, srcH, srcW, H, W);
    cout << "Resized: " << x0_host.num_row() << " x " << x0_host.num_col() << endl;
    
    cout << "Creating x0..." << endl;
    auto x0 = Matrix<float>(2.f * (x0_host / 255.f) - 1.f);
    cout << "x0: " << x0.num_row() << " x " << x0.num_col() << endl;
    
    cout << "Creating noise..." << endl;
    auto eps = Matrix<float>::randn(d, batchsize);
    cout << "eps: " << eps.num_row() << " x " << eps.num_col() << endl;
    
    cout << "Creating t matrices..." << endl;
    Matrix<float> t_sin_row("t_sin", 1, batchsize);
    Matrix<float> t_cos_row("t_cos", 1, batchsize);
    for (int b = 0; b < batchsize; ++b) {
        float tn = 0.5f;  // Fixed value for testing
        t_sin_row.elem(0, b) = std::sin(2.f * 3.14159265358979323846f * tn);
        t_cos_row.elem(0, b) = std::cos(2.f * 3.14159265358979323846f * tn);
    }
    cout << "t matrices created" << endl;
    
    cout << "Creating input..." << endl;
    auto t_sin_cu = Matrix<float>(t_sin_row);
    auto t_cos_cu = Matrix<float>(t_cos_row);
    auto ones_d_1 = Matrix<float>::ones(d, 1);
    auto ones_d_1_tsin = ones_d_1 * t_sin_cu;
    auto ones_d_1_tcos = ones_d_1 * t_cos_cu;
    auto inp = vstack<float>({MatrixView<float>(x0), MatrixView<float>(ones_d_1_tsin), MatrixView<float>(ones_d_1_tcos)});
    cout << "Input created: " << inp.num_row() << " x " << inp.num_col() << endl;
    
    cout << "Forward pass..." << endl;
    const auto& eps_pred = net.fwd(inp);
    cout << "Forward pass complete: " << eps_pred.num_row() << " x " << eps_pred.num_col() << endl;
    
    cout << "Computing loss..." << endl;
    auto diff_eps = eps_pred - eps;
    cout << "Loss computed, shape: " << diff_eps.num_row() << " x " << diff_eps.num_col() << endl;
    
    cout << "Backward pass..." << endl;
    auto grad = 2.f * diff_eps / (float)batchsize;
    cout << "Gradient computed" << endl;
    net.bwd(inp, std::move(grad));
    cout << "Backward pass complete" << endl;
    
    cout << "Test passed for 1 iteration!" << endl;
    
    // Now try multiple iterations
    cout << "\nRunning 10 iterations..." << endl;
    for (int iter = 0; iter < 10; ++iter) {
        cout << "Iter " << iter << ": loading batch..." << endl;
        auto [x_iter, lbl_iter] = loader.next_batch();
        cout << "  resizing..." << endl;
        auto x0_iter = Matrix<float>(2.f * (resize_batch_bilinear(x_iter, srcH, srcW, H, W) / 255.f) - 1.f);
        auto eps_iter = Matrix<float>::randn(d, batchsize);
        cout << "  forward..." << endl;
        auto ones_d = Matrix<float>::ones(d, 1);
        auto t_ones = Matrix<float>::ones(1, batchsize) * 0.5f;
        auto tsin_bcast = ones_d * (Matrix<float>::ones(1, batchsize) * std::sin(3.1415926535f));
        auto tcos_bcast = ones_d * (Matrix<float>::ones(1, batchsize) * std::cos(3.1415926535f));
        auto inp_iter = vstack<float>({MatrixView<float>(x0_iter), MatrixView<float>(tsin_bcast), MatrixView<float>(tcos_bcast)});
        const auto& pred_iter = net.fwd(inp_iter);
        cout << "  backward..." << endl;
        auto diff_iter = pred_iter - eps_iter;
        auto grad_iter = 2.f * diff_iter / (float)batchsize;
        net.bwd(inp_iter, std::move(grad_iter));
        cout << "  done" << endl;
    }
    
    cout << "All iterations passed!" << endl;
    return 0;
}

#endif
