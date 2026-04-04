/**
 * @file testCNNTrainParity.cu
 * @brief CPU vs GPU (cuDNN) training parity for the CNN used in demo_cnn_mnist.
 *
 * Run this test in the CPU build first (saves a reference loss file), then in
 * the CUDA build (compares against the saved reference).
 *
 *   1. CPU build:  runs 100 iters, writes PROJECT_DIR/tests/cnn_train_ref.txt
 *   2. CUDA build: runs 100 iters, reads the reference file and compares.
 */

#include "../ml/layer.hpp"
#include "../ml/dataloader.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace Juzhen;

// Architecture (same as demo_cnn_mnist) ──────────────────────────────────────
static constexpr int C = 1, H = 28, W = 28;
static constexpr int kH = 3, kW = 3, pad = 1;
static constexpr int C1 = 16, S1 = 1;
static constexpr int H1 = (H + 2*pad - kH)/S1 + 1;
static constexpr int W1 = (W + 2*pad - kW)/S1 + 1;
static constexpr int C2 = 32, S2 = 2;
static constexpr int H2 = (H1 + 2*pad - kH)/S2 + 1;
static constexpr int W2 = (W1 + 2*pad - kW)/S2 + 1;
static constexpr int C3 = 64, S3 = 2;
static constexpr int H3 = (H2 + 2*pad - kH)/S3 + 1;
static constexpr int W3 = (W2 + 2*pad - kW)/S3 + 1;
static constexpr int conv_out = C3 * H3 * W3;
static constexpr int fc_hidden = 256;
static constexpr int k_classes = 10;
static constexpr int BS = 64;
static constexpr int N_ITERS = 100;

static const std::string REF_FILE =
    std::string(PROJECT_DIR) + "/tests/cnn_train_ref.txt";

// Pre-load N_ITERS batches from MNIST into CPU float matrices.
static void load_batches(std::vector<Matrix<float>>& Xs,
                         std::vector<Matrix<float>>& Ys) {
    std::string mnist_dir = std::string(PROJECT_DIR) + "/datasets/MNIST";
    DataLoader<float, int> loader(mnist_dir, "train", BS);
    Xs.reserve(N_ITERS);
    Ys.reserve(N_ITERS);
    for (int i = 0; i < N_ITERS; ++i) {
        auto [Xb, yb] = loader.next_batch();
        while (Xb.num_col() != BS) std::tie(Xb, yb) = loader.next_batch();
        Xs.push_back(Xb / 255.0f);
        Ys.push_back(one_hot(yb, k_classes));
    }
}

#if defined(CUDA) && defined(CUDNN_AVAILABLE)
// ── GPU training path ────────────────────────────────────────────────────────
int compute() {
    GPUSampler sampler(42);
    global_rand_gen.seed(42);

    std::vector<Matrix<float>> Xs, Ys;
    load_batches(Xs, Ys);

    ConvLayer c1(BS, C,  H,  W,  C1, kH, kW, pad, S1);
    ConvLayer c2(BS, C1, H1, W1, C2, kH, kW, pad, S2);
    ConvLayer c3(BS, C2, H2, W2, C3, kH, kW, pad, S3);
    ReluLayer<CUDAfloat>   L1(fc_hidden, conv_out, BS);
    LinearLayer<CUDAfloat> L2(k_classes, fc_hidden, BS);

    std::list<Layer<CUDAfloat>*> net;
    net.push_back(&L2); net.push_back(&L1);
    net.push_back(&c3); net.push_back(&c2); net.push_back(&c1);

    std::vector<float> gpu_losses;
    gpu_losses.reserve(N_ITERS);

    for (int iter = 0; iter < N_ITERS; ++iter) {
        auto Xg = Matrix<CUDAfloat>(Xs[iter]);
        auto Yg = Matrix<CUDAfloat>(Ys[iter]);

        forward(net, Xg);
        LogisticLayer<CUDAfloat> loss(BS, std::move(Yg));
        loss.eval(L2.value());
        gpu_losses.push_back(loss.value().to_host().elem(0, 0));

        std::list<Layer<CUDAfloat>*> train;
        train.push_back(&loss); train.push_back(&L2); train.push_back(&L1);
        train.push_back(&c3);   train.push_back(&c2); train.push_back(&c1);
        backprop(train, Xg);
    }

    // Try to load CPU reference
    std::ifstream ref(REF_FILE);
    if (!ref.is_open()) {
        std::cout << "No CPU reference file found at " << REF_FILE << "\n";
        std::cout << "Run the CPU build first to generate it.\n";
        std::cout << "\nGPU losses (100 iters):\n";
        for (int i = 0; i < N_ITERS; ++i)
            std::cout << "  iter " << std::setw(3) << i
                      << " loss=" << std::fixed << std::setprecision(6)
                      << gpu_losses[i] << "\n";
        return 0;
    }

    std::vector<float> cpu_losses;
    std::string line;
    while (std::getline(ref, line)) {
        std::istringstream ss(line);
        float v;  ss >> v;  cpu_losses.push_back(v);
    }

    float max_diff = 0.0f;
    std::cout << std::setw(5) << "iter"
              << std::setw(12) << "loss_cpu"
              << std::setw(12) << "loss_gpu"
              << std::setw(12) << "diff" << "\n";
    for (int i = 0; i < N_ITERS; ++i) {
        const float diff = std::fabs(cpu_losses[i] - gpu_losses[i]);
        max_diff = std::max(max_diff, diff);
        std::cout << std::setw(5) << i
                  << std::fixed << std::setprecision(6)
                  << std::setw(12) << cpu_losses[i]
                  << std::setw(12) << gpu_losses[i]
                  << std::setw(12) << diff << "\n";
    }

    std::cout << "\nmax |loss_cpu - loss_gpu| over " << N_ITERS
              << " iters: " << max_diff << "\n";

    if (max_diff > 5e-2f) {
        std::cout << "[FAIL] testCNNTrainParity\n";
        return 1;
    }
    std::cout << "[PASS] testCNNTrainParity\n";
    return 0;
}

#else
// ── CPU training path (generates reference) ──────────────────────────────────
int compute() {
    global_rand_gen.seed(42);

    std::vector<Matrix<float>> Xs, Ys;
    load_batches(Xs, Ys);

    ConvLayer c1(BS, C,  H,  W,  C1, kH, kW, pad, S1);
    ConvLayer c2(BS, C1, H1, W1, C2, kH, kW, pad, S2);
    ConvLayer c3(BS, C2, H2, W2, C3, kH, kW, pad, S3);
    ReluLayer<float>   L1(fc_hidden, conv_out, BS);
    LinearLayer<float> L2(k_classes, fc_hidden, BS);

    std::list<Layer<float>*> net;
    net.push_back(&L2); net.push_back(&L1);
    net.push_back(&c3); net.push_back(&c2); net.push_back(&c1);

    std::ofstream out(REF_FILE);
    std::cout << "CPU training (100 iters) — writing reference to " << REF_FILE << "\n";
    std::cout << std::setw(5) << "iter" << std::setw(12) << "loss_cpu" << "\n";

    for (int iter = 0; iter < N_ITERS; ++iter) {
        const Matrix<float>& Xh = Xs[iter];
        const Matrix<float>& Yh = Ys[iter];

        forward(net, Xh);
        LogisticLayer<float> loss(BS, Yh);
        loss.eval(L2.value());
        const float lv = loss.value().elem(0, 0);
        out << lv << "\n";

        std::list<Layer<float>*> train;
        train.push_back(&loss); train.push_back(&L2); train.push_back(&L1);
        train.push_back(&c3);   train.push_back(&c2); train.push_back(&c1);
        backprop(train, Xh);

        std::cout << std::setw(5) << iter
                  << std::fixed << std::setprecision(6)
                  << std::setw(12) << lv << "\n";
    }

    std::cout << "Reference saved to " << REF_FILE << "\n";
    std::cout << "Now run the CUDA build's testCNNTrainParity to compare.\n";
    return 0;
}
#endif
