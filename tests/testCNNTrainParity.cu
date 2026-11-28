/**
 * @file testCNNTrainParity.cu
 * @brief Training parity of the CNN of demo_cnn_mnist between backends.
 *
 * Every build trains the same network for 100 Adam steps on the same MNIST
 * batches, starting from the same weights, and compares its 100 losses with
 * the checked-in reference tests/cnn_train_ref.txt (written by a CPU build).
 *
 * The starting weights come from an integer formula, not from the random
 * generators: those differ between backends and, for std::normal_distribution,
 * between standard libraries, and a network that learns amplifies a different
 * start into a different loss curve. The weights have the variance of the
 * layers' default rule, so the network does learn (loss about 2.35 -> 0.3).
 *
 * To regenerate the reference: run a CPU build with JUZHEN_WRITE_CNN_REF=1.
 */

#include "../ml/layer.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include "../ml/dataloader.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Juzhen;

// the element type of the build's ConvLayer
#if defined(CUDA) && defined(CUDNN_AVAILABLE)
using Backend = CUDAfloat;
static const char* const backend_name = "CUDA";
#elif defined(ROCM_HIP)
using Backend = ROCMfloat;
static const char* const backend_name = "ROCm";
#elif defined(APPLE_SILICON)
using Backend = MPSfloat;
static const char* const backend_name = "Metal";
#else
using Backend = float;
static const char* const backend_name = "CPU";
#endif

static constexpr int BS = 64;
static constexpr int N_ITERS = 100;
// Rounding differences between backends grow as training goes on: the first
// steps must agree closely, the whole run within a looser bound.
static constexpr int EARLY_ITERS = 10;
static constexpr float EARLY_TOLERANCE = 1e-4f;
static constexpr float TOLERANCE = 5e-3f;

static const std::string REF_FILE = std::string(PROJECT_DIR) + "/tests/cnn_train_ref.txt";

// uniform in [-1, 1), the same on every platform: integer arithmetic only (splitmix64)
static float unit(unsigned long long k, unsigned long long salt) {
    unsigned long long z = k + salt * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return float(double(z >> 40) / 16777216.0 * 2.0 - 1.0);
}

// weights in storage order (column after column) with the variance of N(0, std^2)
template <class D>
static void fill(Matrix<D>& w, double std, int salt) {
    Matrix<float> host("w", w.num_row(), w.num_col());
    const float scale = float(std * std::sqrt(3.0));
    for (size_t j = 0; j < host.num_col(); ++j)
        for (size_t i = 0; i < host.num_row(); ++i)
            host.elem(i, j) = scale * unit(j * host.num_row() + i, salt);
    if constexpr (std::is_same_v<D, float>)
        w = std::move(host);
    else
        w = Matrix<D>(host);
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(42);  // the constructors draw on the device before fill() replaces the weights
#endif
    global_rand_gen.seed(42);

    // the architecture of demo_cnn_mnist: 28x28 -> 16@28x28 -> 32@14x14 -> 64@7x7 -> 256 -> 10
    ConvLayer c1(BS, 1, 28, 28, 16, 3, 3, 1, 1);
    ConvLayer c2(BS, 16, 28, 28, 32, 3, 3, 1, 2);
    ConvLayer c3(BS, 32, 14, 14, 64, 3, 3, 1, 2);
    ReluLayer<Backend> L1(256, 64 * 7 * 7, BS);
    LinearLayer<Backend> L2(10, 256, BS);
    fill(c1.W(), std::sqrt(2.0 / 9), 1);
    fill(c2.W(), std::sqrt(2.0 / 144), 2);
    fill(c3.W(), std::sqrt(2.0 / 288), 3);
    fill(L1.W(), std::sqrt(2.0 / 3136), 4);
    fill(L2.W(), std::sqrt(1.0 / 256), 5);

    std::list<Layer<Backend>*> net{&L2, &L1, &c3, &c2, &c1};
    AdamWOptions options;  // plain Adam at the default learning rate
    options.weight_decay = 0;
    AdamW<Backend> optimizer(parameters(net), options);

    DataLoader<float, int> loader(std::string(PROJECT_DIR) + "/datasets/MNIST", "train", BS);
    std::vector<float> losses;
    for (int iter = 0; iter < N_ITERS; ++iter) {
        auto [Xb, yb] = loader.next_batch();
        while (Xb.num_col() != BS) std::tie(Xb, yb) = loader.next_batch();
        const Matrix<Backend> X(Xb / 255.0f);  // divided on the host in every build
        optimizer.zero_grad();
        auto [loss, grad] = softmax_cross_entropy(forward(net, X), yb);  // class numbers from the loader
        backward(net, X, std::move(grad));
        optimizer.step();
        losses.push_back(item(loss));
    }

    const char* write = std::getenv("JUZHEN_WRITE_CNN_REF");
    if (write && *write && std::string(write) != "0") {
        if (!std::is_same_v<Backend, float>) {
            std::cout << "the reference is written by a CPU build; this is a " << backend_name << " build\n";
            return 1;
        }
        std::ofstream out(REF_FILE);
        for (float v : losses) out << std::setprecision(9) << v << "\n";
        std::cout << "wrote " << REF_FILE << ": loss " << losses.front() << " -> " << losses.back() << "\n";
        return out.good() ? 0 : 1;
    }

    std::ifstream ref(REF_FILE);
    std::vector<float> reference;
    for (float v; ref >> v;) reference.push_back(v);
    if (reference.size() != size_t(N_ITERS)) {
        std::cout << "no usable reference at " << REF_FILE << " (" << reference.size() << " values); "
                  << "run a CPU build with JUZHEN_WRITE_CNN_REF=1\n";
        return 77;  // ctest SKIP_RETURN_CODE
    }

    float early = 0, whole = 0;
    for (int i = 0; i < N_ITERS; ++i) {
        const float diff = std::fabs(reference[i] - losses[i]);
        if (!(diff == diff)) whole = early = INFINITY;  // nan
        if (i < EARLY_ITERS) early = std::max(early, diff);
        whole = std::max(whole, diff);
        if (i % 10 == 0 || i == N_ITERS - 1)
            std::cout << std::setw(5) << i << std::fixed << std::setprecision(6) << std::setw(12) << reference[i]
                      << std::setw(12) << losses[i] << std::setw(12) << diff << "\n";
    }
    std::cout << std::defaultfloat << "max |reference - " << backend_name << "| over the first " << EARLY_ITERS
              << " iterations: " << early << " (allowed " << EARLY_TOLERANCE << "), over all " << N_ITERS << ": "
              << whole << " (allowed " << TOLERANCE << ")\n";
    const bool learned = losses.back() < 0.5f * losses.front();
    if (!learned) std::cout << "the network did not learn: loss " << losses.front() << " -> " << losses.back() << "\n";
    const bool ok = early <= EARLY_TOLERANCE && whole <= TOLERANCE && learned;
#ifdef CUDA
    if (!ok)
        std::cout << "on Ampere and later GPUs run with NVIDIA_TF32_OVERRIDE=0 (ctest sets it): TensorFloat-32 "
                     "changes the convolution results by about 1e-4\n";
#endif
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " testCNNTrainParity (" << backend_name << ")\n";
    return ok ? 0 : 1;
}
