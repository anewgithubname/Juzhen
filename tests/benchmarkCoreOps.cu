/**
 * Repeatable single-device benchmarks. JUZHEN_BENCH_WARMUP and
 * JUZHEN_BENCH_ITERS control run length. CUDA uses events; CPU uses steady_clock.
 */
#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
using namespace Juzhen;

namespace {
#if defined(CUDA)
using Backend = CUDAfloat; constexpr const char* backend_name = "CUDA";
#elif defined(APPLE_SILICON)
using Backend = MPSfloat; constexpr const char* backend_name = "MPS";
#elif defined(ROCM_HIP)
using Backend = ROCMfloat; constexpr const char* backend_name = "ROCm";
#else
using Backend = float; constexpr const char* backend_name = "CPU";
#endif

int env_count(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 ? static_cast<int>(parsed) : fallback;
}

struct Result { std::string operation, shape; double ms, throughput; std::string unit; };

template <class F> double measure_ms(F&& operation, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) operation();
#if defined(CUDA)
    CudaErrorCheck(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CudaErrorCheck(cudaEventCreate(&start)); CudaErrorCheck(cudaEventCreate(&stop));
    CudaErrorCheck(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) operation();
    CudaErrorCheck(cudaEventRecord(stop)); CudaErrorCheck(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CudaErrorCheck(cudaEventElapsedTime(&elapsed, start, stop));
    CudaErrorCheck(cudaEventDestroy(start)); CudaErrorCheck(cudaEventDestroy(stop));
    return static_cast<double>(elapsed) / iterations;
#else
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) operation();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
#endif
}

void print(const Result& r) {
    std::cout << std::left << std::setw(24) << r.operation << std::setw(25) << r.shape
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.ms
              << std::setw(14) << std::setprecision(2) << r.throughput << " " << r.unit << '\n';
}

Result gemm(int w, int n) {
    constexpr int dim = 512;
    auto a = Matrix<Backend>::randn(dim, dim), b = Matrix<Backend>::randn(dim, dim);
    Matrix<Backend> out("bench_gemm", dim, dim);
    const double ms = measure_ms([&] { out = a * b; }, w, n);
    return {"GEMM", "512x512x512", ms, 2.0 * dim * dim * dim / (ms * 1e6), "GFLOP/s"};
}

Result conv2d(int w, int n) {
    constexpr int batch=4, cin=3, h=32, width=32, cout=16;
    ConvLayer layer(batch, cin, h, width, cout, 3, 3, 1, 1, false);
    auto input = Matrix<Backend>::randn(cin*h*width, batch);
    const double ms = measure_ms([&] { layer.eval(input); }, w, n);
    return {"Conv2D forward", "N4 C3x32x32 K16 3x3", ms, batch*1000.0/ms, "images/s"};
}

Result layernorm(int w, int n) {
    constexpr int dim=256, tokens=512;
    LayerNorm<Backend> layer(dim, tokens);
    auto input = Matrix<Backend>::randn(dim, tokens);
    Matrix<Backend> out("bench_ln", dim, tokens);
    const double ms = measure_ms([&] { out = layer.forward(input); }, w, n);
    return {"LayerNorm forward", "dim256 tokens512", ms, dim*tokens/(ms*1000.0), "Melem/s"};
}

Result attention(int w, int n) {
    constexpr int d=128, dk=128, ff=512, seq=64, batch=2, heads=4;
    TransformerLayer<Backend> layer(d, dk, ff, seq, batch, heads, true);
    auto input = Matrix<Backend>::randn(d, seq*batch);
    const double ms = measure_ms([&] { layer.eval(input); }, w, n);
    return {"Attention block forward", "d128 s64 b2 h4 ff512", ms, seq*batch*1000.0/ms, "tokens/s"};
}

Result adam(int w, int n) {
    constexpr int elements=1<<20;
    adam_state<Backend> state(1e-3, elements, 1);
    auto gradient = Matrix<Backend>::randn(elements, 1);
    const double ms = measure_ms([&] {
        auto updated = adam_update(std::move(gradient), state);
        gradient = std::move(updated);
    }, w, n);
    return {"Adam update", "1048576 elements", ms, elements/(ms*1000.0), "Melem/s"};
}
}

int compute() {
    global_rand_gen.seed(42);
#if defined(CUDA)
    GPUSampler sampler(42);
#endif
    const int warmup=env_count("JUZHEN_BENCH_WARMUP",3), iterations=env_count("JUZHEN_BENCH_ITERS",10);
    std::cout << "Juzhen core operator benchmark\nbackend=" << backend_name
              << " warmup=" << warmup << " iterations=" << iterations << "\n\n"
              << std::left << std::setw(24) << "operation" << std::setw(25) << "shape"
              << std::right << std::setw(12) << "mean_ms" << std::setw(14) << "throughput" << " unit\n";
    print(gemm(warmup,iterations)); print(conv2d(warmup,iterations));
    print(layernorm(warmup,iterations)); print(attention(warmup,iterations)); print(adam(warmup,iterations));
    return 0;
}
