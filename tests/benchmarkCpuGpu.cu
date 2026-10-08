/** Same Juzhen FP32 workloads on CPU and ROCm, timed to completion. */
#include "../ml/layer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef ROCM_HIP
#include <hip/hip_runtime.h>
#endif
using namespace Juzhen;
#ifdef ROCM_HIP
using D = ROCMfloat;
constexpr const char* backend = "ROCm";
#else
using D = float;
constexpr const char* backend = "CPU";
#endif

static void sync_device() {
#ifdef ROCM_HIP
    auto rc = hipDeviceSynchronize();
    if (rc != hipSuccess) throw std::runtime_error(hipGetErrorString(rc));
#endif
}
static int count(const char* name, int fallback) {
    const char* text = std::getenv(name);
    if (!text) return fallback;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*end || value < 1 || value > 100000) throw std::runtime_error(name);
    return static_cast<int>(value);
}
// Identical integer PRNG and exactly representable inputs in both builds.
static Matrix<float> input(int rows, int cols, uint32_t seed, float scale=1) {
    Matrix<float> m("benchmark_input", rows, cols);
    for (int c=0;c<cols;++c) for (int r=0;r<rows;++r) {
        seed = seed * 1664525u + 1013904223u;
        m.elem(r,c) = (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f) * scale;
    }
    return m;
}
template<class T> static Matrix<float> host(const Matrix<T>& m) {
    if constexpr (std::is_same_v<T,float>) return m;
    else return m.to_host();
}
static void dump(const std::string& name, const Matrix<D>& m) {
    const auto h=host(m);
    for (size_t c=0;c<h.num_col();++c) for (size_t r=0;r<h.num_row();++r)
        if (!std::isfinite(h.elem(r,c))) throw std::runtime_error("Non-finite output: "+name);
    const char* dir=std::getenv("JUZHEN_BENCH_OUTPUT");
    if (!dir) return;
    std::ofstream f(std::string(dir)+"/"+name+".bin", std::ios::binary);
    int32_t shape[2]={static_cast<int32_t>(h.num_row()),static_cast<int32_t>(h.num_col())};
    f.write(reinterpret_cast<const char*>(shape),sizeof(shape));
    for (size_t c=0;c<h.num_col();++c) for (size_t r=0;r<h.num_row();++r) {
        float v=h.elem(r,c); f.write(reinterpret_cast<const char*>(&v),sizeof(v));
    }
    if (!f) throw std::runtime_error("Cannot write benchmark output: "+name);
}
template<class F> static void measure(const std::string& name, F&& operation) {
    int warmup=count("JUZHEN_BENCH_WARMUP",5), iterations=count("JUZHEN_BENCH_ITERS",20);
    for (int i=0;i<warmup;++i) operation();
    sync_device();
    std::vector<double> samples;
    for (int i=0;i<iterations;++i) {
        auto start=std::chrono::steady_clock::now();
        operation(); sync_device();
        samples.push_back(std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count());
    }
    double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
    std::sort(samples.begin(),samples.end());
    std::cout << std::fixed << std::setprecision(6)
              << "RESULT case=" << name << " backend=" << backend << " mean_ms=" << mean
              << " p50_ms=" << samples[samples.size()/2]
              << " p95_ms=" << samples[static_cast<size_t>(0.95*(samples.size()-1))]
              << " iterations=" << iterations << std::endl;
}
static void gemm(int n) {
    Matrix<D> a(input(n,n,11)), b(input(n,n,22)), result("gemm",n,n);
    std::string name="gemm_"+std::to_string(n);
    measure(name,[&] { result=a*b; });
    dump(name,result);
}
static void transformer(int d, int seq, int batch, const std::string& tag) {
    TransformerLayer<D> layer(d,d,4*d,seq,batch,4,true);
    unsigned seed=100;
    for (auto& [name,param] : layer.checkpoint_parameters()) {
        if (name.find("gamma") != std::string::npos) param->ones();
        else if (name.find("beta") != std::string::npos || name[0]=='b') param->zeros();
        else *param=Matrix<D>(input(param->num_row(),param->num_col(),++seed,1.0f/std::sqrt(float(d))));
    }
    layer.set_lr(1e-4f);
    Matrix<D> x(input(d,seq*batch,33)), u(input(d,seq*batch,44));
    Matrix<D> result("result",d,seq*batch);
    measure(tag+"_forward",[&] { layer.eval(x); });
    dump(tag+"_forward",layer.value());
    // The previous forward cache is reused across tangent directions.
    measure(tag+"_cached_jvp",[&] { result=layer.jvp(x,u); });
    dump(tag+"_cached_jvp",result);
    measure(tag+"_forward_jvp",[&] { layer.eval(x); result=layer.jvp(x,u); });
    dump(tag+"_forward_jvp",result);
    measure(tag+"_train",[&] { layer.eval(x); result=layer.backward(x,Matrix<D>(u)); });
    dump(tag+"_train_dx",result);
    // Also validate actual updates, not merely that the training loop ran.
    for (auto& [name,param] : layer.checkpoint_parameters()) dump(tag+"_train_"+name,*param);
}
int compute() {
    std::cout << "META backend=" << backend;
#if defined(_WIN32) && !defined(JUZHEN_NO_BLAS)
    std::cout << " openblas_threads=" << openblas_get_num_threads()
              << " openblas_core=" << openblas_get_corename();
#endif
    std::cout << std::endl;
#ifdef ROCM_HIP
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop,0)!=hipSuccess) return 1;
    std::cout << "GPU " << prop.name << " arch=" << prop.gcnArchName << std::endl;
#endif
    gemm(512); gemm(1024);
    transformer(128,64,2,"tf_small");
    transformer(256,128,4,"tf_large");
    return 0;
}
