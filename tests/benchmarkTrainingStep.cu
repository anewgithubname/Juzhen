/** End-to-end Transformer forward + backward + Adam update benchmark. */
#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
using namespace Juzhen;

#if defined(CUDA)
using Backend = CUDAfloat; constexpr const char* backend_name = "CUDA";
#elif defined(APPLE_SILICON)
using Backend = MPSfloat; constexpr const char* backend_name = "MPS";
#elif defined(ROCM_HIP)
using Backend = ROCMfloat; constexpr const char* backend_name = "ROCm";
#else
using Backend = float; constexpr const char* backend_name = "CPU";
#endif

static int env_count(const char* name, int fallback) {
    const char* s=std::getenv(name); if (!s) return fallback;
    char* end=nullptr; long v=std::strtol(s,&end,10);
    return end!=s && *end=='\0' && v>0 ? static_cast<int>(v) : fallback;
}

template<class F> static double one_step_ms(F&& f) {
#if defined(CUDA)
    cudaEvent_t a,b; CudaErrorCheck(cudaEventCreate(&a)); CudaErrorCheck(cudaEventCreate(&b));
    CudaErrorCheck(cudaEventRecord(a)); f(); CudaErrorCheck(cudaEventRecord(b));
    CudaErrorCheck(cudaEventSynchronize(b)); float ms=0; CudaErrorCheck(cudaEventElapsedTime(&ms,a,b));
    CudaErrorCheck(cudaEventDestroy(a)); CudaErrorCheck(cudaEventDestroy(b)); return ms;
#else
    auto a=std::chrono::steady_clock::now(); f(); auto b=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(b-a).count();
#endif
}

static double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(),v.end());
    const size_t i=static_cast<size_t>(q*(v.size()-1)); return v[i];
}

int compute() {
    constexpr int d=128, dk=128, ff=512, seq=64, batch=2, heads=4;
    const int warmup=env_count("JUZHEN_BENCH_WARMUP",3), iterations=env_count("JUZHEN_BENCH_ITERS",20);
    global_rand_gen.seed(42);
#if defined(CUDA)
    GPUSampler sampler(42); size_t free_before=0,total=0; CudaErrorCheck(cudaMemGetInfo(&free_before,&total));
#endif
    TransformerLayer<Backend> layer(d,dk,ff,seq,batch,heads,true); layer.set_lr(1e-4f);
    auto input=Matrix<Backend>::randn(d,seq*batch);
    auto upstream_seed=Matrix<Backend>::randn(d,seq*batch);
    auto step=[&] {
        layer.eval(input);
        auto upstream=Matrix<Backend>(upstream_seed);
        auto dx=layer.backward(input,std::move(upstream));
        (void)dx;
    };
    for(int i=0;i<warmup;++i) step();
#if defined(CUDA)
    CudaErrorCheck(cudaDeviceSynchronize()); size_t min_free=0; CudaErrorCheck(cudaMemGetInfo(&min_free,&total));
#endif
    std::vector<double> samples; samples.reserve(iterations);
    for(int i=0;i<iterations;++i) {
        samples.push_back(one_step_ms(step));
#if defined(CUDA)
        size_t now=0; CudaErrorCheck(cudaMemGetInfo(&now,&total)); min_free=std::min(min_free,now);
#endif
    }
    const double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
    const double p50=percentile(samples,0.50), p95=percentile(samples,0.95);
    double peak_mb=0.0;
#if defined(CUDA)
    peak_mb=static_cast<double>(free_before-min_free)/(1024.0*1024.0);
#endif
    const double tokens_per_second=seq*batch*1000.0/mean;
    std::cout << "End-to-end Transformer training step\n"
              << "backend="<<backend_name<<" config=d128,dk128,ff512,seq64,batch2,heads4"
              << " warmup="<<warmup<<" iterations="<<iterations<<"\n"
              << std::fixed<<std::setprecision(3)
              << "mean_ms="<<mean<<" p50_ms="<<p50<<" p95_ms="<<p95
              << " tokens_per_second="<<tokens_per_second<<" peak_device_mb="<<peak_mb<<"\n"
              << "RESULT backend="<<backend_name<<" mean_ms="<<mean<<" p50_ms="<<p50
              << " p95_ms="<<p95<<" tokens_per_second="<<tokens_per_second
              << " peak_device_mb="<<peak_mb<<"\n";
    return 0;
}
