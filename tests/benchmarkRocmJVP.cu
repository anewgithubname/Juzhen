/** ROCm Transformer JVP benchmark: cached derivative and forward + derivative. */
#include "../ml/layer.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
using namespace Juzhen;

static void check(hipError_t rc) {
    if (rc != hipSuccess) throw std::runtime_error(hipGetErrorString(rc));
}
static int count(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value && std::atoi(value) > 0 ? std::atoi(value) : fallback;
}
int compute() {
    constexpr int d=128, dk=128, ff=512, seq=64, batch=2, heads=4;
    const int warmup=count("JUZHEN_BENCH_WARMUP",20);
    const int iterations=count("JUZHEN_BENCH_ITERS",100);
    TransformerLayer<ROCMfloat> layer(d,dk,ff,seq,batch,heads);
    auto x=Matrix<ROCMfloat>::randn(d,seq*batch);
    auto u=Matrix<ROCMfloat>::randn(d,seq*batch);
    for (bool cached : {true,false}) {
        layer.eval(x);
        auto step=[&] {
            if (!cached) layer.eval(x);
            auto tangent=layer.jvp(x,u);
        };
        for (int i=0;i<warmup;++i) step();
        check(hipDeviceSynchronize());
        hipEvent_t a,b; check(hipEventCreate(&a)); check(hipEventCreate(&b));
        double wall=0, gpu=0;
        for (int i=0;i<iterations;++i) {
            const auto start=std::chrono::steady_clock::now();
            check(hipEventRecord(a)); step(); check(hipEventRecord(b));
            check(hipEventSynchronize(b));
            wall+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            float ms=0; check(hipEventElapsedTime(&ms,a,b)); gpu+=ms;
        }
        check(hipEventDestroy(a)); check(hipEventDestroy(b));
        if (!std::isfinite(layer.jvp(x,u).norm())) return 1;
        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT mode=" << (cached ? "cached_jvp" : "forward_jvp")
                  << " wall_mean_ms=" << wall/iterations << " gpu_mean_ms=" << gpu/iterations
                  << " iterations=" << iterations << "\n";
    }
    return 0;
}
