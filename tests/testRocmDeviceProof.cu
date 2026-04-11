#include "../cpp/juzhen.hpp"

#ifdef ROCM_HIP
#include "../cpp/hipbackend.hpp"
#if defined(__HIPCC__) && __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#endif
#endif

int main() {
#ifndef ROCM_HIP
    LOG_INFO("ROCM_HIP is not enabled; skipping ROCm device proof test.");
    return 0;
#else
    const bool runtime_ok = Juzhen::RocmRuntimeAvailable();
    std::cout << "ROCM_RUNTIME_AVAILABLE=" << (runtime_ok ? 1 : 0) << std::endl;

#if defined(__HIPCC__) && __has_include(<hip/hip_runtime.h>)
    int dev_count = 0;
    hipError_t st = hipGetDeviceCount(&dev_count);
    std::cout << "HIP_DEVICE_COUNT=" << dev_count << std::endl;
    if (st != hipSuccess || dev_count <= 0) {
        LOG_ERROR("hipGetDeviceCount failed or no devices found.");
        return 2;
    }

    hipDeviceProp_t prop{};
    st = hipGetDeviceProperties(&prop, 0);
    if (st == hipSuccess) {
        std::cout << "HIP_DEVICE_NAME=" << prop.name << std::endl;
        std::cout << "HIP_ARCH=" << prop.gcnArchName << std::endl;
    }
#endif

    // This forces the ROCm CM GEMM path and validates numeric output.
    CM A(M("A", {{1, 2}, {3, 4}}));
    CM B(M("B", {{5, 6}, {7, 8}}));
    CM C = A * B;
    M Ch = C.to_host();

    M expected("E", {{19, 22}, {43, 50}});
    float err = (Ch - expected).norm();
    std::cout << "ROCM_GEMM_ERR=" << err << std::endl;
    if (err > 1e-4f) {
        LOG_ERROR("ROCm GEMM result mismatch: err={}", err);
        return 3;
    }

    LOG_INFO("ROCm device proof test passed.");
    return 0;
#endif
}
