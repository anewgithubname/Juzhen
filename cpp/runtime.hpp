/**
 * @file runtime.hpp
 * @brief Process-wide setup and teardown
 *
 * Shared by main() in launcher.cu, the iOS app and language bindings, so the
 * backend is set up in one place.
 */

#pragma once

namespace Juzhen {

namespace runtime_detail {
inline bool &initialized() {
    static bool flag = false;
    return flag;
}
}  // namespace runtime_detail

// Set up the GPU backend: the cuBLAS handle (TF32 math when NVIDIA_TF32 is
// set) or the Metal device. Calling it again does nothing until shutdown().
inline void init() {
    if (runtime_detail::initialized()) return;
#ifdef CUDA
    CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
    // opt-in TF32 tensor-core math for all cuBLAS GEMMs (fp32 storage,
    // 10-bit mantissa multiplies). Enable with NVIDIA_TF32=1.
    const char *tf32 = std::getenv("NVIDIA_TF32");
    if (tf32 && *tf32 && std::string(tf32) != "0") {
        CuBLASErrorCheck(cublasSetMathMode(Matrix<CUDAfloat>::global_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        std::cout << "TF32 tensor-core math: ON (NVIDIA_TF32=" << tf32 << ")" << std::endl;
    }
#endif
#ifdef APPLE_SILICON
    mpsInit();
#endif
    runtime_detail::initialized() = true;
}

// Free every block of every memory pool, cached or still in use. Matrices
// that hold one of them must not be used afterwards.
inline void release_memory() {
#ifdef APPLE_SILICON
    mpsSynchronize();  // let the GPU finish with the buffers first
    Memory<MPSfloat>::release_all();
#endif
#ifdef CUDA
    Memory<CUDAfloat>::release_all();
#endif
#ifdef ROCM_HIP
    Memory<ROCMfloat>::release_all();
#endif
    Memory<double>::release_all();
    Memory<float>::release_all();
    Memory<int>::release_all();
}

// Free the cached blocks that no matrix is using; live matrices stay valid.
inline void release_cached() {
#ifdef APPLE_SILICON
    mpsSynchronize();  // a cached buffer may still be read by queued GPU work
    Memory<MPSfloat>::release_cached();
#endif
#ifdef CUDA
    Memory<CUDAfloat>::release_cached();
#endif
#ifdef ROCM_HIP
    Memory<ROCMfloat>::release_cached();
#endif
    Memory<double>::release_cached();
    Memory<float>::release_cached();
    Memory<int>::release_cached();
}

// release_memory(), then undo init(). Failures are logged rather than thrown,
// so it is safe to call while cleaning up after an error.
inline void shutdown() {
    release_memory();
    if (!runtime_detail::initialized()) return;
#ifdef APPLE_SILICON
    mpsDestroy();
#endif
#ifdef CUDA
    if (cublasDestroy(Matrix<CUDAfloat>::global_handle) != CUBLAS_STATUS_SUCCESS)
        LOG_ERROR("cublasDestroy failed");
#endif
    runtime_detail::initialized() = false;
}

}  // namespace Juzhen
