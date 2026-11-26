#pragma once
// Integer ids (token ids, class numbers) on the host and on the device.
//
// The public type is the host Matrix<int> that Juzhen already has (read<int>,
// DataLoader<float, int>): one id per entry, read in storage order. The
// operators that need the ids on a GPU copy them into a pooled Matrix<D>
// block as bit patterns; no new allocator, no raw device malloc.
#include "../cpp/juzhen.hpp"
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ROCM_HIP
#include "../cpp/hipbackend.hpp"
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#endif

namespace Juzhen {
namespace token_detail {

// The ids of a Matrix<int> in storage order: column after column, so a
// seq_len x batch matrix gives sequence 0, then sequence 1, ...
inline std::vector<int> flat(const Matrix<int>& ids) {
    std::vector<int> out;
    out.reserve(ids.num_row() * ids.num_col());
    for (size_t j = 0; j < ids.num_col(); ++j)
        for (size_t i = 0; i < ids.num_row(); ++i)
            out.push_back(ids.elem(i, j));
    return out;
}

// every id must be in [0, limit)
inline void check_range(const std::vector<int>& ids, size_t limit, const char* what) {
    for (size_t k = 0; k < ids.size(); ++k)
        if (ids[k] < 0 || size_t(ids[k]) >= limit)
            throw std::invalid_argument(std::string(what) + " " + std::to_string(ids[k]) + " at position " +
                                        std::to_string(k) + " is outside 0 to " + std::to_string(limit - 1));
}

// One kernel source for nvcc and for clang -x hip (which defines __HIPCC__).
#if defined(CUDA) || (defined(ROCM_HIP) && defined(__HIPCC__))
#define JZ_ID_KERNELS 1
static_assert(sizeof(int) == sizeof(float) && sizeof(int32_t) == sizeof(int), "ids are stored as 32-bit patterns");
#if defined(CUDA)
template <class D> inline constexpr bool on_device = std::is_same_v<D, CUDAfloat>;
inline void launched() { CudaErrorCheck(cudaGetLastError()); }
inline void upload_words(float* device, const void* host, size_t count) {
    CudaErrorCheck(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
}
#else
template <class D> inline constexpr bool on_device = std::is_same_v<D, ROCMfloat>;
inline void launched() { RocmCheck(int(hipGetLastError())); }
inline void upload_words(float* device, const void* host, size_t count) {
    RocmCheck(RocmMemcpyH2D(device, reinterpret_cast<const float*>(host), count));
}
#endif
inline unsigned blocks(size_t count) { return unsigned(std::min(size_t(4096), (count - 1) / 256 + 1)); }

// int32 values in a pooled device block of D: bytes are copied, never converted
template <class D> Matrix<D> device_words(const std::vector<int>& values) {
    Matrix<D> block("ids", 1, values.size());
    upload_words(const_cast<float*>(reinterpret_cast<const float*>(block.data())), values.data(), values.size());
    return block;
}
template <class D> const int32_t* words(const Matrix<D>& block) { return reinterpret_cast<const int32_t*>(block.data()); }
template <class D> float* floats(const Matrix<D>& m) { return const_cast<float*>(reinterpret_cast<const float*>(m.data())); }
#else
template <class D> inline constexpr bool on_device = false;
#endif

}  // namespace token_detail
}  // namespace Juzhen
