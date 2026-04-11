/**
 * @file hipkernels.cpp
 * @brief ROCm/HIP vertical-slice kernels: RNG + basic elementwise ops.
 */

#include "hipbackend.hpp"
#include "helper.hpp"

#ifdef ROCM_HIP

#if defined(__HIPCC__) && __has_include(<hip/hip_runtime.h>) && __has_include(<rocrand/rocrand.h>)
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#define JUZHEN_ROCM_KERNELS_AVAILABLE 1
#else
#define JUZHEN_ROCM_KERNELS_AVAILABLE 0
#endif

namespace Juzhen {

bool RocmRuntimeAvailable() {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return true;
#else
    return false;
#endif
}

#if JUZHEN_ROCM_KERNELS_AVAILABLE
namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void FillKernel(float* out, std::size_t n, float value) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = value;
}

__global__ void ExpKernel(float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(out[idx]);
}

__global__ void LogKernel(float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(out[idx]);
}

__global__ void AxpbyKernel(float* out, const float* a, const float* b, std::size_t n,
                            float s1, float s2) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = s1 * a[idx] + s2 * b[idx];
}

__global__ void AffineInplaceKernel(float* out, std::size_t n, float s1, float a) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = s1 * out[idx] + a;
}

__global__ void HadamardKernel(float* out, const float* a, const float* b, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void HadamardInplaceKernel(float* out, const float* b, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] *= b[idx];
}

__global__ void ElemInvInplaceKernel(float* out, std::size_t n, float l) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = l / out[idx];
}

__global__ void CopyKernel(float* out, const float* in, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void TanhKernel(float* out, const float* in, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

__global__ void TanhInplaceKernel(float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(out[idx]);
}

__global__ void DTanhKernel(float* out, const float* in, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(in[idx]);
        out[idx] = 1.0f - t * t;
    }
}

__global__ void DTanhInplaceKernel(float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(out[idx]);
        out[idx] = 1.0f - t * t;
    }
}

__global__ void SquareKernel(float* out, const float* in, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * in[idx];
}

__global__ void SquareInplaceKernel(float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = out[idx] * out[idx];
}

__global__ void StackCopyColsKernel(float* dst,
                                    std::size_t dst_numrow,
                                    std::size_t dst_col_offset,
                                    const float* src,
                                    std::size_t src_numrow,
                                    std::size_t src_numcol,
                                    bool src_transpose) {
    const std::size_t logical_rows = src_transpose ? src_numcol : src_numrow;
    const std::size_t logical_cols = src_transpose ? src_numrow : src_numcol;
    const std::size_t total = logical_rows * logical_cols;

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const std::size_t r = idx % logical_rows;
    const std::size_t c = idx / logical_rows;

    const std::size_t src_idx = src_transpose ? (r * src_numrow + c) : (c * src_numrow + r);
    const std::size_t dst_idx = (dst_col_offset + c) * dst_numrow + r;
    dst[dst_idx] = src[src_idx];
}

__global__ void StackCopyRowsKernel(float* dst,
                                    std::size_t dst_numrow,
                                    std::size_t dst_row_offset,
                                    const float* src,
                                    std::size_t src_numrow,
                                    std::size_t src_numcol,
                                    bool src_transpose) {
    const std::size_t logical_rows = src_transpose ? src_numcol : src_numrow;
    const std::size_t logical_cols = src_transpose ? src_numrow : src_numcol;
    const std::size_t total = logical_rows * logical_cols;

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const std::size_t r = idx % logical_rows;
    const std::size_t c = idx / logical_rows;

    const std::size_t src_idx = src_transpose ? (r * src_numrow + c) : (c * src_numrow + r);
    const std::size_t dst_idx = c * dst_numrow + (dst_row_offset + r);
    dst[dst_idx] = src[src_idx];
}

__global__ void SliceExtractKernel(float* dst,
                                   std::size_t dst_numrow,
                                   const float* src,
                                   std::size_t src_numrow,
                                   bool src_transpose,
                                   std::size_t rstart,
                                   std::size_t cstart,
                                   std::size_t rows,
                                   std::size_t cols) {
    const std::size_t total = rows * cols;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const std::size_t r = idx % rows;
    const std::size_t c = idx / rows;
    const std::size_t sr = rstart + r;
    const std::size_t sc = cstart + c;

    const std::size_t src_idx = src_transpose ? (sr * src_numrow + sc) : (sc * src_numrow + sr);
    const std::size_t dst_idx = c * dst_numrow + r;
    dst[dst_idx] = src[src_idx];
}

__global__ void SliceAssignKernel(float* dst,
                                  std::size_t dst_numrow,
                                  bool dst_transpose,
                                  std::size_t rstart,
                                  std::size_t cstart,
                                  const float* src,
                                  std::size_t src_numrow,
                                  bool src_transpose,
                                  std::size_t rows,
                                  std::size_t cols) {
    const std::size_t total = rows * cols;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const std::size_t r = idx % rows;
    const std::size_t c = idx / rows;
    const std::size_t dr = rstart + r;
    const std::size_t dc = cstart + c;

    const std::size_t src_idx = src_transpose ? (r * src_numrow + c) : (c * src_numrow + r);
    const std::size_t dst_idx = dst_transpose ? (dr * dst_numrow + dc) : (dc * dst_numrow + dr);
    dst[dst_idx] = src[src_idx];
}

// ── Convolution kernels (im2col / col2im / reshape) ─────────────────────

/**
 * im2col: input(C_in*H_in*W_in, N) → col(K, P*N)
 * K = C_in*kH*kW, P = H_out*W_out
 * col is column-major (K, P*N): element (k, c) at offset c*K + k.
 */
__global__ void Im2colKernel(const float* input, float* col,
                             int C_in, int H_in, int W_in,
                             int kH, int kW,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w,
                             int H_out, int W_out,
                             int N, int K, int P) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(K) * P * N;
    if (idx >= total) return;

    // col is column-major (K, P*N): row = k, col = c
    const int k = static_cast<int>(idx % K);
    const int c = static_cast<int>(idx / K);
    const int n = c / P;
    const int p = c % P;

    const int oh = p / W_out;
    const int ow = p % W_out;
    const int ci = k / (kH * kW);
    const int rem = k % (kH * kW);
    const int kh_i = rem / kW;
    const int kw_i = rem % kW;

    const int ih = oh * stride_h - pad_h + kh_i;
    const int iw = ow * stride_w - pad_w + kw_i;

    float val = 0.0f;
    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
        // input is column-major (C_in*H_in*W_in, N)
        val = input[static_cast<std::size_t>(n) * (C_in * H_in * W_in)
                   + static_cast<std::size_t>(ci) * (H_in * W_in)
                   + ih * W_in + iw];
    }
    col[idx] = val;
}

/**
 * col2im: col(K, P*N) → dx(C_in*H_in*W_in, N), gathering contributions.
 * Each thread computes one element of dx.
 */
__global__ void Col2imKernel(const float* col, float* dx,
                             int C_in, int H_in, int W_in,
                             int kH, int kW,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w,
                             int H_out, int W_out,
                             int N, int K, int P) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t input_size = static_cast<std::size_t>(C_in) * H_in * W_in;
    const std::size_t total = input_size * N;
    if (idx >= total) return;

    const int n  = static_cast<int>(idx / input_size);
    const int r  = static_cast<int>(idx % input_size);
    const int ci = r / (H_in * W_in);
    const int rem = r % (H_in * W_in);
    const int ih = rem / W_in;
    const int iw = rem % W_in;

    float val = 0.0f;
    // Gather: for each (kh_i, kw_i), find the (oh, ow) that maps here
    for (int kh_i = 0; kh_i < kH; ++kh_i) {
        int oh_num = ih + pad_h - kh_i;
        if (oh_num < 0 || oh_num % stride_h != 0) continue;
        int oh = oh_num / stride_h;
        if (oh >= H_out) continue;

        for (int kw_i = 0; kw_i < kW; ++kw_i) {
            int ow_num = iw + pad_w - kw_i;
            if (ow_num < 0 || ow_num % stride_w != 0) continue;
            int ow = ow_num / stride_w;
            if (ow >= W_out) continue;

            int k = ci * kH * kW + kh_i * kW + kw_i;
            int p = oh * W_out + ow;
            int c = n * P + p;
            // col is column-major (K, P*N)
            val += col[static_cast<std::size_t>(c) * K + k];
        }
    }
    dx[idx] = val;
}

/**
 * Forward reshape + bias: y2d(C_out, P*N) → out(C_out*P, N) + bias[co]
 * Both y2d and out are column-major.
 */
__global__ void ConvForwardReshapeBiasKernel(const float* y2d, float* out, const float* bias,
                                             int C_out, int P, int N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(C_out) * P * N;
    if (idx >= total) return;

    // out is column-major (C_out*P, N)
    const int outCols = C_out * P;
    const int n  = static_cast<int>(idx / outCols);
    const int r  = static_cast<int>(idx % outCols);
    const int co = r / P;
    const int p  = r % P;

    // y2d is column-major (C_out, P*N): element at row co, col (n*P + p)
    const std::size_t src = static_cast<std::size_t>(n * P + p) * C_out + co;
    out[idx] = y2d[src] + bias[co];
}

/**
 * Backward reshape: t(C_out*P, N) → t2d(C_out, P*N)
 * Inverse of ConvForwardReshapeBiasKernel (without bias).
 */
__global__ void ConvBackwardReshapeKernel(const float* t, float* t2d,
                                          int C_out, int P, int N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(C_out) * P * N;
    if (idx >= total) return;

    // t2d is column-major (C_out, P*N): element at (co, col)
    const int co  = static_cast<int>(idx % C_out);
    const int col = static_cast<int>(idx / C_out);
    const int n   = col / P;
    const int p   = col % P;

    // t is column-major (C_out*P, N): element at row (co*P + p), col n
    const std::size_t src = static_cast<std::size_t>(n) * (C_out * P) + co * P + p;
    t2d[idx] = t[src];
}

// ── Transposed convolution kernels ──────────────────────────────────────

/**
 * ConvTrans forward scatter: patches(C_out*kH*kW, N*P_in) → out(C_out*H_out*W_out, N)
 * with bias. P_in = H_in*W_in. Each output element gathers from all input patches.
 */
__global__ void ConvTransScatterKernel(const float* patches, float* out, const float* bias,
                                       int C_out, int H_out, int W_out,
                                       int H_in, int W_in,
                                       int kH, int kW,
                                       int pad_h, int pad_w,
                                       int stride_h, int stride_w,
                                       int N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int out_size = C_out * H_out * W_out;
    const std::size_t total = static_cast<std::size_t>(out_size) * N;
    if (idx >= total) return;

    const int n  = static_cast<int>(idx / out_size);
    const int r  = static_cast<int>(idx % out_size);
    const int co = r / (H_out * W_out);
    const int rem = r % (H_out * W_out);
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    float val = bias[co];
    const int P_in = H_in * W_in;
    const int patch_rows = C_out * kH * kW;

    // Gather from all input positions (ih, iw) that contribute to (oh, ow)
    // Relationship: oh = ih * stride_h - pad_h + kh  =>  kh = oh - ih*stride_h + pad_h
    for (int ih = 0; ih < H_in; ++ih) {
        for (int iw = 0; iw < W_in; ++iw) {
            int kh_i = oh - ih * stride_h + pad_h;
            int kw_i = ow - iw * stride_w + pad_w;
            if (kh_i < 0 || kh_i >= kH || kw_i < 0 || kw_i >= kW) continue;
            int patch_row = co * kH * kW + kh_i * kW + kw_i;
            int patch_col = n * P_in + ih * W_in + iw;
            // patches is column-major (patch_rows, N*P_in)
            val += patches[static_cast<std::size_t>(patch_col) * patch_rows + patch_row];
        }
    }
    out[idx] = val;
}

/**
 * ConvTrans backward gather: t(C_out*H_out*W_out, N) → tp(C_out*kH*kW, N*P_in)
 * Inverse of scatter: for each element in tp, gather from t.
 */
__global__ void ConvTransGatherKernel(const float* t, float* tp,
                                      int C_out, int H_out, int W_out,
                                      int H_in, int W_in,
                                      int kH, int kW,
                                      int pad_h, int pad_w,
                                      int stride_h, int stride_w,
                                      int N) {
    const int P_in = H_in * W_in;
    const int patch_rows = C_out * kH * kW;
    const std::size_t total = static_cast<std::size_t>(patch_rows) * N * P_in;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // tp is column-major (patch_rows, N*P_in)
    const int pr  = static_cast<int>(idx % patch_rows);
    const int pc  = static_cast<int>(idx / patch_rows);
    const int n   = pc / P_in;
    const int pin = pc % P_in;
    const int ih  = pin / W_in;
    const int iw  = pin % W_in;

    const int co    = pr / (kH * kW);
    const int krem  = pr % (kH * kW);
    const int kh_i  = krem / kW;
    const int kw_i  = krem % kW;

    const int oh = ih * stride_h - pad_h + kh_i;
    const int ow = iw * stride_w - pad_w + kw_i;

    float val = 0.0f;
    if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
        // t is column-major (C_out*H_out*W_out, N)
        const int out_size = C_out * H_out * W_out;
        val = t[static_cast<std::size_t>(n) * out_size + co * H_out * W_out + oh * W_out + ow];
    }
    tp[idx] = val;
}

/**
 * Reshape for convtrans: bidirectional between (C, spatial*N) and (C*spatial, N).
 * direction=0: (C*spatial, N) → (C, spatial*N)  [pack_input / unpack_dx]
 * direction=1: (C, spatial*N) → (C*spatial, N)  [unused but symmetric]
 */
__global__ void ConvTransReshapeKernel(const float* src, float* dst,
                                       int C, int spatial, int N, int direction) {
    const std::size_t total = static_cast<std::size_t>(C) * spatial * N;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    if (direction == 0) {
        // src is (C*spatial, N), dst is (C, spatial*N)
        const int CS = C * spatial;
        const int n = static_cast<int>(idx / CS);
        const int r = static_cast<int>(idx % CS);
        const int c_ch = r / spatial;
        const int s = r % spatial;
        // dst[col*C + c_ch] where col = n*spatial + s
        dst[static_cast<std::size_t>(n * spatial + s) * C + c_ch] = src[idx];
    } else {
        // src is (C, spatial*N), dst is (C*spatial, N)
        const int c_ch = static_cast<int>(idx % C);
        const int col = static_cast<int>(idx / C);
        const int n = col / spatial;
        const int s = col % spatial;
        dst[static_cast<std::size_t>(n) * C * spatial + c_ch * spatial + s] = src[idx];
    }
}

inline int BlocksFor(std::size_t n) {
    return static_cast<int>((n + static_cast<std::size_t>(kThreadsPerBlock) - 1) /
                            static_cast<std::size_t>(kThreadsPerBlock));
}

inline int HipToInt(hipError_t status) {
    return status == hipSuccess ? 0 : static_cast<int>(status);
}

inline int RocRandToInt(rocrand_status status) {
    return status == ROCRAND_STATUS_SUCCESS ? 0 : static_cast<int>(status);
}

inline rocrand_generator& GlobalRocRandGenerator() {
    static rocrand_generator gen = nullptr;
    static bool initialized = false;
    if (!initialized) {
        auto st = rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
        if (st != ROCRAND_STATUS_SUCCESS) {
            LOG_ERROR("rocrand_create_generator failed with code {}", RocRandToInt(st));
            return gen;
        }
        initialized = true;
    }
    return gen;
}

inline int EnsureRocRandSeed(rocrand_generator gen, std::uint64_t seed) {
    static bool seeded = false;
    static std::uint64_t currentSeed = 0;

    // seed=0 means keep advancing current generator state without reseeding.
    if (seed == 0) {
        if (!seeded) {
            constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;
            auto st = rocrand_set_seed(gen, kDefaultSeed);
            if (st != ROCRAND_STATUS_SUCCESS) return RocRandToInt(st);
            seeded = true;
            currentSeed = kDefaultSeed;
        }
        return 0;
    }

    if (!seeded || currentSeed != seed) {
        auto st = rocrand_set_seed(gen, seed);
        if (st != ROCRAND_STATUS_SUCCESS) return RocRandToInt(st);
        seeded = true;
        currentSeed = seed;
    }
    return 0;
}

}  // namespace
#endif

int RocmMalloc(float** ptr, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    if (ptr == nullptr) return -1;
    return HipToInt(hipMalloc(reinterpret_cast<void**>(ptr), count * sizeof(float)));
#else
    (void)ptr;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmMalloc is unavailable.");
    return -1;
#endif
}

int RocmFree(float* ptr) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return HipToInt(hipFree(ptr));
#else
    (void)ptr;
    LOG_ERROR("ROCm runtime headers not found; RocmFree is unavailable.");
    return -1;
#endif
}

int RocmMemcpyH2D(float* dst_device, const float* src_host, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return HipToInt(hipMemcpy(dst_device, src_host, count * sizeof(float), hipMemcpyHostToDevice));
#else
    (void)dst_device;
    (void)src_host;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmMemcpyH2D is unavailable.");
    return -1;
#endif
}

int RocmMemcpyD2H(float* dst_host, const float* src_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return HipToInt(hipMemcpy(dst_host, src_device, count * sizeof(float), hipMemcpyDeviceToHost));
#else
    (void)dst_host;
    (void)src_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmMemcpyD2H is unavailable.");
    return -1;
#endif
}

int RocmMemcpyD2D(float* dst_device, const float* src_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return HipToInt(hipMemcpy(dst_device, src_device, count * sizeof(float), hipMemcpyDeviceToDevice));
#else
    (void)dst_device;
    (void)src_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmMemcpyD2D is unavailable.");
    return -1;
#endif
}

int RocmRandUniform(float* dst_device, std::size_t count, std::uint64_t seed) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    rocrand_generator gen = GlobalRocRandGenerator();
    if (gen == nullptr) return -1;
    int seedRc = EnsureRocRandSeed(gen, seed);
    if (seedRc != 0) return seedRc;

    auto st = rocrand_generate_uniform(gen, dst_device, count);
    return RocRandToInt(st);
#else
    (void)dst_device;
    (void)count;
    (void)seed;
    LOG_ERROR("rocRAND headers not found; RocmRandUniform is unavailable.");
    return -1;
#endif
}

int RocmRandNormal(float* dst_device, std::size_t count, float mean, float stddev,
                   std::uint64_t seed) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    rocrand_generator gen = GlobalRocRandGenerator();
    if (gen == nullptr) return -1;
    int seedRc = EnsureRocRandSeed(gen, seed);
    if (seedRc != 0) return seedRc;

    auto st = rocrand_generate_normal(gen, dst_device, count, mean, stddev);
    return RocRandToInt(st);
#else
    (void)dst_device;
    (void)count;
    (void)mean;
    (void)stddev;
    (void)seed;
    LOG_ERROR("rocRAND headers not found; RocmRandNormal is unavailable.");
    return -1;
#endif
}

int RocmFill(float* dst_device, std::size_t count, float value) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(FillKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, count, value);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)count;
    (void)value;
    LOG_ERROR("ROCm runtime headers not found; RocmFill is unavailable.");
    return -1;
#endif
}

int RocmExpInplace(float* dst_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(ExpKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmExpInplace is unavailable.");
    return -1;
#endif
}

int RocmLogInplace(float* dst_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(LogKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmLogInplace is unavailable.");
    return -1;
#endif
}

int RocmAxpby(float* out_device, const float* a_device, const float* b_device,
              std::size_t count, float s1, float s2) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(AxpbyKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       out_device, a_device, b_device, count, s1, s2);
    return HipToInt(hipGetLastError());
#else
    (void)out_device;
    (void)a_device;
    (void)b_device;
    (void)count;
    (void)s1;
    (void)s2;
    LOG_ERROR("ROCm runtime headers not found; RocmAxpby is unavailable.");
    return -1;
#endif
}

int RocmAffineInplace(float* inout_device, std::size_t count, float s1, float a) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(AffineInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, count, s1, a);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)count;
    (void)s1;
    (void)a;
    LOG_ERROR("ROCm runtime headers not found; RocmAffineInplace is unavailable.");
    return -1;
#endif
}

int RocmHadamard(float* out_device, const float* a_device, const float* b_device,
                 std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(HadamardKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       out_device, a_device, b_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)out_device;
    (void)a_device;
    (void)b_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmHadamard is unavailable.");
    return -1;
#endif
}

int RocmHadamardInplace(float* inout_device, const float* other_device,
                        std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(HadamardInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, other_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)other_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmHadamardInplace is unavailable.");
    return -1;
#endif
}

int RocmElemInvInplace(float* inout_device, std::size_t count, float l) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(ElemInvInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, count, l);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)count;
    (void)l;
    LOG_ERROR("ROCm runtime headers not found; RocmElemInvInplace is unavailable.");
    return -1;
#endif
}

int RocmCopy(float* dst_device, const float* src_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    return HipToInt(hipMemcpy(dst_device, src_device, count * sizeof(float), hipMemcpyDeviceToDevice));
#else
    (void)dst_device;
    (void)src_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmCopy is unavailable.");
    return -1;
#endif
}

int RocmSquare(float* out_device, const float* in_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(SquareKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       out_device, in_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)out_device;
    (void)in_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmSquare is unavailable.");
    return -1;
#endif
}

int RocmSquareInplace(float* inout_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(SquareInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmSquareInplace is unavailable.");
    return -1;
#endif
}

int RocmTanh(float* out_device, const float* in_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(TanhKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       out_device, in_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)out_device;
    (void)in_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmTanh is unavailable.");
    return -1;
#endif
}

int RocmTanhInplace(float* inout_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(TanhInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmTanhInplace is unavailable.");
    return -1;
#endif
}

int RocmDTanh(float* out_device, const float* in_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(DTanhKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       out_device, in_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)out_device;
    (void)in_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmDTanh is unavailable.");
    return -1;
#endif
}

int RocmDTanhInplace(float* inout_device, std::size_t count) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    hipLaunchKernelGGL(DTanhInplaceKernel, dim3(BlocksFor(count)), dim3(kThreadsPerBlock), 0, 0,
                       inout_device, count);
    return HipToInt(hipGetLastError());
#else
    (void)inout_device;
    (void)count;
    LOG_ERROR("ROCm runtime headers not found; RocmDTanhInplace is unavailable.");
    return -1;
#endif
}

int RocmStackCopyCols(float* dst_device,
                      std::size_t dst_numrow,
                      std::size_t dst_col_offset,
                      const float* src_device,
                      std::size_t src_numrow,
                      std::size_t src_numcol,
                      bool src_transpose) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t logical_rows = src_transpose ? src_numcol : src_numrow;
    const std::size_t logical_cols = src_transpose ? src_numrow : src_numcol;
    const std::size_t total = logical_rows * logical_cols;
    hipLaunchKernelGGL(StackCopyColsKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, dst_numrow, dst_col_offset,
                       src_device, src_numrow, src_numcol, src_transpose);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)dst_numrow;
    (void)dst_col_offset;
    (void)src_device;
    (void)src_numrow;
    (void)src_numcol;
    (void)src_transpose;
    LOG_ERROR("ROCm runtime headers not found; RocmStackCopyCols is unavailable.");
    return -1;
#endif
}

int RocmStackCopyRows(float* dst_device,
                      std::size_t dst_numrow,
                      std::size_t dst_row_offset,
                      const float* src_device,
                      std::size_t src_numrow,
                      std::size_t src_numcol,
                      bool src_transpose) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t logical_rows = src_transpose ? src_numcol : src_numrow;
    const std::size_t logical_cols = src_transpose ? src_numrow : src_numcol;
    const std::size_t total = logical_rows * logical_cols;
    hipLaunchKernelGGL(StackCopyRowsKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, dst_numrow, dst_row_offset,
                       src_device, src_numrow, src_numcol, src_transpose);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)dst_numrow;
    (void)dst_row_offset;
    (void)src_device;
    (void)src_numrow;
    (void)src_numcol;
    (void)src_transpose;
    LOG_ERROR("ROCm runtime headers not found; RocmStackCopyRows is unavailable.");
    return -1;
#endif
}

int RocmSliceExtract(float* dst_device,
                     std::size_t dst_numrow,
                     const float* src_device,
                     std::size_t src_numrow,
                     bool src_transpose,
                     std::size_t rstart,
                     std::size_t cstart,
                     std::size_t rows,
                     std::size_t cols) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = rows * cols;
    hipLaunchKernelGGL(SliceExtractKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, dst_numrow,
                       src_device, src_numrow, src_transpose,
                       rstart, cstart, rows, cols);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)dst_numrow;
    (void)src_device;
    (void)src_numrow;
    (void)src_transpose;
    (void)rstart;
    (void)cstart;
    (void)rows;
    (void)cols;
    LOG_ERROR("ROCm runtime headers not found; RocmSliceExtract is unavailable.");
    return -1;
#endif
}

int RocmSliceAssign(float* dst_device,
                    std::size_t dst_numrow,
                    bool dst_transpose,
                    std::size_t rstart,
                    std::size_t cstart,
                    const float* src_device,
                    std::size_t src_numrow,
                    bool src_transpose,
                    std::size_t rows,
                    std::size_t cols) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = rows * cols;
    hipLaunchKernelGGL(SliceAssignKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       dst_device, dst_numrow, dst_transpose,
                       rstart, cstart,
                       src_device, src_numrow, src_transpose,
                       rows, cols);
    return HipToInt(hipGetLastError());
#else
    (void)dst_device;
    (void)dst_numrow;
    (void)dst_transpose;
    (void)rstart;
    (void)cstart;
    (void)src_device;
    (void)src_numrow;
    (void)src_transpose;
    (void)rows;
    (void)cols;
    LOG_ERROR("ROCm runtime headers not found; RocmSliceAssign is unavailable.");
    return -1;
#endif
}

// ── Convolution wrapper functions ───────────────────────────────────────

int RocmIm2col(const float* input, float* col,
               int C_in, int H_in, int W_in,
               int kH, int kW,
               int pad_h, int pad_w,
               int stride_h, int stride_w,
               int H_out, int W_out, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const int K = C_in * kH * kW;
    const int P = H_out * W_out;
    const std::size_t total = static_cast<std::size_t>(K) * P * N;
    hipLaunchKernelGGL(Im2colKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       input, col, C_in, H_in, W_in, kH, kW,
                       pad_h, pad_w, stride_h, stride_w, H_out, W_out, N, K, P);
    return HipToInt(hipGetLastError());
#else
    (void)input; (void)col; (void)C_in; (void)H_in; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w;
    (void)stride_h; (void)stride_w; (void)H_out; (void)W_out; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmIm2col is unavailable.");
    return -1;
#endif
}

int RocmCol2im(const float* col, float* dx,
               int C_in, int H_in, int W_in,
               int kH, int kW,
               int pad_h, int pad_w,
               int stride_h, int stride_w,
               int H_out, int W_out, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const int K = C_in * kH * kW;
    const int P = H_out * W_out;
    const std::size_t input_size = static_cast<std::size_t>(C_in) * H_in * W_in;
    const std::size_t total = input_size * N;
    hipLaunchKernelGGL(Col2imKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       col, dx, C_in, H_in, W_in, kH, kW,
                       pad_h, pad_w, stride_h, stride_w, H_out, W_out, N, K, P);
    return HipToInt(hipGetLastError());
#else
    (void)col; (void)dx; (void)C_in; (void)H_in; (void)W_in;
    (void)kH; (void)kW; (void)pad_h; (void)pad_w;
    (void)stride_h; (void)stride_w; (void)H_out; (void)W_out; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmCol2im is unavailable.");
    return -1;
#endif
}

int RocmConvForwardReshapeBias(const float* y2d, float* out, const float* bias,
                                int C_out, int P, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = static_cast<std::size_t>(C_out) * P * N;
    hipLaunchKernelGGL(ConvForwardReshapeBiasKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       y2d, out, bias, C_out, P, N);
    return HipToInt(hipGetLastError());
#else
    (void)y2d; (void)out; (void)bias; (void)C_out; (void)P; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmConvForwardReshapeBias is unavailable.");
    return -1;
#endif
}

int RocmConvBackwardReshape(const float* t, float* t2d,
                             int C_out, int P, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = static_cast<std::size_t>(C_out) * P * N;
    hipLaunchKernelGGL(ConvBackwardReshapeKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       t, t2d, C_out, P, N);
    return HipToInt(hipGetLastError());
#else
    (void)t; (void)t2d; (void)C_out; (void)P; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmConvBackwardReshape is unavailable.");
    return -1;
#endif
}

int RocmConvTransScatter(const float* patches, float* out, const float* bias,
                          int C_out, int H_out, int W_out,
                          int H_in, int W_in,
                          int kH, int kW,
                          int pad_h, int pad_w,
                          int stride_h, int stride_w, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = static_cast<std::size_t>(C_out) * H_out * W_out * N;
    hipLaunchKernelGGL(ConvTransScatterKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       patches, out, bias,
                       C_out, H_out, W_out, H_in, W_in,
                       kH, kW, pad_h, pad_w, stride_h, stride_w, N);
    return HipToInt(hipGetLastError());
#else
    (void)patches; (void)out; (void)bias; (void)C_out; (void)H_out; (void)W_out;
    (void)H_in; (void)W_in; (void)kH; (void)kW;
    (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmConvTransScatter is unavailable.");
    return -1;
#endif
}

int RocmConvTransGather(const float* t, float* tp,
                         int C_out, int H_out, int W_out,
                         int H_in, int W_in,
                         int kH, int kW,
                         int pad_h, int pad_w,
                         int stride_h, int stride_w, int N) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const int P_in = H_in * W_in;
    const int patch_rows = C_out * kH * kW;
    const std::size_t total = static_cast<std::size_t>(patch_rows) * N * P_in;
    hipLaunchKernelGGL(ConvTransGatherKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       t, tp, C_out, H_out, W_out, H_in, W_in,
                       kH, kW, pad_h, pad_w, stride_h, stride_w, N);
    return HipToInt(hipGetLastError());
#else
    (void)t; (void)tp; (void)C_out; (void)H_out; (void)W_out;
    (void)H_in; (void)W_in; (void)kH; (void)kW;
    (void)pad_h; (void)pad_w; (void)stride_h; (void)stride_w; (void)N;
    LOG_ERROR("ROCm runtime headers not found; RocmConvTransGather is unavailable.");
    return -1;
#endif
}

int RocmConvTransReshape(const float* src, float* dst,
                          int C, int spatial, int N, int direction) {
#if JUZHEN_ROCM_KERNELS_AVAILABLE
    const std::size_t total = static_cast<std::size_t>(C) * spatial * N;
    hipLaunchKernelGGL(ConvTransReshapeKernel, dim3(BlocksFor(total)), dim3(kThreadsPerBlock), 0, 0,
                       src, dst, C, spatial, N, direction);
    return HipToInt(hipGetLastError());
#else
    (void)src; (void)dst; (void)C; (void)spatial; (void)N; (void)direction;
    LOG_ERROR("ROCm runtime headers not found; RocmConvTransReshape is unavailable.");
    return -1;
#endif
}

}  // namespace Juzhen

#endif
