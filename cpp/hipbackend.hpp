#ifndef HIPBACKEND_HPP
#define HIPBACKEND_HPP

#include <cstddef>
#include <cstdint>

namespace Juzhen {

// Phase-2 ROCm vertical slice API: memory, RNG, GEMM, and elementwise ops.
bool RocmRuntimeAvailable();

int RocmMalloc(float** ptr, std::size_t count);
int RocmFree(float* ptr);
int RocmMemcpyH2D(float* dst_device, const float* src_host, std::size_t count);
int RocmMemcpyD2H(float* dst_host, const float* src_device, std::size_t count);
int RocmMemcpyD2D(float* dst_device, const float* src_device, std::size_t count);

int RocmRandUniform(float* dst_device, std::size_t count, std::uint64_t seed);
int RocmRandNormal(float* dst_device, std::size_t count, float mean, float stddev,
                   std::uint64_t seed);

int RocmFill(float* dst_device, std::size_t count, float value);
int RocmExpInplace(float* dst_device, std::size_t count);
int RocmLogInplace(float* dst_device, std::size_t count);
int RocmAxpby(float* out_device, const float* a_device, const float* b_device,
              std::size_t count, float s1, float s2);
int RocmAffineInplace(float* inout_device, std::size_t count, float s1, float a);
int RocmHadamard(float* out_device, const float* a_device, const float* b_device,
                 std::size_t count);
int RocmHadamardInplace(float* inout_device, const float* other_device,
                        std::size_t count);
int RocmElemInvInplace(float* inout_device, std::size_t count, float l);
int RocmCopy(float* dst_device, const float* src_device, std::size_t count);
int RocmSquare(float* out_device, const float* in_device, std::size_t count);
int RocmSquareInplace(float* inout_device, std::size_t count);
int RocmTanh(float* out_device, const float* in_device, std::size_t count);
int RocmTanhInplace(float* inout_device, std::size_t count);
int RocmDTanh(float* out_device, const float* in_device, std::size_t count);
int RocmDTanhInplace(float* inout_device, std::size_t count);
int RocmStackCopyCols(float* dst_device,
                      std::size_t dst_numrow,
                      std::size_t dst_col_offset,
                      const float* src_device,
                      std::size_t src_numrow,
                      std::size_t src_numcol,
                      bool src_transpose);
int RocmStackCopyRows(float* dst_device,
                      std::size_t dst_numrow,
                      std::size_t dst_row_offset,
                      const float* src_device,
                      std::size_t src_numrow,
                      std::size_t src_numcol,
                      bool src_transpose);
int RocmSliceExtract(float* dst_device,
                     std::size_t dst_numrow,
                     const float* src_device,
                     std::size_t src_numrow,
                     bool src_transpose,
                     std::size_t rstart,
                     std::size_t cstart,
                     std::size_t rows,
                     std::size_t cols);
int RocmSliceAssign(float* dst_device,
                    std::size_t dst_numrow,
                    bool dst_transpose,
                    std::size_t rstart,
                    std::size_t cstart,
                    const float* src_device,
                    std::size_t src_numrow,
                    bool src_transpose,
                    std::size_t rows,
                    std::size_t cols);

// General column-major GEMM:
// C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C(m x n)
// op(X) is transpose when transX=true.
int RocmGemm(const float* A_device, const float* B_device, float* C_device,
             int m, int n, int k,
             bool transA, bool transB,
             int lda, int ldb, int ldc,
             float alpha = 1.0f, float beta = 0.0f);

// y = alpha * op(A) * x + beta * y
int RocmGemv(const float* A_device, const float* x_device, float* y_device,
             int m, int n,
             bool transA,
             int lda,
             int incx = 1,
             int incy = 1,
             float alpha = 1.0f,
             float beta = 0.0f);

// C = alpha * A(m x k) * B(k x n) + beta * C(m x n)
int RocmGemmNN(const float* A_device, const float* B_device, float* C_device,
               int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);

// ── Convolution primitives ──────────────────────────────────────────────

int RocmIm2col(const float* input, float* col,
               int C_in, int H_in, int W_in,
               int kH, int kW,
               int pad_h, int pad_w,
               int stride_h, int stride_w,
               int H_out, int W_out, int N);

int RocmCol2im(const float* col, float* dx,
               int C_in, int H_in, int W_in,
               int kH, int kW,
               int pad_h, int pad_w,
               int stride_h, int stride_w,
               int H_out, int W_out, int N);

int RocmConvForwardReshapeBias(const float* y2d, float* out, const float* bias,
                                int C_out, int P, int N);

int RocmConvBackwardReshape(const float* t, float* t2d,
                             int C_out, int P, int N);

int RocmConvTransScatter(const float* patches, float* out, const float* bias,
                          int C_out, int H_out, int W_out,
                          int H_in, int W_in,
                          int kH, int kW,
                          int pad_h, int pad_w,
                          int stride_h, int stride_w, int N);

int RocmConvTransGather(const float* t, float* tp,
                         int C_out, int H_out, int W_out,
                         int H_in, int W_in,
                         int kH, int kW,
                         int pad_h, int pad_w,
                         int stride_h, int stride_w, int N);

// direction=0: (C*spatial, N) → (C, spatial*N)
// direction=1: (C, spatial*N) → (C*spatial, N)
int RocmConvTransReshape(const float* src, float* dst,
                          int C, int spatial, int N, int direction);

}  // namespace Juzhen

#endif
