#ifndef MPSWRAPPER_H
#define MPSWRAPPER_H

#include <cstddef>

void mpsInit();
// Name of the system default Metal device; usable before mpsInit().
const char* mpsDeviceName();
void mpsDestroy();

float* mpsMalloc(size_t size);
void mpsFree(float* ptr);
// GPU virtual address of the buffer behind a pointer from mpsMalloc (0 if unknown).
unsigned long long mpsGpuAddress(const float* ptr);

void mpsGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transposeA, bool transposeB);
// C = b*B + a*A(optionally transposed); B and C share a layout. In-place: B == C.
void mpsAdd(const float* A, const float* B, float* C, int rowA, int colA, bool transpose, float a, float b);
// C = A(optionally transposed) .* B; B and C share a layout. In-place: B == C.
void mpsProduct(const float* A, const float* B, float* C, int rowA, int colA, bool transpose);
void mpsAx_b(const float* x, float a, float b, float* y, int N);
// Elementwise ops write dst from src in one dispatch; in-place: src == dst.
void mpsTanh(const float* src, float* dst, int N);
void mpsdTanh(const float* src, float* dst, int N);
void mpsExp(const float* src, float* dst, int N);
void mpsLog(const float* src, float* dst, int N);
void mpsSquare(const float* src, float* dst, int N);
void mpsSqrt(const float* src, float* dst, int N);
void mpsRelu(const float* src, float* dst, int N);
void mpsdRelu(const float* src, float* dst, int N);
void mpsGemv(const float* A, const float* x, float* y, int rowA, int colA, bool transpose);
void mpsTopk(const float* A, float * B, float * C, int rowA, int colA, int k);
void mpsElemInv(const float* src, float* dst, int N, float l);
// Column-wise softmax of a (rows x cols) logical matrix. `transpose` is the
// source's physical-layout flag; dst is written plain row-major (rows x cols).
void mpsSoftmaxCol(const float* src, float* dst, int rows, int cols, bool transpose);
// Causal mask on a square (n x n) scores matrix [query row, key col]:
// entries with key col > query row are set to maskVal.
void mpsCausalMask(float* S, int n, bool transpose, float maskVal);

// ── batched multi-head attention ────────────────────────────────────────────
// C_i = alpha * op(A_i) * op(B_i) + beta * C_i for i in [0, batch): each
// operand is a contiguous batch of row-major matrices (A_i is rowA x colA at
// A + i*rowA*colA, etc.).
void mpsGemmBatched(const float* A, const float* B, float* C, int batch,
                    int rowA, int colA, int rowB, int colB,
                    bool transposeA, bool transposeB, float alpha, float beta = 0.0f);
// Repack a (d_k x batch*seq) row-major projection into batch*heads contiguous
// (seq x d_h) row-major matrices (one per head/sequence block), and back.
void mpsAttnPack(const float* src, float* dst, int d_h, int heads, int seq, int batch);
void mpsAttnUnpack(const float* src, float* dst, int d_h, int heads, int seq, int batch);
// Causal mask over a batch of (n x n) row-major scores matrices.
void mpsCausalMaskBatched(float* S, int n, int batch, float maskVal);
// Row-wise softmax over `rows` contiguous length-n rows (in-place ok: x == y).
void mpsSoftmaxRowsBatched(const float* x, float* y, int n, int rows);
// dS = A .* (dA - rowsum(A .* dA)) * scale, over `rows` contiguous length-n rows.
void mpsSoftmaxBackwardRowsBatched(const float* A, const float* dA, float* dS,
                                   int n, int rows, float scale);

// ── fused training kernels (row-major (dim, N) matrices) ───────────────────
// LayerNorm forward: writes y, xhat and per-column inv_std (length N).
void mpsLayerNormForward(const float* x, const float* gamma, const float* beta,
                         float* y, float* xhat, float* invStd, int dim, int N);
// LayerNorm input gradient from dy, gamma and the cached xhat / inv_std.
void mpsLayerNormBackward(const float* dy, const float* gamma, const float* xhat,
                          const float* invStd, float* dx, int dim, int N);
// LayerNorm tangent: dy = gamma * inv_std * (dx - mean(dx) - xhat * mean(xhat * dx)).
void mpsLayerNormJvp(const float* dx, const float* gamma, const float* xhat,
                     const float* invStd, float* dy, int dim, int N);
// gradient *= (activation > 0), elementwise over `total` elements.
void mpsReluBackwardInplace(const float* activation, float* gradient, int total);
// Maximum of each logical column of a (rows x cols) matrix (`transpose` is
// its physical-layout flag); dst is a plain (1 x cols) row.
void mpsColumnMax(const float* src, float* dst, int rows, int cols, bool transpose);
// dst = min(max(src, lo), hi) elementwise; in place: src == dst.
void mpsClamp(const float* src, float* dst, int N, float lo, float hi);
// dst = src > threshold ? 1 : 0 elementwise; in place: src == dst.
void mpsStep(const float* src, float* dst, int N, float threshold);
// y[i, c] += b[i] over a (rows x N) matrix (total = rows * N elements).
void mpsAddBias(float* y, const float* b, int N, int total);
// rand and randn draw from one MTGP32 uniform generator (randn via
// Box-Muller); mpsSetSeed restarts it, waiting for pending work first.
void mpsSetSeed(int seed);
void mpsRand(float* A, int N); // fill A with uniform numbers in [0, 1)
void mpsRandn(float* A, int N); // fill A with standard normal numbers
// dst (nr x nc, row-major) = src[rows[r], cols[c]] over src's logical
// (srcRows x srcCols) shape; srcTranspose is its physical-layout flag.
void mpsGather(const float* src, float* dst, const unsigned* rows, int nr,
               const unsigned* cols, int nc, int srcRows, int srcCols, bool srcTranspose);
void mpsFill(float* A, int N, float val); // fill A with zeros
void mpsCopyMatrixBlock(const float* src, float* dst,
						int srcRows, int srcCols, bool srcTranspose,
						int srcRowOffset, int srcColOffset,
						int copyRows, int copyCols,
						int dstRows, int dstCols, bool dstTranspose,
						int dstRowOffset, int dstColOffset);

// Convolution data-layout helpers on Metal buffers.
void mpsIm2col(const float* input, float* col,
			   int N, int C, int H, int W,
			   int kH, int kW,
			   int padH, int padW,
			   int strideH, int strideW,
			   int Hout, int Wout);

void mpsCol2im(const float* col, float* output,
			   int N, int C, int H, int W,
			   int kH, int kW,
			   int padH, int padW,
			   int strideH, int strideW,
			   int Hout, int Wout);

void mpsPackFeatureMap2D(const float* featureMap, float* packed,
						int N, int C, int P);

void mpsConv2dOutputAddBias(const float* y2d, const float* bias, float* output,
							int N, int Cout, int Hout, int Wout);

// One fused AdamW step: w, g, m, v are logical (rows x cols) matrices whose
// transposed flags are wt, gt, mt, vt.
void mpsAdamW(float* w, const float* g, float* m, float* v, int rows, int cols,
              bool wt, bool gt, bool mt, bool vt, float lr, float beta1, float beta2,
              float eps, float bc1, float bc2, float decay);

void mpsSynchronize();

#endif
