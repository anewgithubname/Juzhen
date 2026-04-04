#ifndef MPSWRAPPER_H
#define MPSWRAPPER_H

#include <cstddef>

void mpsInit();
void mpsDestroy();

float* mpsMalloc(size_t size);
void mpsFree(float* ptr);

void mpsGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transposeA, bool transposeB);
void mpsAdd(const float* A, float* B, int rowA, int colA, bool transpose, float a, float b);
void mpsProduct(const float* A, float* B, int rowA, int colA, bool transpose);
void mpsAx_b(const float* x, float a, float b, float* y, int N);
void mpsTanh(float* A, int N);
void mpsdTanh(float* A, int N);
void mpsExp(float* A, int N);
void mpsLog(float* A, int N);
void mpsSquare(float* A, int N);
void mpsSqrt(float* A, int N);
void mpsRelu(float* A, int N);
void mpsdRelu(float* A, int N);
void mpsGemv(const float* A, const float* x, float* y, int rowA, int colA, bool transpose);
void mpsTopk(const float* A, float * B, float * C, int rowA, int colA, int k);
void mpsElemInv(float* A, int N, float l);
void mpsRand(float* A, int N); // fill A with random numbers in [0, 1)
void mpsRandn(float* A, int N); // fill A with random numbers
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

void mpsSynchronize();

#endif
