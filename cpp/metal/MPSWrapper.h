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
void mpsExp(float* A, int N);
void mpsLog(float* A, int N);
void mpsGemv(const float* A, const float* x, float* y, int rowA, int colA, bool transpose);
void mpsTopk(const float* A, int * B, float * C, int rowA, int colA, int k);
void mpsElemInv(float* A, int N, float l);
void mpsRandn(float* A, int N); // fill A with random numbers
void mpsFill(float* A, int N, float val); // fill A with zeros
void mpsSynchronize();

#endif
