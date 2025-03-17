#ifndef MPSWRAPPER_H
#define MPSWRAPPER_H

#include <cstddef>

void mpsInit();
void mpsDestroy();

float* mpsMalloc(size_t size);
void mpsFree(float* ptr);

void mpsGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transposeA, bool transposeB);
void mpsRandn(float* A, int N); // fill A with random numbers
void mpsFill(float* A, int N, float val); // fill A with zeros
void mpsSynchronize();

#endif
