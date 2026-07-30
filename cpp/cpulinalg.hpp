/**
 * @file cpulinalg.hpp
 * @brief Handwritten column-major gemm/gemv kernels.
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * Used in place of an external BLAS when the library is built with
 * JUZHEN_NO_BLAS (CMake option BLAS_FREE). Self-contained on purpose so
 * tests can include it alongside cblas.h and cross-check the two
 * implementations.
 *
    Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef CPULINALG_HPP
#define CPULINALG_HPP

#include <cstddef>

// These kernels define results by their exact accumulation order, so the
// compiler must not contract a*b+c into fma (contraction skips the
// intermediate rounding and changes bits — on aarch64 -ffp-contract=fast
// is the default, so an unguarded build gives different results than
// x86). Clang honors the standard pragma; GCC ignores it, so builds must
// also pass -ffp-contract=off (see the BLAS_FREE branch in CMakeLists).
#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif

namespace jz {

// C (m x n) = alpha * op(A) (m x k) * op(B) (k x n) + beta * C
// Column-major, physical leading dimensions lda/ldb/ldc.
template <class T>
inline void gemm(bool transA, bool transB, int m, int n, int k, T alpha,
                 const T *A, int lda, const T *B, int ldb, T beta, T *C,
                 int ldc) {
    for (int j = 0; j < n; j++) {
        T *Cj = C + (size_t)j * ldc;
        if (beta == T(0)) {
            for (int i = 0; i < m; i++) Cj[i] = T(0);
        } else if (beta != T(1)) {
            for (int i = 0; i < m; i++) Cj[i] *= beta;
        }
    }
    if (alpha == T(0) || k == 0) return;

    if (!transA) {
        // axpy form: C[:,j] += alpha * op(B)(l,j) * A[:,l].
        // The inner loop runs down a contiguous column of A and C.
        for (int j = 0; j < n; j++) {
            T *Cj = C + (size_t)j * ldc;
            for (int l = 0; l < k; l++) {
                T blj = transB ? B[(size_t)l * ldb + j] : B[(size_t)j * ldb + l];
                if (blj == T(0)) continue;
                T s = alpha * blj;
                const T *Al = A + (size_t)l * lda;
                for (int i = 0; i < m; i++) Cj[i] += s * Al[i];
            }
        }
    } else {
        // dot form: C(i,j) += alpha * <row i of op(A), column j of op(B)>.
        // Row i of op(A) is the contiguous physical column i of A.
        for (int j = 0; j < n; j++) {
            T *Cj = C + (size_t)j * ldc;
            for (int i = 0; i < m; i++) {
                const T *Ai = A + (size_t)i * lda;
                T acc = T(0);
                if (!transB) {
                    const T *Bj = B + (size_t)j * ldb;
                    for (int l = 0; l < k; l++) acc += Ai[l] * Bj[l];
                } else {
                    for (int l = 0; l < k; l++) acc += Ai[l] * B[(size_t)l * ldb + j];
                }
                Cj[i] += alpha * acc;
            }
        }
    }
}

// y = alpha * op(A) * x + beta * y, A physically m x n, column-major.
// Positive strides only (every call site in this library uses 1).
template <class T>
inline void gemv(bool trans, int m, int n, T alpha, const T *A, int lda,
                 const T *x, int incx, T beta, T *y, int incy) {
    int leny = trans ? n : m;
    if (beta == T(0)) {
        for (int i = 0; i < leny; i++) y[(size_t)i * incy] = T(0);
    } else if (beta != T(1)) {
        for (int i = 0; i < leny; i++) y[(size_t)i * incy] *= beta;
    }
    if (alpha == T(0)) return;

    if (!trans) {
        // y += alpha * x[j] * A[:,j]; inner loop contiguous over A columns.
        for (int j = 0; j < n; j++) {
            T s = alpha * x[(size_t)j * incx];
            if (s == T(0)) continue;
            const T *Aj = A + (size_t)j * lda;
            if (incy == 1) {
                for (int i = 0; i < m; i++) y[i] += s * Aj[i];
            } else {
                for (int i = 0; i < m; i++) y[(size_t)i * incy] += s * Aj[i];
            }
        }
    } else {
        // y[j] += alpha * <A[:,j], x>; contiguous dot per column.
        for (int j = 0; j < n; j++) {
            const T *Aj = A + (size_t)j * lda;
            T acc = T(0);
            if (incx == 1) {
                for (int i = 0; i < m; i++) acc += Aj[i] * x[i];
            } else {
                for (int i = 0; i < m; i++) acc += Aj[i] * x[(size_t)i * incx];
            }
            y[(size_t)j * incy] += alpha * acc;
        }
    }
}

}  // namespace jz

#endif
