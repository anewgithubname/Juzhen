// Parity test for the handwritten kernels in cpp/cpulinalg.hpp.
//
// Checks jz::gemm / jz::gemv against a double-precision naive reference over
// all transpose combinations, alpha/beta values, awkward shapes, and padded
// leading dimensions. When the build links an external BLAS (i.e. not
// JUZHEN_NO_BLAS) it also cross-checks against cblas_sgemm/cblas_sgemv, so the
// two code paths validate each other in the default build.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../cpp/cpulinalg.hpp"
#ifndef JUZHEN_NO_BLAS
#include <cblas.h>
#endif

namespace {

std::mt19937 gen(42);

std::vector<float> randvec(size_t n) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto &e : v) e = d(gen);
    return v;
}

// column-major accessor into a physical rows x cols buffer
inline double at(const std::vector<float> &m, int ld, int r, int c) {
    return m[(size_t)c * ld + r];
}

int failures = 0;

void check(bool ok, const char *what, int tA, int tB, int m, int n, int k) {
    if (!ok) {
        std::printf("FAIL %s tA=%d tB=%d m=%d n=%d k=%d\n", what, tA, tB, m, n, k);
        failures++;
    }
}

void test_gemm(int m, int n, int k, bool transA, bool transB, float alpha,
               float beta, int pad) {
    int arows = (transA ? k : m) + pad, acols = transA ? m : k;
    int brows = (transB ? n : k) + pad, bcols = transB ? k : n;
    int crows = m + pad;

    auto A = randvec((size_t)arows * acols);
    auto B = randvec((size_t)brows * bcols);
    auto C0 = randvec((size_t)crows * n);

    // naive reference in double
    std::vector<double> ref((size_t)crows * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double acc = 0;
            for (int l = 0; l < k; l++) {
                double a = transA ? at(A, arows, l, i) : at(A, arows, i, l);
                double b = transB ? at(B, brows, j, l) : at(B, brows, l, j);
                acc += a * b;
            }
            ref[(size_t)j * crows + i] =
                alpha * acc + beta * at(C0, crows, i, j);
        }

    const double tol = 5e-3;

    std::vector<float> C = C0;
    jz::gemm(transA, transB, m, n, k, alpha, A.data(), arows, B.data(), brows,
             beta, C.data(), crows);
    bool ok = true;
    for (int j = 0; j < n && ok; j++)
        for (int i = 0; i < m; i++)
            if (std::fabs(C[(size_t)j * crows + i] - ref[(size_t)j * crows + i]) > tol) {
                ok = false;
                break;
            }
    // untouched rows in the padding must stay untouched
    for (int j = 0; j < n && ok; j++)
        for (int i = m; i < crows; i++)
            if (C[(size_t)j * crows + i] != C0[(size_t)j * crows + i]) {
                ok = false;
                break;
            }
    check(ok, "jz::gemm vs ref", transA, transB, m, n, k);

#ifndef JUZHEN_NO_BLAS
    std::vector<float> Cb = C0;
    cblas_sgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, A.data(),
                arows, B.data(), brows, beta, Cb.data(), crows);
    ok = true;
    for (int j = 0; j < n && ok; j++)
        for (int i = 0; i < m; i++)
            if (std::fabs(C[(size_t)j * crows + i] - Cb[(size_t)j * crows + i]) > tol) {
                ok = false;
                break;
            }
    check(ok, "jz::gemm vs cblas", transA, transB, m, n, k);
#endif
}

void test_gemv(int m, int n, bool trans, float alpha, float beta, int pad) {
    int arows = m + pad;
    int lenx = trans ? m : n;
    int leny = trans ? n : m;

    auto A = randvec((size_t)arows * n);
    auto x = randvec(lenx);
    auto y0 = randvec(leny);

    std::vector<double> ref(leny);
    for (int i = 0; i < leny; i++) {
        double acc = 0;
        for (int l = 0; l < lenx; l++) {
            double a = trans ? at(A, arows, l, i) : at(A, arows, i, l);
            acc += a * x[l];
        }
        ref[i] = alpha * acc + beta * y0[i];
    }

    const double tol = 5e-3;

    std::vector<float> y = y0;
    jz::gemv(trans, m, n, alpha, A.data(), arows, x.data(), 1, beta, y.data(), 1);
    bool ok = true;
    for (int i = 0; i < leny; i++)
        if (std::fabs(y[i] - ref[i]) > tol) {
            ok = false;
            break;
        }
    check(ok, "jz::gemv vs ref", trans, 0, m, n, 0);

#ifndef JUZHEN_NO_BLAS
    std::vector<float> yb = y0;
    cblas_sgemv(CblasColMajor, trans ? CblasTrans : CblasNoTrans, m, n, alpha,
                A.data(), arows, x.data(), 1, beta, yb.data(), 1);
    ok = true;
    for (int i = 0; i < leny; i++)
        if (std::fabs(y[i] - yb[i]) > tol) {
            ok = false;
            break;
        }
    check(ok, "jz::gemv vs cblas", trans, 0, m, n, 0);
#endif
}

}  // namespace

int main() {
    const int shapes[][3] = {{1, 1, 1},   {1, 7, 3},   {5, 1, 4},  {3, 5, 1},
                             {17, 9, 33}, {32, 32, 32}, {21, 40, 13}};
    const float alphas[] = {0.0f, 1.0f, 0.75f};
    const float betas[] = {0.0f, 1.0f, -0.5f};

    for (auto &s : shapes)
        for (int tA = 0; tA < 2; tA++)
            for (int tB = 0; tB < 2; tB++)
                for (float a : alphas)
                    for (float b : betas)
                        for (int pad : {0, 3})
                            test_gemm(s[0], s[1], s[2], tA, tB, a, b, pad);

    const int vshapes[][2] = {{1, 1}, {1, 9}, {9, 1}, {23, 17}, {64, 64}};
    for (auto &s : vshapes)
        for (int t = 0; t < 2; t++)
            for (float a : alphas)
                for (float b : betas)
                    for (int pad : {0, 3})
                        test_gemv(s[0], s[1], t, a, b, pad);

    if (failures == 0) {
        std::printf("testCpuLinalg: all checks passed.\n");
        return 0;
    }
    std::printf("testCpuLinalg: %d checks FAILED.\n", failures);
    return 1;
}
