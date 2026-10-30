/**
 * @file testMPSGemmAddressBoundary.cu
 * @brief Metal GEMM results that straddle a 2 GiB-aligned GPU address.
 *
 * MPSMatrixMultiplication writes a band of NaNs into its result when the
 * result buffer crosses a 2 GiB-aligned GPU virtual address, or ends less than
 * about 8 MiB below one. The Metal wrapper routes such results through scratch
 * buffers clear of the boundary. This test places results across the boundary
 * and just below it (by recycling a pool buffer at that address) and compares
 * plain, transposed, row-blocked and batched products with host references.
 * Untouched placeholder buffers cost address space only.
 *
 * Built with -DJUZHEN_MPS_GEMM_WORKAROUND=OFF, the test checks MPS itself: a
 * pass means macOS no longer has the bug and the workaround can be removed.
 */
#include "../cpp/juzhen.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef APPLE_SILICON
int compute() {
    // LOG_INFO (not std::cout) keeps spdlog linked before libjuzhen needs it.
    LOG_INFO("testMPSGemmAddressBoundary requires the Metal backend; skipping");
    return 77;
}
#else

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

enum class Placement { Across, JustBelow };

static bool placed(const MPSfloat* p, size_t bytes, Placement where) {
    const unsigned long long lo = mpsGpuAddress(reinterpret_cast<const float*>(p)), hi = lo + bytes - 1;
    const bool across = lo >> 31 != hi >> 31;
    // JustBelow: ends within 6 MiB below a boundary without reaching it.
    return where == Placement::Across ? across : !across && hi >> 31 != (hi + (6ull << 20)) >> 31;
}

static const char* name(Placement where) {
    return where == Placement::Across ? "across 2 GiB" : "just below 2 GiB";
}

// Leave a pool buffer of rows x cols at the given placement in the dead list,
// so the next allocation of that size reuses it. The returned placeholders
// must stay alive until then, and each case uses a size of its own: earlier
// placeholders of the same size would also sit in the dead list.
static std::vector<Matrix<MPSfloat>> leave_buffer(size_t rows, size_t cols, Placement where) {
    std::vector<Matrix<MPSfloat>> placeholders;
    for (int attempt = 0; attempt < 4096; ++attempt) {
        Matrix<MPSfloat> m("placeholder", rows, cols);
        if (placed(m.data(), rows * cols * sizeof(float), where)) return placeholders;  // m returns to the pool
        placeholders.push_back(std::move(m));
    }
    throw std::runtime_error(std::string("could not place a buffer ") + name(where));
}

static Matrix<float> fixture(size_t rows, size_t cols, float phase) {
    Matrix<float> m("fixture", rows, cols);
    for (size_t j = 0; j < cols; ++j)
        for (size_t i = 0; i < rows; ++i) m.elem(i, j) = std::sin(0.37f * float(i) + 0.61f * float(j) + phase);
    return m;
}

static void close(const Matrix<float>& actual, const Matrix<float>& expected, const std::string& tag) {
    require(actual.num_row() == expected.num_row() && actual.num_col() == expected.num_col(), tag + " shape");
    double worst = 0;
    for (size_t j = 0; j < actual.num_col(); ++j)
        for (size_t i = 0; i < actual.num_row(); ++i) {
            const double e = std::abs(double(actual.elem(i, j)) - expected.elem(i, j));
            require(std::isfinite(e), tag + " non-finite result at (" + std::to_string(i) + "," + std::to_string(j) + ")");
            worst = std::max(worst, e / (1.0 + std::abs(double(expected.elem(i, j)))));
        }
    require(worst < 1e-4, tag + " relative error " + std::to_string(worst));
    std::cout << "[PASS] " << tag << " (max rel err " << worst << ")\n";
}

// The same logical matrix stored transposed: physical storage m^T, flagged.
static Matrix<MPSfloat> transposed_storage(const Matrix<float>& m) {
    Matrix<float> storage("storage", m.num_col(), m.num_row());
    for (size_t j = 0; j < m.num_col(); ++j)
        for (size_t i = 0; i < m.num_row(); ++i) storage.elem(j, i) = m.elem(i, j);
    return Matrix<MPSfloat>(storage).T();
}

static void gemm_case(size_t M, size_t N, size_t K, bool transpose_left, Placement where, std::string tag) {
    tag += std::string(" result ") + name(where);
    const auto ah = fixture(M, K, 0.1f), bh = fixture(K, N, 0.7f);
    Matrix<MPSfloat> left = transpose_left ? transposed_storage(ah) : Matrix<MPSfloat>(ah);
    require(bool(left.get_transpose()) == transpose_left, tag + " operand layout");
    Matrix<MPSfloat> right(bh);
    auto placeholders = leave_buffer(M, N, where);
    auto C = left * right;
    require(placed(C.data(), M * N * sizeof(float), where), tag + " was not placed as intended");
    close(C.to_host(), ah * bh, tag);
}

static void batched_case(int batch, int n, float beta, Placement where, std::string tag) {
    tag += std::string(" result ") + name(where);
    const size_t rows = size_t(batch) * n;
    const auto ah = fixture(rows, n, 0.3f), bh = fixture(rows, n, 0.9f), ch = fixture(rows, n, 1.7f);
    // Metal matrices are row-major, so batch matrix i occupies rows [i*n, (i+1)*n).
    Matrix<MPSfloat> A(ah), B(bh);
    auto placeholders = leave_buffer(rows, n, where);
    Matrix<MPSfloat> C(ch);
    require(placed(C.data(), rows * n * sizeof(float), where), tag + " was not placed as intended");
    mpsGemmBatched(reinterpret_cast<const float*>(A.data()), reinterpret_cast<const float*>(B.data()),
                   const_cast<float*>(reinterpret_cast<const float*>(C.data())), batch, n, n, n, n,
                   false, true, 0.5f, beta);
    Matrix<float> expected("expected", rows, n);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int k = 0; k < n; ++k) s += double(ah.elem(b * n + i, k)) * bh.elem(b * n + j, k);
                expected.elem(b * n + i, j) = float(0.5 * s + beta * ch.elem(b * n + i, j));
            }
    close(C.to_host(), expected, tag);
}

int compute() {
#if defined(JUZHEN_MPS_GEMM_WORKAROUND) && !JUZHEN_MPS_GEMM_WORKAROUND
    std::cout << "MPS GEMM workaround disabled: testing MPS itself\n";
#endif
    try {
        const auto across = Placement::Across, below = Placement::JustBelow;
        gemm_case(6000, 2500, 64, false, across, "GEMM");
        gemm_case(6000, 2501, 64, true, across, "transposed-left GEMM");
        // > 256 MB of result: several row blocks, offsets into a transposed operand
        gemm_case(30000, 2500, 16, true, across, "row-blocked GEMM");
        // ~4 MB results that fit entirely inside the 8 MiB below a boundary
        gemm_case(1000, 1003, 64, false, below, "GEMM");
        gemm_case(1001, 1003, 32, true, below, "transposed-left GEMM");
        batched_case(512, 32, 0.0f, across, "batched GEMM");
        batched_case(513, 32, 1.0f, across, "accumulating batched GEMM");
        // > 256 MB of result: several batch groups
        batched_case(16400, 64, 1.0f, across, "grouped batched GEMM");
        batched_case(257, 64, 1.0f, below, "accumulating batched GEMM");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
#endif
