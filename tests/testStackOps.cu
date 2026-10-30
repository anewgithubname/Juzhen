#include "../cpp/juzhen.hpp"

// ---------------------------------------------------------------------------
// CPU hstack / vstack tests
// ---------------------------------------------------------------------------

int test_hstack_basic()
{
    // [1 2 | 4 5]      [1 2 4 5]
    // [3 4 | 6 7]  =>  [3 4 6 7]
    M A = {"A", {{1, 2}, {3, 4}}};
    M B = {"B", {{4, 5}, {6, 7}}};
    M expected = {"E", {{1, 2, 4, 5}, {3, 4, 6, 7}}};

    auto C = hstack<float>({A, B});
    if ((C - expected).norm() > 1e-5f) {
        LOG_ERROR("hstack basic: value mismatch");
        return 1;
    }
    if (C.num_row() != 2 || C.num_col() != 4) {
        LOG_ERROR("hstack basic: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    LOG_INFO("hstack basic passed");
    return 0;
}

int test_vstack_basic()
{
    // [1 2]   [1 2]
    // [3 4] / [3 4] => [1 2; 3 4; 4 5; 6 7]
    // ----
    // [4 5]
    // [6 7]
    M A = {"A", {{1, 2}, {3, 4}}};
    M B = {"B", {{4, 5}, {6, 7}}};
    M expected = {"E", {{1, 2}, {3, 4}, {4, 5}, {6, 7}}};

    auto C = vstack<float>({A, B});
    if ((C - expected).norm() > 1e-5f) {
        LOG_ERROR("vstack basic: value mismatch");
        return 1;
    }
    if (C.num_row() != 4 || C.num_col() != 2) {
        LOG_ERROR("vstack basic: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    LOG_INFO("vstack basic passed");
    return 0;
}

int test_hstack_single()
{
    M A = {"A", {{1, 2, 3}}};
    auto C = hstack<float>({A});
    if ((C - A).norm() > 1e-5f) {
        LOG_ERROR("hstack single: value mismatch");
        return 1;
    }
    LOG_INFO("hstack single passed");
    return 0;
}

int test_vstack_single()
{
    M A = {"A", {{1}, {2}, {3}}};
    auto C = vstack<float>({A});
    if ((C - A).norm() > 1e-5f) {
        LOG_ERROR("vstack single: value mismatch");
        return 1;
    }
    LOG_INFO("vstack single passed");
    return 0;
}

int test_hstack_transposed_input()
{
    // A = 2x3, A.T() = 3x2; hstack(A.T(), A.T()) should be 3x4
    M A = {"A", {{1, 2, 3}, {4, 5, 6}}};
    M expected = {"E", {{1, 4, 1, 4}, {2, 5, 2, 5}, {3, 6, 3, 6}}};
    auto C = hstack<float>({A.T(), A.T()});
    if (C.num_row() != 3 || C.num_col() != 4) {
        LOG_ERROR("hstack transposed input: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    if ((C - expected).norm() > 1e-5f) {
        LOG_ERROR("hstack transposed input: value mismatch");
        return 1;
    }
    LOG_INFO("hstack transposed input passed");
    return 0;
}

int test_vstack_transposed_input()
{
    // A = 2x3, A.T() = 3x2; vstack(A.T(), A.T()) should be 6x2
    M A = {"A", {{1, 2, 3}, {4, 5, 6}}};
    M expected = {"E", {{1, 4}, {2, 5}, {3, 6}, {1, 4}, {2, 5}, {3, 6}}};
    auto C = vstack<float>({A.T(), A.T()});
    if (C.num_row() != 6 || C.num_col() != 2) {
        LOG_ERROR("vstack transposed input: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    if ((C - expected).norm() > 1e-5f) {
        LOG_ERROR("vstack transposed input: value mismatch");
        return 1;
    }
    LOG_INFO("vstack transposed input passed");
    return 0;
}

int test_hstack_three_matrices()
{
    M A = {"A", {{1}, {2}}};
    M B = {"B", {{3}, {4}}};
    M C = {"C", {{5}, {6}}};
    M expected = {"E", {{1, 3, 5}, {2, 4, 6}}};
    auto R = hstack<float>({A, B, C});
    if ((R - expected).norm() > 1e-5f) {
        LOG_ERROR("hstack three matrices: value mismatch");
        return 1;
    }
    LOG_INFO("hstack three matrices passed");
    return 0;
}

int test_vstack_three_matrices()
{
    M A = {"A", {{1, 2}}};
    M B = {"B", {{3, 4}}};
    M C = {"C", {{5, 6}}};
    M expected = {"E", {{1, 2}, {3, 4}, {5, 6}}};
    auto R = vstack<float>({A, B, C});
    if ((R - expected).norm() > 1e-5f) {
        LOG_ERROR("vstack three matrices: value mismatch");
        return 1;
    }
    LOG_INFO("vstack three matrices passed");
    return 0;
}

// ---------------------------------------------------------------------------
// Validation / error-path tests (CPU)
// ---------------------------------------------------------------------------

int test_hstack_mismatched_rows_throws()
{
    try {
        M A = M::ones(2, 2);
        M B = M::ones(3, 1);
        auto C = hstack<float>({A, B});
        (void)C;
        LOG_ERROR("hstack mismatched rows did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("hstack mismatched rows throws passed");
    return 0;
}

int test_vstack_mismatched_cols_throws()
{
    try {
        M A = M::ones(2, 2);
        M B = M::ones(1, 3);
        auto C = vstack<float>({A, B});
        (void)C;
        LOG_ERROR("vstack mismatched cols did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("vstack mismatched cols throws passed");
    return 0;
}

int test_hstack_empty_throws()
{
    try {
        std::vector<MatrixView<float>> empty;
        auto C = hstack<float>(empty);
        (void)C;
        LOG_ERROR("hstack empty input did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("hstack empty input throws passed");
    return 0;
}

int test_vstack_empty_throws()
{
    try {
        std::vector<MatrixView<float>> empty;
        auto C = vstack<float>(empty);
        (void)C;
        LOG_ERROR("vstack empty input did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("vstack empty input throws passed");
    return 0;
}

// ---------------------------------------------------------------------------
// Device tests (CUDA and Metal)
// ---------------------------------------------------------------------------

#if defined(CUDA) || defined(APPLE_SILICON)
#ifdef CUDA
using GD = CUDAfloat;
#else
using GD = MPSfloat;
#endif
using DM = Matrix<GD>;

int test_gpu_hstack_basic()
{
    M Ah = {"A", {{1, 2}, {3, 4}}};
    M Bh = {"B", {{4, 5}, {6, 7}}};
    DM A(Ah), B(Bh);

    auto C = hstack(std::vector<MatrixView<GD>>{A, B});
    M expected = {"E", {{1, 2, 4, 5}, {3, 4, 6, 7}}};

    if (C.num_row() != 2 || C.num_col() != 4) {
        LOG_ERROR("GPU hstack basic: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    if ((C.to_host() - expected).norm() > 1e-5f) {
        LOG_ERROR("GPU hstack basic: value mismatch");
        return 1;
    }
    LOG_INFO("GPU hstack basic passed");
    return 0;
}

int test_gpu_vstack_basic()
{
    M Ah = {"A", {{1, 2}, {3, 4}}};
    M Bh = {"B", {{4, 5}, {6, 7}}};
    DM A(Ah), B(Bh);

    auto C = vstack(std::vector<MatrixView<GD>>{A, B});
    M expected = {"E", {{1, 2}, {3, 4}, {4, 5}, {6, 7}}};

    if (C.num_row() != 4 || C.num_col() != 2) {
        LOG_ERROR("GPU vstack basic: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    if ((C.to_host() - expected).norm() > 1e-5f) {
        LOG_ERROR("GPU vstack basic: value mismatch");
        return 1;
    }
    LOG_INFO("GPU vstack basic passed");
    return 0;
}

int test_gpu_vstack_transposed_regression()
{
    // Regression: vstack used to return hstack().T() (lazy flag flip),
    // which cuDNN reads as wrong memory layout. The result must now be a
    // physically transposed, non-transposed-flagged matrix.
    M Ah = {"Ah", {{1, 2, 3}, {4, 5, 6}}};
    M Bh = {"Bh", {{7, 8, 9}, {10, 11, 12}}};
    DM A(Ah), B(Bh);

    auto C = vstack(std::vector<MatrixView<GD>>{A.T(), B.T()});
    M Cref = vstack<float>(std::vector<MatrixView<float>>{Ah.T(), Bh.T()});

    if (C.get_transpose() != 0) {
        LOG_ERROR("GPU vstack regression: output still has transpose flag set");
        return 1;
    }
    if ((C.to_host() - Cref).norm() > 1e-5f) {
        LOG_ERROR("GPU vstack regression: value mismatch vs CPU reference");
        return 1;
    }
    LOG_INFO("GPU vstack transposed regression passed");
    return 0;
}

int test_gpu_hstack_transposed_input()
{
    M Ah = {"A", {{1, 2, 3}, {4, 5, 6}}};
    DM A(Ah);
    M expected = {"E", {{1, 4, 1, 4}, {2, 5, 2, 5}, {3, 6, 3, 6}}};

    auto C = hstack(std::vector<MatrixView<GD>>{A.T(), A.T()});
    if (C.num_row() != 3 || C.num_col() != 4) {
        LOG_ERROR("GPU hstack transposed input: wrong shape ({} x {})", C.num_row(), C.num_col());
        return 1;
    }
    if ((C.to_host() - expected).norm() > 1e-5f) {
        LOG_ERROR("GPU hstack transposed input: value mismatch");
        return 1;
    }
    LOG_INFO("GPU hstack transposed input passed");
    return 0;
}

int test_gpu_hstack_mismatched_rows_throws()
{
    try {
        DM A(M::ones(2, 2));
        DM B(M::ones(3, 1));
        auto C = hstack(std::vector<MatrixView<GD>>{A, B});
        (void)C;
        LOG_ERROR("GPU hstack mismatched rows did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("GPU hstack mismatched rows throws passed");
    return 0;
}

int test_gpu_hstack_empty_throws()
{
    try {
        std::vector<MatrixView<GD>> empty;
        auto C = hstack(empty);
        (void)C;
        LOG_ERROR("GPU hstack empty input did not throw");
        return 1;
    } catch (const std::invalid_argument&) {}
    LOG_INFO("GPU hstack empty input throws passed");
    return 0;
}

// CPU/CUDA parity: hstack and vstack must agree element-by-element
int test_cpu_gpu_hstack_parity()
{
    M Ah = M::randn(4, 3);
    M Bh = M::randn(4, 5);
    DM A(Ah), B(Bh);

    M cpu_result = hstack<float>({Ah, Bh});
    auto gpu_result = hstack(std::vector<MatrixView<GD>>{A, B});

    if ((gpu_result.to_host() - cpu_result).norm() > 1e-4f) {
        LOG_ERROR("CPU/GPU hstack parity: mismatch");
        return 1;
    }
    LOG_INFO("CPU/GPU hstack parity passed");
    return 0;
}

int test_cpu_gpu_vstack_parity()
{
    M Ah = M::randn(3, 4);
    M Bh = M::randn(5, 4);
    DM A(Ah), B(Bh);

    M cpu_result = vstack<float>({Ah, Bh});
    auto gpu_result = vstack(std::vector<MatrixView<GD>>{A, B});

    if ((gpu_result.to_host() - cpu_result).norm() > 1e-4f) {
        LOG_ERROR("CPU/GPU vstack parity: mismatch");
        return 1;
    }
    LOG_INFO("CPU/GPU vstack parity passed");
    return 0;
}

#endif // CUDA || APPLE_SILICON

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int compute()
{
    int ret = 0;

    // CPU functional tests
    ret += test_hstack_basic();
    ret += test_vstack_basic();
    ret += test_hstack_single();
    ret += test_vstack_single();
    ret += test_hstack_transposed_input();
    ret += test_vstack_transposed_input();
    ret += test_hstack_three_matrices();
    ret += test_vstack_three_matrices();

    // CPU error-path tests
    ret += test_hstack_mismatched_rows_throws();
    ret += test_vstack_mismatched_cols_throws();
    ret += test_hstack_empty_throws();
    ret += test_vstack_empty_throws();

#if defined(CUDA) || defined(APPLE_SILICON)
    // Device functional tests
    ret += test_gpu_hstack_basic();
    ret += test_gpu_vstack_basic();
    ret += test_gpu_vstack_transposed_regression();
    ret += test_gpu_hstack_transposed_input();

    // Device error-path tests
    ret += test_gpu_hstack_mismatched_rows_throws();
    ret += test_gpu_hstack_empty_throws();

    // Parity tests
    ret += test_cpu_gpu_hstack_parity();
    ret += test_cpu_gpu_vstack_parity();
#endif

    if (ret == 0) {
        LOG_INFO("--------------------");
        LOG_INFO("|      ALL OK!     |");
        LOG_INFO("--------------------");
    } else {
        LOG_ERROR("--------------------");
        LOG_ERROR("|    NOT ALL OK!   |");
        LOG_ERROR("--------------------");
    }

    return ret;
}
