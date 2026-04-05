/**
 * @file testMPSRandnStability.cu
 * @brief Stress-test MPS randn generation for non-finite values.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <cmath>
#include <iostream>

using namespace Juzhen;

#if !defined(APPLE_SILICON)
int compute() {
    std::cout << "testMPSRandnStability requires APPLE_SILICON. Skipping.\n";
    return 0;
}
#else

static bool find_non_finite(const Matrix<float>& m, size_t& bad_r, size_t& bad_c, float& bad_v) {
    for (size_t c = 0; c < m.num_col(); ++c) {
        for (size_t r = 0; r < m.num_row(); ++r) {
            const float v = m.elem(r, c);
            if (!std::isfinite(v)) {
                bad_r = r;
                bad_c = c;
                bad_v = v;
                return true;
            }
        }
    }
    return false;
}

int compute() {
    global_rand_gen.seed(2026);

    constexpr int rows = 3072;
    constexpr int cols = 16;
    constexpr int iters = 200;

    std::cout << "[INFO] randn stress: shape=" << rows << "x" << cols
              << " iters=" << iters << std::endl;

    for (int it = 1; it <= iters; ++it) {
        auto x_m = Matrix<MPSfloat>::randn(rows, cols);
        auto x_h = x_m.to_host();

        size_t bad_r = 0, bad_c = 0;
        float bad_v = 0.0f;
        if (find_non_finite(x_h, bad_r, bad_c, bad_v)) {
            std::cout << "[FAIL] non-finite randn at iter=" << it
                      << " row=" << bad_r
                      << " col=" << bad_c
                      << " value=" << bad_v << std::endl;
            return 1;
        }

        if (it % 20 == 0) {
            std::cout << "[INFO] iter " << it << " ok" << std::endl;
        }
    }

    std::cout << "[PASS] testMPSRandnStability" << std::endl;
    return 0;
}
#endif
