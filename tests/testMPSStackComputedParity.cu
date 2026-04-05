/**
 * @file testMPSStackComputedParity.cu
 * @brief CPU vs MPS parity check for hstack/vstack on computed tensors.
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Juzhen;

#if !defined(APPLE_SILICON)
int compute() {
    std::cout << "testMPSStackComputedParity requires APPLE_SILICON. Skipping.\n";
    return 0;
}
#else

static void compare_matrix(const Matrix<float>& a,
                           const Matrix<float>& b,
                           const std::string& tag,
                           float& max_abs,
                           float& rel_l2) {
    max_abs = 0.0f;
    double num = 0.0;
    double den = 0.0;

    for (size_t c = 0; c < a.num_col(); ++c) {
        for (size_t r = 0; r < a.num_row(); ++r) {
            const float da = a.elem(r, c);
            const float db = b.elem(r, c);
            const float d = da - db;
            max_abs = std::max(max_abs, std::fabs(d));
            num += (double)d * (double)d;
            den += (double)da * (double)da;
        }
    }

    rel_l2 = (float)std::sqrt(num / std::max(den, 1e-12));
    std::cout << tag << " max_abs=" << max_abs << " rel_l2=" << rel_l2 << std::endl;
}

int compute() {
    global_rand_gen.seed(2026);

    constexpr int batch = 32;
    constexpr int d = 2;

    auto x0_h = Matrix<float>::randn(d, batch);
    auto x1_h = Matrix<float>::randn(d, batch);
    auto t_h = Matrix<float>::rand(1, batch);

    auto x0_m = Matrix<MPSfloat>(x0_h);
    auto x1_m = Matrix<MPSfloat>(x1_h);
    auto t_m = Matrix<MPSfloat>(t_h);

    auto one_h = Matrix<float>::ones(d, 1);
    auto one_m = Matrix<MPSfloat>::ones(d, 1);

    auto xt_h = hadmd(x0_h, one_h * (1 - t_h)) + hadmd(x1_h, one_h * t_h);
    auto xt_m = hadmd(x0_m, one_m * (1 - t_m)) + hadmd(x1_m, one_m * t_m);

    float max_abs = 0.0f;
    float rel_l2 = 0.0f;

    compare_matrix(xt_h, xt_m.to_host(), "computed source", max_abs, rel_l2);
    if (max_abs > 2e-6f || rel_l2 > 2e-6f) {
        std::cout << "[FAIL] computed source mismatch\n";
        return 1;
    }

    auto aux_h = x1_h - x0_h;
    auto aux_m = x1_m - x0_m;

    compare_matrix(aux_h, aux_m.to_host(), "computed aux source", max_abs, rel_l2);
    if (max_abs > 2e-6f || rel_l2 > 2e-6f) {
        std::cout << "[FAIL] computed aux source mismatch\n";
        return 1;
    }

    auto hcat_h = hstack<float>({xt_h, aux_h});
    auto hcat_m = hstack({xt_m, aux_m}).to_host();

    compare_matrix(hcat_h, hcat_m, "hstack(computed)", max_abs, rel_l2);
    if (max_abs > 2e-6f || rel_l2 > 2e-6f) {
        std::cout << "[FAIL] hstack(computed) mismatch\n";
        return 1;
    }

    auto inp_h = vstack<float>({xt_h, t_h});
    auto inp_m = vstack({xt_m, t_m}).to_host();

    compare_matrix(inp_h, inp_m, "vstack(computed)", max_abs, rel_l2);
    if (max_abs > 2e-6f || rel_l2 > 2e-6f) {
        std::cout << "[FAIL] vstack(computed) mismatch\n";
        return 1;
    }

    std::cout << "[PASS] testMPSStackComputedParity\n";
    return 0;
}
#endif