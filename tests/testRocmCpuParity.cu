#include "../cpp/juzhen.hpp"

#ifdef ROCM_HIP

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int CheckClose(const M& got, const M& expected, const char* name, float tol = 1e-4f) {
    if (got.num_row() != expected.num_row() || got.num_col() != expected.num_col()) {
        LOG_ERROR("{} shape mismatch: got {}x{}, expected {}x{}",
                  name,
                  got.num_row(),
                  got.num_col(),
                  expected.num_row(),
                  expected.num_col());
        return 1;
    }

    float err = (got - expected).norm();
    if (err > tol) {
        LOG_ERROR("{} mismatch, err={}", name, err);
        std::cout << "got:\n" << got << std::endl;
        std::cout << "expected:\n" << expected << std::endl;
        return 1;
    }
    return 0;
}

int CheckScalarClose(float got, float expected, const char* name, float tol = 1e-4f) {
    float err = std::fabs(got - expected);
    if (err > tol) {
        LOG_ERROR("{} mismatch, got={}, expected={}, err={}", name, got, expected, err);
        return 1;
    }
    return 0;
}

int CheckFinite(const M& m, const char* name) {
    for (size_t r = 0; r < m.num_row(); ++r) {
        for (size_t c = 0; c < m.num_col(); ++c) {
            if (!std::isfinite(m.elem(r, c))) {
                LOG_ERROR("{} has non-finite value at ({}, {})", name, r, c);
                return 1;
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    int ret = 0;

    M A("A", {{1.25f, 2.50f, 3.75f},
               {4.25f, 5.50f, 6.75f},
               {7.25f, 8.50f, 9.75f}});
    M B("B", {{0.50f, 1.50f, 2.50f},
               {3.50f, 4.50f, 5.50f},
               {6.50f, 7.50f, 8.50f}});

    CM RA(A);
    CM RB(B);

    ret += CheckClose(RA.to_host(), A, "to_host_ctor");
    ret += CheckClose(RA.T().to_host(), A.T(), "transpose_T");

    ret += CheckClose(Matrix<ROCMfloat>::ones(3, 3).to_host(), Matrix<float>::ones(3, 3), "static_ones");
    ret += CheckClose(Matrix<ROCMfloat>::zeros(3, 3).to_host(), Matrix<float>::zeros(3, 3), "static_zeros");

    {
        CM x("x", 3, 3);
        x.ones();
        ret += CheckClose(x.to_host(), Matrix<float>::ones(3, 3), "ones_inplace");
        x.zeros();
        ret += CheckClose(x.to_host(), Matrix<float>::zeros(3, 3), "zeros_inplace");
    }

    ret += CheckClose(static_cast<const CM&>(RA).add(RB, 1.2f, -0.3f).to_host(),
                      static_cast<const M&>(A).add(B, 1.2f, -0.3f),
                      "add_matrix_const");

    {
        CM x(A);
        M y(A);
        x.add(RB, 1.2f, -0.3f);
        y.add(B, 1.2f, -0.3f);
        ret += CheckClose(x.to_host(), y, "add_matrix_inplace");
    }

    ret += CheckClose(static_cast<const CM&>(RA).add(0.75f, -2.0f).to_host(),
                      static_cast<const M&>(A).add(0.75f, -2.0f),
                      "add_scalar_const");

    {
        CM x(A);
        M y(A);
        x.add(0.75f, -2.0f);
        y.add(0.75f, -2.0f);
        ret += CheckClose(x.to_host(), y, "add_scalar_inplace");
    }

    {
        CM x(A);
        M y(A);
        x.scale(0.125f);
        y.scale(0.125f);
        ret += CheckClose(x.to_host(), y, "scale_inplace");
    }

    ret += CheckClose((RA * RB).to_host(), A * B, "dot_gemm");
    ret += CheckClose((RA.T() * RB).to_host(), A.T() * B, "dot_gemm_transpose_a");

    ret += CheckClose(sum(RA, 0).to_host(), sum(A, 0), "sum_dim0");
    ret += CheckClose(sum(RA, 1).to_host(), sum(A, 1), "sum_dim1");

    ret += CheckClose(static_cast<const CM&>(RA).eleminv(1.0).to_host(),
                      static_cast<const M&>(A).eleminv(1.0),
                      "eleminv_const");

    {
        CM x(A);
        M y(A);
        x.eleminv(2.0);
        y.eleminv(2.0);
        ret += CheckClose(x.to_host(), y, "eleminv_inplace");
    }

    ret += CheckClose(hadmd(RA, RB).to_host(), hadmd(A, B), "hadamard_const");
    ret += CheckClose(hadmd(CM(A), RB).to_host(), hadmd(M(A), B), "hadamard_rvalue_left");
    ret += CheckClose(hadmd(RA, CM(B)).to_host(), hadmd(A, M(B)), "hadamard_rvalue_right");

    ret += CheckClose(square(RA).to_host(), square(A), "square_const");
    ret += CheckClose(square(CM(A)).to_host(), square(M(A)), "square_inplace_rvalue");

    ret += CheckClose(tanh(RA).to_host(), tanh(A), "tanh_const");
    ret += CheckClose(tanh(CM(A)).to_host(), tanh(M(A)), "tanh_inplace_rvalue");

    ret += CheckClose(d_tanh(RA).to_host(), d_tanh(A), "dtanh_const");
    ret += CheckClose(d_tanh(CM(A)).to_host(), d_tanh(M(A)), "dtanh_inplace_rvalue");

    ret += CheckClose(log(RA).to_host(), log(A), "log_const");
    ret += CheckClose(exp(RA).to_host(), exp(A), "exp_const", 1e-3f);
    ret += CheckClose(log(exp(RA)).to_host(), log(exp(A)), "log_exp_chain", 1e-3f);

    {
        auto rs = static_cast<const CM&>(RA).slice(1, 3, 0, 2).to_host();
        auto cs = static_cast<const M&>(A).slice(1, 3, 0, 2);
        ret += CheckClose(rs, cs, "slice_extract");
    }

    {
        CM x(A);
        M y(A);
        CM patch(M("patch", {{100.0f, 101.0f}, {110.0f, 111.0f}}));
        M patch_cpu("patch_cpu", {{100.0f, 101.0f}, {110.0f, 111.0f}});
        x.slice(1, 3, 1, 3, patch);
        y.slice(1, 3, 1, 3, patch_cpu);
        ret += CheckClose(x.to_host(), y, "slice_assign");
    }

    {
        ret += CheckClose(RA.rows(0, 2).to_host(), A.rows(0, 2), "rows_extract");
        ret += CheckClose(RA.columns(1, 3).to_host(), A.columns(1, 3), "columns_extract");

        CM x(A);
        M y(A);
        CM rpatch(M("rpatch", {{-3.0f, -2.0f, -1.0f}}));
        M rpatch_cpu("rpatch_cpu", {{-3.0f, -2.0f, -1.0f}});
        x.rows(2, 3, rpatch);
        y.rows(2, 3, rpatch_cpu);

        CM cpatch(M("cpatch", {{9.0f}, {8.0f}, {7.0f}}));
        M cpatch_cpu("cpatch_cpu", {{9.0f}, {8.0f}, {7.0f}});
        x.columns(0, 1, cpatch);
        y.columns(0, 1, cpatch_cpu);

        ret += CheckClose(x.to_host(), y, "rows_columns_assign");
    }

    {
        CM x(A);
        M y(A);
        fill(x, 0.25);
        y.add(0.25f, 0.0f);
        ret += CheckClose(x.to_host(), y, "fill");
    }

    {
        CM dst("dst", 3, 3);
        copy(dst, RA);
        ret += CheckClose(dst.to_host(), A, "copy");
    }

    {
        CM hs = hstack(std::vector<MatrixView<ROCMfloat>>{RA, RB});
        M hs_cpu = hstack(std::vector<MatrixView<float>>{A, B});
        ret += CheckClose(hs.to_host(), hs_cpu, "hstack");

        CM vs = vstack(std::vector<MatrixView<ROCMfloat>>{RA, RB});
        M vs_cpu = vstack(std::vector<MatrixView<float>>{A, B});
        ret += CheckClose(vs.to_host(), vs_cpu, "vstack");
    }

    {
        float gpu_norm = RA.norm();
        float cpu_norm = A.norm();
        ret += CheckScalarClose(gpu_norm, cpu_norm, "norm");
    }

    {
        CM rr = Matrix<ROCMfloat>::rand(32, 32);
        M rrh = rr.to_host();
        ret += CheckFinite(rrh, "rand_finite");
        ret += CheckScalarClose(rrh.norm(), rrh.norm(), "rand_norm_sanity", 0.0f);

        CM rn = Matrix<ROCMfloat>::randn(32, 32);
        M rnh = rn.to_host();
        ret += CheckFinite(rnh, "randn_finite");
        ret += CheckScalarClose(rnh.norm(), rnh.norm(), "randn_norm_sanity", 0.0f);
    }

    if (ret == 0) {
        LOG_INFO("ROCm CPU parity test passed.");
        return 0;
    }

    LOG_ERROR("ROCm CPU parity test failed with {} checks.", ret);
    return 1;
}

#else

int main() {
    LOG_INFO("ROCM_HIP is not enabled; skipping ROCm CPU parity test.");
    return 0;
}

#endif
