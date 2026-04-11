#include "../cpp/juzhen.hpp"

#ifdef ROCM_HIP

namespace {

int CheckClose(const M& got, const M& expected, const char* name, float tol = 1e-4f) {
    float err = (got - expected).norm();
    if (err > tol) {
        LOG_ERROR("{} failed, err={}", name, err);
        std::cout << "got:\n" << got << std::endl;
        std::cout << "expected:\n" << expected << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int ret = 0;

    CM A(M("A", {{1, 2}, {3, 4}}));
    CM B(M("B", {{5, 6}, {7, 8}}));

    // add: A + B
    {
        M got = static_cast<const CM&>(A).add(B, 1.0f, 1.0f).to_host();
        M expected("add_expected", {{6, 8}, {10, 12}});
        ret += CheckClose(got, expected, "add");
    }

    // scale: 2 * A
    {
        CM C(A);
        C.scale(2.0f);
        M got = C.to_host();
        M expected("scale_expected", {{2, 4}, {6, 8}});
        ret += CheckClose(got, expected, "scale");
    }

    // hadamard: A .* B
    {
        M got = hadmd(A, B).to_host();
        M expected("hadamard_expected", {{5, 12}, {21, 32}});
        ret += CheckClose(got, expected, "hadamard");
    }

    // sum over rows (dim=0): column-wise sum
    {
        M got = sum(A, 0).to_host();
        M expected("sum_dim0_expected", {{4, 6}});
        ret += CheckClose(got, expected, "sum_dim0");
    }

    // sum over cols (dim=1): row-wise sum
    {
        M got = sum(A, 1).to_host();
        M expected("sum_dim1_expected", {{3}, {7}});
        ret += CheckClose(got, expected, "sum_dim1");
    }

    // eleminv: l / A with l=1
    {
        M got = static_cast<const CM&>(A).eleminv(1.0).to_host();
        M expected("eleminv_expected", {{1.0f, 0.5f}, {1.0f / 3.0f, 0.25f}});
        ret += CheckClose(got, expected, "eleminv");
    }

    // hstack: [A | B]
    {
        CM h = hstack(std::vector<MatrixView<ROCMfloat>>{A, B});
        M got = h.to_host();
        M expected("hstack_expected", {{1, 2, 5, 6}, {3, 4, 7, 8}});
        ret += CheckClose(got, expected, "hstack");
    }

    // vstack: [A; B]
    {
        CM v = vstack(std::vector<MatrixView<ROCMfloat>>{A, B});
        M got = v.to_host();
        M expected("vstack_expected", {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        ret += CheckClose(got, expected, "vstack");
    }

    if (ret == 0) {
        LOG_INFO("ROCm ops smoke test passed.");
        return 0;
    }

    LOG_ERROR("ROCm ops smoke test failed with {} checks.", ret);
    return 1;
}

#else

int main() {
    LOG_INFO("ROCM_HIP is not enabled; skipping ROCm ops smoke test.");
    return 0;
}

#endif
