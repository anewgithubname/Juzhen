// Juzhen::init() and Juzhen::release_cached(): calling init() again is a no-op,
// and freeing the cached pool blocks leaves live matrices intact.
#include "../cpp/juzhen.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Keep one matrix alive while blocks of the same size go back to the cache,
// free the cache, then compute with the live matrix and allocate again.
template <class D, class ToHost>
void check(const char* backend, ToHost to_host) {
    Matrix<float> values("values", {{1, 2, 3}, {4, 5, 6}});
    Matrix<D> live(values);
    {
        Matrix<D> a(values);
        auto b = a + a;
    }
    Juzhen::release_cached();
    Matrix<float> twice = to_host(live + live);
    for (size_t i = 0; i < values.num_row(); ++i)
        for (size_t j = 0; j < values.num_col(); ++j)
            require(twice.elem(i, j) == 2 * values.elem(i, j),
                    std::string(backend) + ": live matrix changed after release_cached()");
    std::cout << "[PASS] " << backend << " matrices survive release_cached()" << std::endl;
}

}  // namespace

int compute() {
    Juzhen::init();  // main() already called it
    check<float>("CPU", [](const Matrix<float>& m) { return m; });
#ifdef CUDA
    check<CUDAfloat>("CUDA", [](const Matrix<CUDAfloat>& m) { return m.to_host(); });
#elif defined(ROCM_HIP)
    check<ROCMfloat>("ROCm", [](const Matrix<ROCMfloat>& m) { return m.to_host(); });
#elif defined(APPLE_SILICON)
    check<MPSfloat>("Metal", [](const Matrix<MPSfloat>& m) { return m.to_host(); });
#endif
    return 0;
}
