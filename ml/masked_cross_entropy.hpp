#pragma once

#include "layer.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace Juzhen {
namespace masked_ce_detail {
// CUDA extended lambdas must be defined outside private/protected members.
template<class D> Matrix<D> column_max(const Matrix<D>& input) {
    return reduce(
        [] __GPU_CPU__(float* v, float* result, int count, int) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < count; ++i) maximum = maximum > v[i] ? maximum : v[i];
            result[0] = maximum;
        }, input, 0, 1);
}
} // namespace masked_ce_detail

// Weighted softmax cross-entropy for masked-token training. Targets are one-hot
// columns (or normalized probability distributions); weights have shape (1,N).
// No implicit batch averaging is applied: weights contain the desired scale.
// A zero weight removes its column from both the loss and its input gradient.
template<class D> class MaskedCELayer : public Layer<D> {
    Matrix<D> output_, weights_, ones_;

    void check_input(const Matrix<D>& input) const {
        if (input.num_row() != output_.num_row() || input.num_col() != output_.num_col())
            throw std::invalid_argument("masked cross-entropy input/target shape mismatch");
    }

public:
    MaskedCELayer(size_t nb, Matrix<D>&& target, Matrix<D>&& weights)
        : Layer<D>(2, 2, nb), output_(std::move(target)),
          weights_(std::move(weights)), ones_("masked CE ones", output_.num_row(), 1) {
        if (!nb || !output_.num_row() || output_.num_col() != nb ||
            weights_.num_row() != 1 || weights_.num_col() != nb)
            throw std::invalid_argument("masked cross-entropy target/weight shape mismatch");
        Matrix<float> host_weights = [&]() {
            if constexpr (std::is_same_v<D, float>) return weights_;
            else return weights_.to_host();
        }();
        for (size_t j = 0; j < nb; ++j)
            if (!std::isfinite(host_weights.elem(0, j)) || host_weights.elem(0, j) < 0)
                throw std::invalid_argument("masked cross-entropy weights must be finite and nonnegative");
        ones_.ones();
    }

    bool supports_backward() const override { return true; }
    std::vector<ParameterRef<D>> parameters() override { return {}; }

    Matrix<D> grad(const Matrix<D>& input) const override {
        check_input(input);
        auto maximum = masked_ce_detail::column_max(input);
        auto exponents = exp(input - ones_ * maximum);
        auto denominator = ones_ * sum(exponents, 0);
        auto probabilities = exponents / std::move(denominator);
        return hadmd(probabilities - output_, ones_ * weights_);
    }

    Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream) override {
        check_input(input);
        if (upstream.num_row() != 1 || upstream.num_col() != 1)
            throw std::invalid_argument("masked cross-entropy requires a scalar upstream gradient");
        auto broadcast = Matrix<D>::ones(input.num_row(), 1) * upstream
                       * Matrix<D>::ones(1, input.num_col());
        return hadmd(grad(input), std::move(broadcast));
    }

    Matrix<D> backward_update(const Matrix<D>& input, Matrix<D>&& upstream) override {
        return backward(input, std::move(upstream));
    }

    Matrix<D> jvp(const Matrix<D>&, const Matrix<D>&) override {
        throw std::logic_error("masked cross-entropy is a loss head; compute network JVP without the loss");
    }

    void eval(const Matrix<D>& input) override {
        check_input(input);
        auto maximum = masked_ce_detail::column_max(input);
        auto lse = log(sum(exp(input - ones_ * maximum), 0)) + maximum;
        auto target_score = sum(hadmd(input, output_), 0);
        this->val = sum(hadmd(weights_, lse - target_score), 1);
    }
};
} // namespace Juzhen
