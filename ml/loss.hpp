#pragma once

#include "../cpp/core.hpp"
#include "../cpp/matrix.hpp"
#include "./util.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace Juzhen {

// Loss functions. Each takes the network output (one column per sample) and
// returns the scalar loss together with its gradient with respect to that
// output, ready to pass to Sequential::backward:
//
//     auto [loss, dpred] = softmax_cross_entropy(network.eval(x), labels);
//     network.backward(std::move(dpred));
//
// `value` is a 1x1 matrix that stays on the device, so reading it (and the
// host synchronization that implies) is left to the caller.
template<class D> struct LossResult {
    Matrix<D> value;
    Matrix<D> grad;
};

namespace loss_detail {
template<class D> void check_same_shape(const Matrix<D>& a, const Matrix<D>& b, const char* message) {
    if (a.num_row() != b.num_row() || a.num_col() != b.num_col() || !a.num_row() || !a.num_col())
        throw std::invalid_argument(message);
}

// maximum of each column, shape (1, N)
template<class D> Matrix<D> column_max(const Matrix<D>& input) {
    return reduce(
        [] __GPU_CPU__(float* v, float* result, int count, int) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < count; ++i)
                maximum = maximum > v[i] ? maximum : v[i];
            result[0] = maximum;
        }, input, 0, 1);
}
} // namespace loss_detail

// Mean squared error, averaged over the batch (columns):
//   value = sum((prediction - target)^2) / N
//   grad  = 2 (prediction - target) / N
template<class D>
LossResult<D> mse(const Matrix<D>& prediction, const Matrix<D>& target) {
    loss_detail::check_same_shape(prediction, target, "mse: prediction/target shape mismatch");
    const size_t n = prediction.num_col();
    auto value = sum(sum(square(target - prediction), 0), 1) / n;
    auto grad = 2.0 * (prediction - target) / n;
    return {std::move(value), std::move(grad)};
}

// Softmax cross-entropy, averaged over the batch. `labels` holds one-hot
// columns (or probability distributions) with the same shape as `logits`.
//   value = -sum_c (<logits_c, labels_c> - logsumexp(logits_c)) / N
//   grad  = (softmax(logits) - labels) / N
template<class D>
LossResult<D> softmax_cross_entropy(const Matrix<D>& logits, const Matrix<D>& labels) {
    loss_detail::check_same_shape(logits, labels, "softmax_cross_entropy: logits/labels shape mismatch");
    const size_t n = logits.num_col();
    const auto ones = Matrix<D>::ones(logits.num_row(), 1);

    // subtract each column's maximum before exp() so it cannot overflow
    auto maximum = loss_detail::column_max(logits);
    auto exponents = exp(logits - ones * maximum);
    auto totals = sum(exponents, 0);                      // (1, N)
    auto log_sum_exp = log(Matrix<D>(totals)) + maximum;  // (1, N)

    auto value = -sum(sum(hadmd(logits, labels), 0) - log_sum_exp, 1) / n;
    auto grad = -(labels - exponents / (ones * std::move(totals))) / n;
    return {std::move(value), std::move(grad)};
}

// Weighted softmax cross-entropy for masked-token training. `weights` has
// shape (1, N), must be finite and nonnegative, and already contains any
// desired scaling: no batch averaging is applied. A zero weight removes its
// column from both the loss and the gradient.
//   value = sum_c w_c (logsumexp(logits_c) - <logits_c, labels_c>)
//   grad  = (softmax(logits) - labels) * w   (w broadcast down each column)
template<class D>
LossResult<D> masked_softmax_cross_entropy(const Matrix<D>& logits, const Matrix<D>& labels,
                                           const Matrix<D>& weights) {
    loss_detail::check_same_shape(logits, labels,
                                  "masked_softmax_cross_entropy: logits/labels shape mismatch");
    const size_t n = logits.num_col();
    if (weights.num_row() != 1 || weights.num_col() != n)
        throw std::invalid_argument("masked_softmax_cross_entropy: weights must have shape (1, N)");

    Matrix<float> host_weights = [&]() {
        if constexpr (std::is_same_v<D, float>) return weights;
        else return weights.to_host();
    }();
    for (size_t j = 0; j < n; ++j)
        if (!std::isfinite(host_weights.elem(0, j)) || host_weights.elem(0, j) < 0)
            throw std::invalid_argument("masked_softmax_cross_entropy: weights must be finite and nonnegative");

    const auto ones = Matrix<D>::ones(logits.num_row(), 1);
    auto maximum = loss_detail::column_max(logits);
    auto exponents = exp(logits - ones * maximum);
    auto totals = sum(exponents, 0);                      // (1, N)
    auto log_sum_exp = log(Matrix<D>(totals)) + maximum;  // (1, N)

    auto target_score = sum(hadmd(logits, labels), 0);
    auto value = sum(hadmd(weights, log_sum_exp - target_score), 1);
    auto probabilities = exponents / (ones * std::move(totals));
    auto grad = hadmd(probabilities - labels, ones * weights);
    return {std::move(value), std::move(grad)};
}

// Fraction of columns whose largest score is not at the labelled class.
// An evaluation metric only: it has no gradient.
template<class D>
Matrix<D> error_rate(const Matrix<D>& scores, const Matrix<D>& labels) {
    loss_detail::check_same_shape(scores, labels, "error_rate: scores/labels shape mismatch");
    auto predicted = predict_one_hot(scores);
    return 1 - sum(sum(hadmd(predicted, labels), 0), 1) / scores.num_col();
}

} // namespace Juzhen
