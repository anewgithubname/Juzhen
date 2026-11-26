#pragma once

#include "../cpp/core.hpp"
#include "../cpp/matrix.hpp"
#include "./util.cuh"
#include "./ids.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
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
#ifdef APPLE_SILICON
// One fused kernel instead of the generic reduction's host round trip.
inline Matrix<MPSfloat> column_max(const Matrix<MPSfloat>& input) {
    Matrix<MPSfloat> result("column_max", 1, input.num_col());
    mpsColumnMax(reinterpret_cast<const float*>(input.data()),
                 const_cast<float*>(reinterpret_cast<const float*>(result.data())),
                 (int)input.num_row(), (int)input.num_col(), input.get_transpose());
    return result;
}
#endif

// logsumexp(z_j) - <z_j, labels_j> for every column j, shape (1, N), from the
// logits shifted by their column maximum:
//   log(totals) - <shifted, labels> + maximum (1 - sum(labels))
// The target score is taken from the shifted logits, so for one-hot labels
// (and any labels that sum to one) nothing of the size of the logits is
// subtracted: the value stays exact when the logits are huge.
template<class D> Matrix<D> column_loss(const Matrix<D>& shifted, const Matrix<D>& labels,
                                        const Matrix<D>& maximum, const Matrix<D>& totals) {
    auto unlabelled = 1 - sum(labels, 0);  // (1, N), zero for one-hot columns
    return log(Matrix<D>(totals)) + hadmd(maximum, Matrix<D>(unlabelled)) - sum(hadmd(shifted, labels), 0);
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
    auto shifted = logits - ones * maximum;               // every entry <= 0
    auto exponents = exp(shifted);
    auto totals = sum(exponents, 0);                      // (1, N)

    auto value = sum(loss_detail::column_loss(shifted, labels, maximum, totals), 1) / n;
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
    auto shifted = logits - ones * maximum;               // every entry <= 0
    auto exponents = exp(shifted);
    auto totals = sum(exponents, 0);                      // (1, N)

    auto value = sum(hadmd(weights, loss_detail::column_loss(shifted, labels, maximum, totals)), 1);
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


// ---------------------------------------------------------------------------
// The same losses with labels given as class numbers.
//
// `labels` holds one integer per column of `logits`, read in storage order:
// 1 x N, or seq_len x batch for a language model (one sequence per column).
// A label equal to `ignore_index` is skipped: it adds nothing to the value or
// to the gradient. The dense one-hot matrix is never built.

namespace loss_detail {

// Labels in storage order, with the skipped ones mapped to -1; throws on a
// label outside [0, classes) or on a wrong count.
inline std::vector<int> class_numbers(const Matrix<int>& labels, size_t classes, size_t columns,
                                      int ignore_index, const char* what) {
    if (labels.num_row() * labels.num_col() != columns)
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(labels.num_row() * labels.num_col()) +
                                    " labels for " + std::to_string(columns) + " columns of logits");
    std::vector<int> out;
    out.reserve(columns);
    for (size_t j = 0; j < labels.num_col(); ++j)
        for (size_t i = 0; i < labels.num_row(); ++i) {
            const int y = labels.elem(i, j);
            if (y == ignore_index) {
                out.push_back(-1);
            } else if (y < 0 || size_t(y) >= classes) {
                throw std::invalid_argument(std::string(what) + ": label " + std::to_string(y) + " at position " +
                                            std::to_string(out.size()) + " is outside 0 to " +
                                            std::to_string(classes - 1) + " (and is not ignore_index " +
                                            std::to_string(ignore_index) + ")");
            } else {
                out.push_back(y);
            }
        }
    return out;
}

#ifdef JZ_ID_KERNELS
// One block per column: the maximum, the log-sum-exp and the loss of that
// column (zeros when its label is skipped). `flagged`: the logits are stored
// with the transpose flag.
static __global__ void class_ce_forward_kernel(const float* z, const int32_t* y, float* maxima, float* lse,
                                               float* losses, size_t classes, size_t n, bool flagged) {
    __shared__ float shared[256];
    const size_t j = blockIdx.x;
    const unsigned t = threadIdx.x;
    const int32_t label = y[j];
    if (label < 0) {
        if (!t) { maxima[j] = 0; lse[j] = 0; losses[j] = 0; }
        return;
    }
    float maximum = -INFINITY;
    for (size_t c = t; c < classes; c += blockDim.x) maximum = fmaxf(maximum, z[flagged ? c * n + j : j * classes + c]);
    shared[t] = maximum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (t < stride) shared[t] = fmaxf(shared[t], shared[t + stride]);
        __syncthreads();
    }
    maximum = shared[0];
    __syncthreads();
    float total = 0;
    for (size_t c = t; c < classes; c += blockDim.x) total += expf(z[flagged ? c * n + j : j * classes + c] - maximum);
    shared[t] = total;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (t < stride) shared[t] += shared[t + stride];
        __syncthreads();
    }
    if (!t) {
        const float log_sum = logf(shared[0]);
        maxima[j] = maximum;
        lse[j] = log_sum;
        losses[j] = (maximum - z[flagged ? size_t(label) * n + j : j * classes + size_t(label)]) + log_sum;
    }
}
// dz (classes x n, plain) = w_j (softmax(z_j) - onehot(y_j))
static __global__ void class_ce_backward_kernel(float* dz, const float* z, const int32_t* y, const float* w,
                                                const float* maxima, const float* lse, size_t classes, size_t n,
                                                size_t count, bool flagged) {
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += size_t(blockDim.x) * gridDim.x) {
        const size_t c = k % classes, j = k / classes;
        const int32_t label = y[j];
        dz[k] = label < 0 ? 0.f
                          : w[j] * (expf((z[flagged ? c * n + j : k] - maxima[j]) - lse[j]) - (c == size_t(label) ? 1.f : 0.f));
    }
}
#endif

// value = sum_j w_j (logsumexp(z_j) - z_j[y_j]),  grad_j = w_j (softmax(z_j) - onehot(y_j));
// columns with y_j < 0 or w_j = 0 give zeros
template<class D>
LossResult<D> class_cross_entropy(const Matrix<D>& logits, const std::vector<int>& y, const std::vector<float>& w) {
    const size_t classes = logits.num_row(), n = logits.num_col();
    if constexpr (token_detail::on_device<D>) {
#ifdef JZ_ID_KERNELS
        using namespace token_detail;
        const Matrix<D> labels = device_words<D>(y);
        Matrix<float> host_w("w", 1, n);
        for (size_t j = 0; j < n; ++j) host_w.elem(0, j) = y[j] < 0 ? 0.f : w[j];
        const Matrix<D> weights(host_w);
        Matrix<D> maxima("ce_max", 1, n), lse("ce_lse", 1, n), losses("ce_loss", 1, n), grad("dlogits", classes, n);
        class_ce_forward_kernel<<<unsigned(n), 256>>>(floats(logits), words(labels), floats(maxima), floats(lse),
                                                      floats(losses), classes, n, logits.get_transpose());
        launched();
        const size_t count = classes * n;
        class_ce_backward_kernel<<<blocks(count), 256>>>(floats(grad), floats(logits), words(labels), floats(weights),
                                                         floats(maxima), floats(lse), classes, n, count,
                                                         logits.get_transpose());
        launched();
        return {sum(hadmd(weights, losses), 1), std::move(grad)};
#endif
    }
    if constexpr (std::is_same_v<D, float>) {
        Matrix<D> grad("dlogits", classes, n);
        double total = 0;
        for (size_t j = 0; j < n; ++j) {
            if (y[j] < 0 || w[j] == 0) {
                for (size_t c = 0; c < classes; ++c) grad.elem(c, j) = 0;
                continue;
            }
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < classes; ++c) maximum = std::max(maximum, logits.elem(c, j));
            double sum = 0;
            for (size_t c = 0; c < classes; ++c) sum += std::exp(double(logits.elem(c, j)) - maximum);
            const float log_sum = float(std::log(sum));
            // the target is taken from the shifted logits: exact for huge logits
            total += double(w[j]) * ((maximum - logits.elem(size_t(y[j]), j)) + log_sum);
            for (size_t c = 0; c < classes; ++c)
                grad.elem(c, j) = w[j] * (std::exp((logits.elem(c, j) - maximum) - log_sum) -
                                          (c == size_t(y[j]) ? 1.f : 0.f));
        }
        Matrix<D> value("loss", 1, 1);
        value.elem(0, 0) = float(total);
        return {std::move(value), std::move(grad)};
    } else {
        // Composition from existing device operations (the first Metal path;
        // CUDA and ROCm get kernels): a host-built one-hot and the weights.
        Matrix<float> one_hot = Matrix<float>::zeros(classes, n), weights("weights", 1, n);
        for (size_t j = 0; j < n; ++j) {
            weights.elem(0, j) = y[j] < 0 ? 0.f : w[j];
            if (y[j] >= 0) one_hot.elem(size_t(y[j]), j) = 1;
        }
        return masked_softmax_cross_entropy(logits, Matrix<D>(one_hot), Matrix<D>(weights));
    }
}

} // namespace loss_detail

// Softmax cross-entropy, averaged over the labels that are not ignored.
// When every label is ignored the value is 0 and the gradient is zero.
template<class D>
LossResult<D> softmax_cross_entropy(const Matrix<D>& logits, const Matrix<int>& labels, int ignore_index = -100) {
    if (!logits.num_row() || !logits.num_col())
        throw std::invalid_argument("softmax_cross_entropy: the logits are empty");
    const auto y = loss_detail::class_numbers(labels, logits.num_row(), logits.num_col(), ignore_index,
                                              "softmax_cross_entropy");
    size_t counted = 0;
    for (int label : y) counted += label >= 0;
    std::vector<float> w(y.size(), counted ? 1.f / float(counted) : 0.f);
    return loss_detail::class_cross_entropy(logits, y, w);
}

// Weighted sum without averaging, as the one-hot form: weights (1 x N) are
// finite, nonnegative and already contain any scaling.
template<class D>
LossResult<D> masked_softmax_cross_entropy(const Matrix<D>& logits, const Matrix<int>& labels,
                                           const Matrix<D>& weights, int ignore_index = -100) {
    if (!logits.num_row() || !logits.num_col())
        throw std::invalid_argument("masked_softmax_cross_entropy: the logits are empty");
    const size_t n = logits.num_col();
    if (weights.num_row() * weights.num_col() != n)
        throw std::invalid_argument("masked_softmax_cross_entropy: " +
                                    std::to_string(weights.num_row() * weights.num_col()) + " weights for " +
                                    std::to_string(n) + " columns of logits");
    const auto y = loss_detail::class_numbers(labels, logits.num_row(), n, ignore_index,
                                              "masked_softmax_cross_entropy");
    Matrix<float> host_weights = [&]() {
        if constexpr (std::is_same_v<D, float>) return weights;
        else return weights.to_host();
    }();
    std::vector<float> w(n);
    for (size_t j = 0; j < n; ++j) {
        w[j] = weights.num_row() == 1 ? host_weights.elem(0, j) : host_weights.elem(j, 0);
        if (!std::isfinite(w[j]) || w[j] < 0)
            throw std::invalid_argument("masked_softmax_cross_entropy: weights must be finite and nonnegative");
    }
    return loss_detail::class_cross_entropy(logits, y, w);
}

// Fraction of the counted columns whose largest score is not at the label.
template<class D>
Matrix<D> error_rate(const Matrix<D>& scores, const Matrix<int>& labels, int ignore_index = -100) {
    const auto y = loss_detail::class_numbers(labels, scores.num_row(), scores.num_col(), ignore_index, "error_rate");
    // the position of each column's maximum, found on the device; only the
    // N positions are read back (an evaluation metric: the caller reads it anyway)
    auto best = reduce(
        [] __GPU_CPU__(float* v, float* out, int count, int) {
            int at = 0;
            for (int i = 1; i < count; ++i)
                at = v[i] > v[at] ? i : at;
            out[0] = (float)at;
        }, scores, 0, 1);
    Matrix<float> positions = [&]() {
        if constexpr (std::is_same_v<D, float>) return best;
        else return best.to_host();
    }();
    size_t counted = 0, wrong = 0;
    for (size_t j = 0; j < y.size(); ++j) {
        if (y[j] < 0) continue;
        ++counted;
        wrong += size_t(positions.elem(0, j)) != size_t(y[j]);
    }
    Matrix<float> result("error_rate", 1, 1);
    result.elem(0, 0) = counted ? float(wrong) / float(counted) : 0.f;
    if constexpr (std::is_same_v<D, float>) return result;
    else return Matrix<D>(result);
}

} // namespace Juzhen
