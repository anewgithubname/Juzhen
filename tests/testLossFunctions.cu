/** Loss functions (ml/loss.hpp): FP64 references, finite differences, edge cases and training. */
#include "../ml/layer.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include <functional>
#include <iostream>
#include <limits>

using namespace Juzhen;

template<class D> Matrix<float> host(const Matrix<D>& x) {
    if constexpr (std::is_same_v<D, float>) return x;
    else return x.to_host();
}

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static void close(const Matrix<float>& actual, const Matrix<float>& expected,
                  const std::string& message, double tolerance) {
    require(actual.num_row() == expected.num_row() && actual.num_col() == expected.num_col(), message + " shape");
    double worst = 0;
    for (size_t j = 0; j < actual.num_col(); ++j)
        for (size_t i = 0; i < actual.num_row(); ++i) {
            const double error = std::abs(double(actual.elem(i, j)) - expected.elem(i, j));
            require(std::isfinite(error), message + " finite");
            worst = std::max(worst, error);
        }
    require(worst <= tolerance, message + " error=" + std::to_string(worst));
}

static double scalar(const Matrix<float>& m) { return m.elem(0, 0); }

template<class F> void rejects(F function, const std::string& message) {
    bool caught = false;
    try { function(); } catch (const std::logic_error&) { caught = true; }
    require(caught, message);
}

// ── Independent FP64 references ─────────────────────────────────────────────

static double log_sum_exp(const Matrix<float>& x, size_t j) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < x.num_row(); ++i) maximum = std::max(maximum, double(x.elem(i, j)));
    double total = 0;
    for (size_t i = 0; i < x.num_row(); ++i) total += std::exp(double(x.elem(i, j)) - maximum);
    return std::log(total) + maximum;
}

static double dot(const Matrix<float>& a, const Matrix<float>& b, size_t j) {
    double result = 0;
    for (size_t i = 0; i < a.num_row(); ++i) result += double(a.elem(i, j)) * b.elem(i, j);
    return result;
}

static double reference_mse(const Matrix<float>& p, const Matrix<float>& t) {
    double total = 0;
    for (size_t j = 0; j < p.num_col(); ++j)
        for (size_t i = 0; i < p.num_row(); ++i) {
            const double e = double(p.elem(i, j)) - t.elem(i, j);
            total += e * e;
        }
    return total / p.num_col();
}

static double reference_ce(const Matrix<float>& x, const Matrix<float>& y) {
    double total = 0;
    for (size_t j = 0; j < x.num_col(); ++j) total += log_sum_exp(x, j) - dot(x, y, j);
    return total / x.num_col();
}

static double reference_masked_ce(const Matrix<float>& x, const Matrix<float>& y, const Matrix<float>& w) {
    double total = 0;
    for (size_t j = 0; j < x.num_col(); ++j) total += w.elem(0, j) * (log_sum_exp(x, j) - dot(x, y, j));
    return total;
}

// central finite differences of `f` with respect to every entry of `x`
static Matrix<float> finite_differences(Matrix<float> x, const std::function<double(const Matrix<float>&)>& f) {
    Matrix<float> result("finite differences", x.num_row(), x.num_col());
    for (size_t j = 0; j < x.num_col(); ++j)
        for (size_t i = 0; i < x.num_row(); ++i) {
            const float original = x.elem(i, j);
            x.elem(i, j) = original + .001f;
            const double upper = x.elem(i, j), plus = f(x);
            x.elem(i, j) = original - .001f;
            const double lower = x.elem(i, j), minus = f(x);
            x.elem(i, j) = original;
            result.elem(i, j) = float((plus - minus) / (upper - lower));
        }
    return result;
}

// ── Test data ───────────────────────────────────────────────────────────────

constexpr size_t classes = 5, batch = 4;

static Matrix<float> make_logits() {
    Matrix<float> logits("logits", classes, batch);
    for (size_t j = 0; j < batch; ++j)
        for (size_t i = 0; i < classes; ++i) logits.elem(i, j) = .65f * std::sin(float(3 * i + 7 * j));
    return logits;
}

// one-hot columns, except column 2 which is a (normalized) soft target
static Matrix<float> make_labels() {
    Matrix<float> labels = Matrix<float>::zeros(classes, batch);
    for (size_t j = 0; j < batch; ++j) labels.elem((j * 2 + 1) % classes, j) = 1;
    labels.elem(0, 2) = .25f;
    labels.elem(1, 2) = .75f;
    return labels;
}

// ── Checks ──────────────────────────────────────────────────────────────────

template<class D> void mse_checks() {
    auto ph = make_logits();
    Matrix<float> th("target", classes, batch);
    for (size_t j = 0; j < batch; ++j)
        for (size_t i = 0; i < classes; ++i) th.elem(i, j) = .3f * std::cos(float(i + 2 * j));
    Matrix<D> p(ph), t(th);
    auto [value, grad] = mse(p, t);
    require(std::abs(scalar(host(value)) - reference_mse(ph, th)) < 3e-6, "mse value");
    auto expected = finite_differences(ph, [&](const Matrix<float>& x) { return reference_mse(x, th); });
    close(host(grad), expected, "mse finite differences", 2e-5);
    close(host(p), ph, "mse preserves prediction", 0);
    rejects([&] { mse(p, Matrix<D>::ones(classes, batch - 1)); }, "mse rejects shape mismatch");
}

template<class D> void softmax_ce_checks() {
    auto xh = make_logits(), yh = make_labels();
    Matrix<D> x(xh), y(yh);
    auto [value, grad] = softmax_cross_entropy(x, y);
    require(std::abs(scalar(host(value)) - reference_ce(xh, yh)) < 3e-6, "softmax CE value");
    auto expected = finite_differences(xh, [&](const Matrix<float>& z) { return reference_ce(z, yh); });
    close(host(grad), expected, "softmax CE finite differences", 2e-6);
    close(host(x), xh, "softmax CE preserves logits", 0);
    rejects([&] { softmax_cross_entropy(x, Matrix<D>::ones(classes - 1, batch)); },
            "softmax CE rejects shape mismatch");

    // very large logits must not overflow exp()
    Matrix<float> large("large", classes, batch);
    for (size_t j = 0; j < batch; ++j)
        for (size_t i = 0; i < classes; ++i) large.elem(i, j) = 1000.f * float(i) - 2000.f;
    auto [large_value, large_grad] = softmax_cross_entropy(Matrix<D>(large), y);
    require(std::abs(scalar(host(large_value)) - reference_ce(large, yh)) < .001, "softmax CE large logits");
    auto large_gradient = host(large_grad);
    for (size_t j = 0; j < batch; ++j)
        for (size_t i = 0; i < classes; ++i)
            require(std::isfinite(large_gradient.elem(i, j)), "softmax CE large logits finite gradient");
}

template<class D> void masked_ce_checks() {
    auto xh = make_logits(), yh = make_labels();
    Matrix<float> wh("weights", 1, batch);
    wh.elem(0, 0) = 0; wh.elem(0, 1) = .125f; wh.elem(0, 2) = .75f; wh.elem(0, 3) = 2;
    Matrix<D> x(xh), y(yh), w(wh);

    auto [value, grad] = masked_softmax_cross_entropy(x, y, w);
    require(std::abs(scalar(host(value)) - reference_masked_ce(xh, yh, wh)) < 3e-6, "masked CE value");
    auto expected = finite_differences(xh, [&](const Matrix<float>& z) { return reference_masked_ce(z, yh, wh); });
    auto gradient = host(grad);
    close(gradient, expected, "masked CE finite differences", 2e-6);
    for (size_t i = 0; i < classes; ++i) require(gradient.elem(i, 0) == 0, "zero weight gives zero gradient");

    auto [zero_value, zero_grad] = masked_softmax_cross_entropy(x, y, Matrix<D>::zeros(1, batch));
    require(scalar(host(zero_value)) == 0, "all-zero weights give zero loss");
    close(host(zero_grad), Matrix<float>::zeros(classes, batch), "all-zero weights give zero gradient", 0);

    rejects([&] { masked_softmax_cross_entropy(x, y, Matrix<D>::ones(batch, 1)); }, "masked CE rejects weight shape");
    rejects([&] { masked_softmax_cross_entropy(x, Matrix<D>::ones(classes, batch - 1), w); },
            "masked CE rejects label shape");
    for (float invalid : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto bad = wh;
        bad.elem(0, 0) = invalid;
        rejects([&] { masked_softmax_cross_entropy(x, y, Matrix<D>(bad)); }, "masked CE rejects invalid weights");
    }
}

template<class D> void error_rate_checks() {
    // columns 0, 1 and 3 are classified correctly, column 2 is not
    Matrix<float> scores("scores", 3, 4), labels = Matrix<float>::zeros(3, 4);
    const float s[3][4] = {{3, 0, 0, -1}, {1, 2, 5, 0}, {0, 1, 1, 4}};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) scores.elem(i, j) = s[i][j];
    labels.elem(0, 0) = 1; labels.elem(1, 1) = 1; labels.elem(2, 2) = 1; labels.elem(2, 3) = 1;
    require(std::abs(scalar(host(error_rate(Matrix<D>(scores), Matrix<D>(labels)))) - .25) < 1e-6, "error rate");
    rejects([&] { error_rate(Matrix<D>(scores), Matrix<D>::ones(3, 3)); }, "error rate rejects shape mismatch");
}

// a small softmax classifier trained with the explicit API must make progress
template<class D> void training_checks() {
    LinearLayer<D> layer(classes, 3, batch);
    Sequential<D> network{&layer};
    AdamWOptions options;
    options.lr = .05f;
    AdamW<D> optimizer(network.parameters(), options);
    auto input = Matrix<D>::randn(3, batch);
    Matrix<D> labels(make_labels());

    double first = 0, last = 0;
    for (int step = 0; step < 100; ++step) {
        optimizer.zero_grad();
        auto [loss, grad] = softmax_cross_entropy(network.eval(input), labels);
        last = scalar(host(loss));
        if (step == 0) first = last;
        network.backward(std::move(grad));
        optimizer.step();
    }
    require(last < first * .5, "training reduces softmax CE: " + std::to_string(first) + " -> " + std::to_string(last));
}

// Column maxima and clamping run as device kernels on Metal; check both
// layouts and NaN propagation against the host reference.
template<class D> void column_max_and_clamp_checks() {
    Matrix<float> h("h", 4, 3);
    for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < 4; ++i) h.elem(i, j) = std::sin(1.3f * float(i) + 0.7f * float(j)) * 4.f;
    h.elem(2, 1) = -1e30f;
    for (bool transposed : {false, true}) {
        Matrix<float> storage("storage", 3, 4);
        for (size_t j = 0; j < 3; ++j)
            for (size_t i = 0; i < 4; ++i) storage.elem(j, i) = h.elem(i, j);
        Matrix<D> x = transposed ? Matrix<D>(storage).T() : Matrix<D>(h);
        auto maxima = host(loss_detail::column_max(x));
        require(maxima.num_row() == 1 && maxima.num_col() == 3, "column max shape");
        for (size_t j = 0; j < 3; ++j) {
            float expected = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < 4; ++i) expected = std::max(expected, h.elem(i, j));
            require(maxima.elem(0, j) == expected, "column max value");
        }
        auto clamped = host(clamp(x, -1.f, 2.f));
        for (size_t j = 0; j < 3; ++j)
            for (size_t i = 0; i < 4; ++i)
                require(clamped.elem(i, j) == std::min(std::max(h.elem(i, j), -1.f), 2.f), "clamp value");
        close(host(x), h, "clamp preserves its input", 0);
    }
    Matrix<float> nan("nan", 1, 1);
    nan.elem(0, 0) = std::numeric_limits<float>::quiet_NaN();
    require(std::isnan(host(clamp(Matrix<D>(nan), -1.f, 1.f)).elem(0, 0)), "clamp keeps NaN");
}

template<class D> void checks(const char* backend) {
    column_max_and_clamp_checks<D>();
    mse_checks<D>();
    softmax_ce_checks<D>();
    masked_ce_checks<D>();
    error_rate_checks<D>();
    training_checks<D>();
    std::cout << "[PASS] " << backend << " loss functions\n";
}

int compute() {
    try {
        global_rand_gen.seed(927);
        checks<float>("CPU");
#ifdef CUDA
        GPUSampler sampler(927);
        checks<CUDAfloat>("CUDA");
        CudaErrorCheck(cudaDeviceSynchronize());
#endif
#ifdef ROCM_HIP
        checks<ROCMfloat>("ROCm");
#endif
#ifdef APPLE_SILICON
        GPUSampler sampler(927);
        checks<MPSfloat>("Metal");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
