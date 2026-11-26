/** Token layers (Embedding, positions, tied output), class-number losses and the default
 *  initialisation, for float and for the build's GPU type.
 *
 *  Test data and references are host matrices; device values are read back with to_host().
 *  Each suite also compares the device type with the float implementation on the same inputs.
 *  A failed check is printed and the run goes on, so one run lists every failure; any failure
 *  gives exit code 1. The convolution lines run in the suite of the type the build's ConvLayer uses.
 */
#include "../ml/embedding.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include "../ml/checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <vector>
using namespace Juzhen;
using H = Matrix<float>;   // host matrices: test data and references

static int checks = 0, failures = 0;
static void require(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "  [FAILED] " << what << std::endl;
    }
}
template <class D> static H host(const Matrix<D>& x) {
    if constexpr (std::is_same_v<D, float>) return x;
    else return x.to_host();
}
template <class D> static Matrix<D> dev(const H& x) {
    if constexpr (std::is_same_v<D, float>) return x;
    else return Matrix<D>(x);
}
// largest |a - b|; infinity when an entry is not finite
static float max_abs_diff(const H& a, const H& b) {
    if (a.num_row() != b.num_row() || a.num_col() != b.num_col()) throw std::runtime_error("shape differs");
    float worst = 0;
    for (size_t j = 0; j < a.num_col(); ++j)
        for (size_t i = 0; i < a.num_row(); ++i) {
            const float d = std::fabs(a.elem(i, j) - b.elem(i, j));
            if (!std::isfinite(d)) return std::numeric_limits<float>::infinity();
            worst = std::max(worst, d);
        }
    return worst;
}
static void close(const H& a, const H& b, const std::string& what, float tol = 3e-5f) {
    const float d = max_abs_diff(a, b);
    require(d <= tol, what + " max_abs=" + std::to_string(d));
}
template <class Fn> static std::string message(Fn f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "NO EXCEPTION";
}
static double stddev_of(const H& m) {
    double s = 0, s2 = 0; const size_t n = m.num_row() * m.num_col();
    for (size_t j = 0; j < m.num_col(); ++j) for (size_t i = 0; i < m.num_row(); ++i) { s += m.elem(i, j); s2 += double(m.elem(i, j)) * m.elem(i, j); }
    return std::sqrt(s2 / n - (s / n) * (s / n));
}
static H pattern(size_t rows, size_t cols, int salt) {
    H m("pattern", rows, cols);
    for (size_t j = 0; j < cols; ++j) for (size_t i = 0; i < rows; ++i) m.elem(i, j) = float(int((i * 17 + j * 13 + salt * 3) % 23) - 11) * .025f;
    return m;
}
// the same logical matrix kept with the transpose flag set (the other storage order)
template <class D> static Matrix<D> flagged(const H& m) {
    H t("t", m.num_col(), m.num_row());
    for (size_t j = 0; j < m.num_col(); ++j) for (size_t i = 0; i < m.num_row(); ++i) t.elem(j, i) = m.elem(i, j);
    Matrix<D> out = dev<D>(t).T();
    return out;
}
// seeds the generator that Matrix<D>::randn draws from
template <class D> static void reseed(int seed) {
    global_rand_gen.seed(seed);
#if defined(CUDA)
    // cuRAND's generator lives as long as its sampler, so one sampler is kept for the whole run
    static GPUSampler* sampler = new GPUSampler(1);
    if constexpr (!std::is_same_v<D, float>) sampler->setseed(seed);
#elif defined(ROCM_HIP) || defined(APPLE_SILICON)
    if constexpr (!std::is_same_v<D, float>) { GPUSampler sampler(seed); }
#endif
}
// does the build's ConvLayer hold matrices of type D?
template <class D> constexpr bool conv_is = std::is_base_of_v<Layer<D>, ConvLayer>;

// ---------------------------------------------------------------- 1. Embedding
template <class D> static void embedding_checks() {
    EmbeddingLayer<D> e(7, 3);
    require(e.W().num_row() == 7 && e.W().num_col() == 3, "table is vocab x dim");
    H table("table", 7, 3);
    for (size_t v = 0; v < 7; ++v) for (size_t i = 0; i < 3; ++i) table.elem(v, i) = float(10 * v + i) / 17;
    e.W() = dev<D>(table);
    Matrix<int> ids("ids", {{2, 2, 6, 0}});
    e.eval_ids(ids);
    H expected("lookup", 3, 4), up("up", 3, 4), gradient = H::zeros(7, 3);
    const int raw[4] = {2, 2, 6, 0};
    for (size_t j = 0; j < 4; ++j) for (size_t i = 0; i < 3; ++i) {
        expected.elem(i, j) = table.elem(raw[j], i);
        up.elem(i, j) = float(1 + i + 3 * j) / 13;
        gradient.elem(raw[j], i) += up.elem(i, j);
    }
    close(host(e.value()), expected, "embedding lookup", 0);
    auto refs = e.parameters();
    require(refs.size() == 1 && refs[0].name == "weight" && refs[0].decay, "one parameter named weight");
    AdamW<D> opt(refs);
    opt.zero_grad();
    auto returned = e.backward_ids(ids, dev<D>(up));
    close(host(returned), up, "backward returns the gradient of the embedded tokens", 0);
    close(host(refs[0].grad()), gradient, "repeated ids add up");
    e.backward_ids(ids, dev<D>(up));
    close(host(refs[0].grad()), gradient * 2, "a second backward accumulates");
    close(host(e.W()), table, "backward leaves the table alone", 0);
    // a seq_len x batch matrix is read one sequence (column) after another
    Matrix<int> grid("grid", 2, 2);
    grid.elem(0, 0) = 2; grid.elem(1, 0) = 2; grid.elem(0, 1) = 6; grid.elem(1, 1) = 0;
    e.eval_ids(grid);
    close(host(e.value()), expected, "seq_len x batch ids", 0);

    // the table and the upstream gradient stored with the transpose flag
    {
        EmbeddingLayer<D> f(7, 3);
        f.W() = flagged<D>(table);
        require(f.W().get_transpose(), "the table carries the transpose flag");
        f.eval_ids(ids);
        close(host(f.value()), expected, "lookup in a table stored with the transpose flag", 0);
        auto frefs = f.parameters();
        AdamW<D> fopt(frefs);
        fopt.zero_grad();
        auto back = f.backward_ids(ids, flagged<D>(up));
        close(host(back), up, "backward returns a flagged upstream unchanged", 0);
        close(host(frefs[0].grad()), gradient, "scatter-add from a flagged upstream", 0);
        f.backward_ids(ids, dev<D>(up));
        close(host(frefs[0].grad()), gradient * 2, "second scatter-add, plain upstream");
    }

    std::cout << "  bad id        : " << message([&] { e.eval_ids(Matrix<int>("ids", {{1, 7}})); }) << "\n";
    std::cout << "  negative id   : " << message([&] { e.eval_ids(Matrix<int>("ids", {{-1}})); }) << "\n";
    std::cout << "  float input   : " << message([&] { e.eval(Matrix<D>::ones(3, 2)); }) << "\n";
    std::cout << "  backward first: " << message([&] { e.backward_ids(ids, dev<D>(up)); }) << "\n";
    std::cout << "  jvp           : " << message([&] { e.jvp(dev<D>(up), dev<D>(up)); }) << "\n";
    // the messages above must be errors, and a failed eval must not leave an old lookup usable
    require(message([&] { e.eval_ids(Matrix<int>("ids", {{1, 7}})); }) != "NO EXCEPTION", "id == vocab is rejected");
    require(message([&] { e.eval_ids(Matrix<int>("ids", {{-1}})); }) != "NO EXCEPTION", "negative id is rejected");
    require(message([&] { e.backward_ids(ids, dev<D>(up)); }) != "NO EXCEPTION", "backward after a failed eval is rejected");
}

// ---------------------------------------------------------------- 2. PositionEmbedding
template <class D> static void position_checks() {
    PositionEmbeddingLayer<D> p(3, 2);
    H table("table", 3, 2);
    for (size_t t = 0; t < 3; ++t) for (size_t i = 0; i < 2; ++i) table.elem(t, i) = float(1 + t) + .1f * i;
    p.W() = dev<D>(table);
    H xh = pattern(2, 6, 1), uph = pattern(2, 6, 2);   // two sequences of three tokens
    Matrix<D> x = dev<D>(xh);
    p.eval(x);
    H expected("out", 2, 6), gradient = H::zeros(3, 2);
    for (size_t n = 0; n < 6; ++n) for (size_t i = 0; i < 2; ++i) {
        expected.elem(i, n) = xh.elem(i, n) + table.elem(n % 3, i);
        gradient.elem(n % 3, i) += uph.elem(i, n);
    }
    close(host(p.value()), expected, "positions added", 1e-6f);
    auto refs = p.parameters();
    AdamW<D> opt(refs);
    opt.zero_grad();
    close(host(p.backward(x, dev<D>(uph))), uph, "position backward passes the gradient through", 0);
    close(host(refs[0].grad()), gradient, "position gradient sums over sequences", 1e-6f);
    // any number of sequences
    Matrix<D> one = dev<D>(pattern(2, 3, 5));
    p.eval(one);
    require(p.value().num_col() == 3, "one sequence");
    // values after the column count changed (6 -> 3 -> 6)
    {
        H oneh = pattern(2, 3, 5), exp3("out", 2, 3);
        for (size_t n = 0; n < 3; ++n) for (size_t i = 0; i < 2; ++i) exp3.elem(i, n) = oneh.elem(i, n) + table.elem(n, i);
        close(host(p.value()), exp3, "positions added, one sequence", 1e-6f);
        p.eval(x);
        close(host(p.value()), expected, "positions added, back to two sequences", 1e-6f);
    }
    std::cout << "  wrong count   : " << message([&] { Matrix<D> bad = dev<D>(pattern(2, 4, 1)); p.eval(bad); }) << "\n";
}

// ---------------------------------------------------------------- 3. losses with class numbers
template <class D> static void loss_checks() {
    H zh("logits", 7, 4);
    for (size_t j = 0; j < 4; ++j) for (size_t i = 0; i < 7; ++i) zh.elem(i, j) = float(int((i * 3 + j * 5) % 17) - 8) * .3f;
    Matrix<int> y("y", {{1, 5, 6, 0}});
    H one_hot_h = H::zeros(7, 4);
    const int raw[4] = {1, 5, 6, 0};
    for (size_t j = 0; j < 4; ++j) one_hot_h.elem(raw[j], j) = 1;
    Matrix<D> z = dev<D>(zh), one_hot = dev<D>(one_hot_h);

    auto a = softmax_cross_entropy(z, y);
    auto b = softmax_cross_entropy(z, one_hot);
    const H ag = host(a.grad), bg = host(b.grad);
    const float av = host(a.value).elem(0, 0), bv = host(b.value).elem(0, 0);
    std::cout << "  class numbers vs one-hot: |value diff| = " << std::fabs(av - bv)
              << ", max|grad diff| = " << max_abs_diff(ag, bg) << "\n";
    require(std::fabs(av - bv) < 2e-6f, "value equals the one-hot loss");
    close(ag, bg, "gradient equals the one-hot loss", 2e-7f);

    // ignored labels: mean over the counted ones = masked loss with weights 1/count
    Matrix<int> yi("y", {{1, -100, 6, -100}});
    H wh("w", 1, 4);
    wh.elem(0, 0) = .5f; wh.elem(0, 1) = 0; wh.elem(0, 2) = .5f; wh.elem(0, 3) = 0;
    auto c = softmax_cross_entropy(z, yi);
    auto d = masked_softmax_cross_entropy(z, one_hot, dev<D>(wh));
    require(std::fabs(host(c.value).elem(0, 0) - host(d.value).elem(0, 0)) < 2e-6f, "ignored labels: value");
    close(host(c.grad), host(d.grad), "ignored labels: gradient", 2e-7f);
    // per-label weights with class numbers = the one-hot masked loss
    H w2h("w", 1, 4);
    w2h.elem(0, 0) = .25f; w2h.elem(0, 1) = 2; w2h.elem(0, 2) = 0; w2h.elem(0, 3) = 1;
    auto e = masked_softmax_cross_entropy(z, y, dev<D>(w2h));
    auto f = masked_softmax_cross_entropy(z, one_hot, dev<D>(w2h));
    require(std::fabs(host(e.value).elem(0, 0) - host(f.value).elem(0, 0)) < 5e-6f, "weights: value");
    close(host(e.grad), host(f.grad), "weights: gradient", 5e-7f);
    // nothing counted
    auto g = softmax_cross_entropy(z, Matrix<int>("y", {{-100, -100, -100, -100}}));
    require(host(g.value).elem(0, 0) == 0, "all ignored: value 0");
    close(host(g.grad), H::zeros(7, 4), "all ignored: zero gradient", 0);
    // huge logits
    H bigh(zh);
    bigh.elem(0, 0) = 10000; bigh.elem(1, 0) = 9999.75f; bigh.elem(2, 0) = 9999.5f;
    const double exact = std::log(1 + std::exp(-.25) + std::exp(-.5)) + .25;   // label 1 in column 0
    Matrix<D> big = dev<D>(bigh);
    auto h = softmax_cross_entropy(big, Matrix<int>("y", {{1, -100, -100, -100}}));
    auto h1 = softmax_cross_entropy(big, [&] { H o = H::zeros(7, 4); o.elem(1, 0) = 1; o.elem(0, 1) = 1; o.elem(0, 2) = 1; o.elem(0, 3) = 1; return dev<D>(o); }());
    const double hv = host(h.value).elem(0, 0);
    std::cout << "  logits near 10000: class-number error = " << std::fabs(hv - exact) << "\n";
    require(std::fabs(hv - exact) < 2e-6, "stable for huge logits");
    (void)h1;
    // printed, not checked: the same data through the two dense one-hot forms, against float64
    {
        H oh = H::zeros(7, 4); oh.elem(1, 0) = 1; oh.elem(0, 1) = 1; oh.elem(0, 2) = 1; oh.elem(0, 3) = 1;
        const int lab[4] = {1, 0, 0, 0};
        double mean_ref = 0;
        for (size_t j = 0; j < 4; ++j) {
            double mx = -1e300, total = 0;
            for (size_t i = 0; i < 7; ++i) mx = std::max(mx, double(bigh.elem(i, j)));
            for (size_t i = 0; i < 7; ++i) total += std::exp(double(bigh.elem(i, j)) - mx);
            mean_ref += (mx - double(bigh.elem(lab[j], j))) + std::log(total);
        }
        mean_ref /= 4;
        H w1 = H::zeros(1, 4); w1.elem(0, 0) = 1;
        auto m = masked_softmax_cross_entropy(big, dev<D>(oh), dev<D>(w1));
        std::cout << "  logits near 10000: dense one-hot mean error = " << std::fabs(double(host(h1.value).elem(0, 0)) - mean_ref)
                  << ", dense masked (weight 1 on column 0) error = " << std::fabs(double(host(m.value).elem(0, 0)) - exact) << "\n";
    }
    // seq_len x batch labels and the metric
    Matrix<int> grid("y", 2, 2);
    grid.elem(0, 0) = 1; grid.elem(1, 0) = 5; grid.elem(0, 1) = 6; grid.elem(1, 1) = 0;
    close(host(softmax_cross_entropy(z, grid).grad), ag, "seq_len x batch labels", 0);
    H scores_h = H::zeros(7, 4);
    scores_h.elem(1, 0) = 1; scores_h.elem(5, 1) = 1; scores_h.elem(2, 2) = 1; scores_h.elem(3, 3) = 1;   // two right, two wrong
    Matrix<D> scores = dev<D>(scores_h);
    require(std::fabs(host(error_rate(scores, y)).elem(0, 0) - .5f) < 1e-6f, "error_rate with class numbers");
    require(std::fabs(host(error_rate(scores, one_hot)).elem(0, 0) - .5f) < 1e-6f, "error_rate with one-hot labels");
    require(std::fabs(host(error_rate(scores, Matrix<int>("y", {{1, 5, -100, -100}}))).elem(0, 0)) < 1e-6f, "error_rate skips ignored labels");
    std::cout << "  label too big : " << message([&] { softmax_cross_entropy(z, Matrix<int>("y", {{1, 7, 0, 0}})); }) << "\n";
    std::cout << "  label count   : " << message([&] { softmax_cross_entropy(z, Matrix<int>("y", {{1, 2}})); }) << "\n";

    // logits stored with the transpose flag (what a transposed product gives)
    {
        Matrix<D> zt = flagged<D>(zh);
        auto t = softmax_cross_entropy(zt, y);
        require(std::fabs(host(t.value).elem(0, 0) - av) < 2e-6f, "flagged logits: value");
        close(host(t.grad), ag, "flagged logits: gradient", 2e-7f);
        auto ti = softmax_cross_entropy(zt, yi);
        require(std::fabs(host(ti.value).elem(0, 0) - host(c.value).elem(0, 0)) < 2e-6f, "flagged logits, ignored labels: value");
        close(host(ti.grad), host(c.grad), "flagged logits, ignored labels: gradient", 2e-7f);
    }
    // ignored columns get an exactly zero gradient, counted ones do not
    {
        const H cg = host(c.grad);
        float ignored = 0, counted = 0;
        for (size_t i = 0; i < 7; ++i) { ignored = std::max({ignored, std::fabs(cg.elem(i, 1)), std::fabs(cg.elem(i, 3))}); counted = std::max(counted, std::fabs(cg.elem(i, 0))); }
        require(ignored == 0 && counted > 0, "ignored columns have zero gradient");
    }
    // errors must be errors
    require(message([&] { softmax_cross_entropy(z, Matrix<int>("y", {{1, 7, 0, 0}})); }) != "NO EXCEPTION", "label == classes is rejected");
    require(message([&] { softmax_cross_entropy(z, Matrix<int>("y", {{1, 2}})); }) != "NO EXCEPTION", "wrong label count is rejected");
}

// ---------------------------------------------------------------- 4. a whole model
template <class D> struct TinyLM {
    EmbeddingLayer<D> token;
    PositionEmbeddingLayer<D> position;
    TransformerLayer<D> block;
    LayerNormLayer<D> norm;
    TiedLinearLayer<D> head;
    Sequential<D> net;
    explicit TinyLM(int batch = 2)
        : token(17, 8), position(4, 8), block(8, 8, 16, 4, batch, 2, true), norm(8), head(token),
          net{&head, &norm, &block, &position, &token} {}
};
template <class D> static void initialize(const std::vector<ParameterRef<D>>& refs) {
    for (size_t k = 0; k < refs.size(); ++k) {
        H v = pattern(refs[k].value->num_row(), refs[k].value->num_col(), int(k));
        if (refs[k].name.find("gamma") != std::string::npos) v = v + 1.0;
        *refs[k].value = dev<D>(v);
    }
}
template <class D> static float entry(const Matrix<D>& m, size_t i, size_t j) { return host(m).elem(i, j); }
template <class D> static void set_entry(Matrix<D>& m, size_t i, size_t j, float v) {
    H h = host(m);
    h.elem(i, j) = v;
    m = dev<D>(h);
}

template <class D> static void model_checks(const std::string& backend) {
    TinyLM<D> model;
    auto refs = model.net.parameters();
    std::cout << "  parameters    :";
    for (const auto& p : refs) std::cout << " " << p.name << "(" << p.value->num_row() << "x" << p.value->num_col() << (p.decay ? "" : ",no decay") << ")";
    std::cout << "\n";
    require(refs.size() == 17, "17 parameters: table, positions, 13 of the block, gamma, beta");
    require(refs[0].name == "layer0.weight" && refs[1].name == "layer1.weight", "names follow the layer index");
    for (const auto& p : refs) require(p.name.rfind("layer4.", 0) != 0, "the tied head registers nothing");
    AdamWOptions options{.002f, .9f, .97f, 1e-8f, .03f};
    AdamW<D> opt(refs, options);
    require(opt.parameters().size() == 17, "no duplicate registration");
    initialize(opt.parameters());

    Matrix<int> ids("ids", 4, 2), targets("targets", 4, 2);
    const int x[8] = {1, 1, 3, 7, 4, 5, 1, 2}, y[8] = {2, -100, 4, 1, 8, 7, -100, 6};
    for (int k = 0; k < 8; ++k) { ids.elem(k % 4, k / 4) = x[k]; targets.elem(k % 4, k / 4) = y[k]; }

    // Finite differences of the loss in entries of every parameter, on the
    // smooth part of the model (no ReLU kinks): table, positions, norm, tied head.
    {
        EmbeddingLayer<D> tok(17, 8); PositionEmbeddingLayer<D> pos(4, 8); LayerNormLayer<D> nrm(8); TiedLinearLayer<D> out(tok);
        Sequential<D> smooth{&out, &nrm, &pos, &tok};
        AdamW<D> o(smooth.parameters(), options);
        initialize(o.parameters());
        o.zero_grad();
        auto [loss, dlogits] = softmax_cross_entropy(smooth.eval(ids), targets);
        auto dx = smooth.backward(dlogits);
        require(dx.num_row() == 8 && dx.num_col() == 8, "backward returns dim x tokens");
        auto value_at = [&]() { return double(host(softmax_cross_entropy(smooth.eval(ids), targets).value).elem(0, 0)); };
        double worst = 0;
        int probes = 0;
        for (const auto& p : o.parameters()) {
            require(p.has_grad(), "every parameter has a gradient: " + p.name);
            const H analytic_all = host(p.grad());
            for (int probe = 0; probe < 8; ++probe, ++probes) {
                const size_t i = (probe * 5 + 1) % p.value->num_row(), j = (probe * 3) % p.value->num_col();
                const float saved = entry(*p.value, i, j), eps = 1e-2f;
                set_entry(*p.value, i, j, saved + eps); const double up = value_at();
                set_entry(*p.value, i, j, saved - eps); const double down = value_at();
                set_entry(*p.value, i, j, saved);
                const double numeric = (up - down) / (2 * eps), analytic = analytic_all.elem(i, j);
                const double err = std::fabs(numeric - analytic) / std::max(2e-2, std::fabs(numeric) + std::fabs(analytic));
                worst = std::max(worst, err);
                require(err < 1e-2, "gradient check " + p.name + " numeric=" + std::to_string(numeric) + " analytic=" + std::to_string(analytic));
            }
        }
        std::cout << "  finite differences, " << probes << " entries of table, positions, gamma, beta: worst error " << worst << "\n";
        // the table gradient of an untied model is the sum of the returned columns of each id
        EmbeddingLayer<D> t(17, 8); LinearLayer<D> lin(17, 8, 8);
        Sequential<D> untied{&lin, &t};
        AdamW<D> ou(untied.parameters(), options);
        initialize(ou.parameters());
        ou.zero_grad();
        auto r = softmax_cross_entropy(untied.eval(ids), targets);
        const H returned = host(untied.backward(std::move(r.grad)));
        H summed = H::zeros(17, 8);
        for (int k = 0; k < 8; ++k) for (size_t i = 0; i < 8; ++i) summed.elem(x[k], i) += returned.elem(i, k);
        close(host(ou.parameters()[0].grad()), summed, "table gradient = returned columns summed per id", 1e-7f);
    }
    opt.zero_grad();
    for (const auto& p : opt.parameters()) require(!p.has_grad(), "zero_grad");
    {
        auto r = softmax_cross_entropy(model.net.eval(ids), targets);
        model.net.backward(std::move(r.grad));
        for (const auto& p : opt.parameters()) require(p.has_grad(), "every parameter of the full model has a gradient: " + p.name);
    }

    // three training steps: every parameter is updated once per step
    for (int step = 0; step < 3; ++step) {
        opt.zero_grad();
        auto r = softmax_cross_entropy(model.net.eval(ids), targets);
        model.net.backward(std::move(r.grad));
        opt.clip_grad_norm(.35);
        require(opt.step() == 17, "one update per parameter");
    }

    // checkpoint: the network's own list is enough
    const std::string file = "tiny_" + backend + ".ckpt";
    TrainingProgress progress{2, 3, 24}, read;
    std::string error;
    require(save_checkpoint(model.net.layers(), file, progress, &error), "save: " + error);
    TinyLM<D> restored;
    AdamW<D> resumed(restored.net.parameters(), options);
    require(load_checkpoint(restored.net.layers(), file, read, &error), "load: " + error);
    require(read.step == 3 && read.epoch == 2 && read.data_position == 24, "progress restored");
    for (TinyLM<D>* m : {&model, &restored}) {
        AdamW<D>& o = m == &model ? opt : resumed;
        o.zero_grad();
        auto r = softmax_cross_entropy(m->net.eval(ids), targets);
        m->net.backward(std::move(r.grad));
        o.clip_grad_norm(.35);
        o.step();
    }
    for (size_t k = 0; k < opt.parameters().size(); ++k)
        close(host(*opt.parameters()[k].value), host(*resumed.parameters()[k].value), "resumed " + opt.parameters()[k].name, 0);
    std::cout << "  checkpoint of network.layers(): resumed step compared bitwise on " << backend << "\n";

    // gradient accumulation: a batch of two = two micro-batches of one, with
    // unequal counted labels (3 and 2), weighted by 1 / (all counted labels)
    TinyLM<D> full(2), micro(1);
    AdamW<D> large(full.net.parameters(), options), small(micro.net.parameters(), options);
    initialize(large.parameters());
    for (size_t k = 0; k < large.parameters().size(); ++k) *small.parameters()[k].value = *large.parameters()[k].value;
    large.zero_grad(); small.zero_grad();
    auto whole = softmax_cross_entropy(full.net.eval(ids), targets);
    full.net.backward(std::move(whole.grad));
    for (int b = 0; b < 2; ++b) {
        Matrix<int> xb("x", 4, 1), yb("y", 4, 1);
        H wb("w", 1, 4);
        for (int t = 0; t < 4; ++t) { xb.elem(t, 0) = x[4 * b + t]; yb.elem(t, 0) = y[4 * b + t]; wb.elem(0, t) = 1.f / 6; }
        auto part = masked_softmax_cross_entropy(micro.net.eval(xb), yb, dev<D>(wb));
        micro.net.backward(std::move(part.grad));
    }
    float accumulation_worst = 0;
    for (size_t k = 0; k < large.parameters().size(); ++k) {
        const H a = host(large.parameters()[k].grad()), b = host(small.parameters()[k].grad());
        accumulation_worst = std::max(accumulation_worst, max_abs_diff(a, b));
        close(a, b, "accumulated " + large.parameters()[k].name, 6e-5f);
    }
    std::cout << "  accumulation over unequal micro-batches against the full batch: worst |diff| " << accumulation_worst << "\n";

    // structure errors
    EmbeddingLayer<D> tok(17, 8); LinearLayer<D> lin(8, 8, 1); TiedLinearLayer<D> tied(tok);
    std::cout << "  embedding not first : " << message([&] { Sequential<D> bad{&tok, &lin}; }) << "\n";
    std::cout << "  tied without owner  : " << message([&] { Sequential<D> bad{&tied, &lin}; }) << "\n";
    Sequential<D> floats{&lin};
    std::cout << "  ids into floats     : " << message([&] { floats.eval(ids); }) << "\n";
    Matrix<D> numbers = Matrix<D>::ones(8, 8);
    std::cout << "  floats into ids     : " << message([&] { model.net.eval(numbers); }) << "\n";
    std::list<Layer<D>*> orphan{&tied, &lin};
    std::cout << "  save tied w/o owner : " << message([&] { TrainingProgress p; save_checkpoint(orphan, "bad_" + backend + ".ckpt", p); }) << "\n";
    // the same five must be refused (save_checkpoint reports through its return value)
    require(message([&] { Sequential<D> bad{&tok, &lin}; }) != "NO EXCEPTION", "embedding not first is rejected");
    require(message([&] { Sequential<D> bad{&tied, &lin}; }) != "NO EXCEPTION", "tied without owner is rejected");
    require(message([&] { floats.eval(ids); }) != "NO EXCEPTION", "ids into a float network are rejected");
    require(message([&] { model.net.eval(numbers); }) != "NO EXCEPTION", "floats into an id network are rejected");
    {
        TrainingProgress p;
        std::string why;
        bool saved = true;
        try { saved = save_checkpoint(orphan, "bad_" + backend + ".ckpt", p, &why); } catch (const std::exception& e) { saved = false; why = e.what(); }
        std::cout << "  save tied w/o owner : returned " << (saved ? "true" : "false") << ", reason: " << why << "\n";
        require(!saved, "a checkpoint of a tied layer without its Embedding is refused");
    }

    // checkpoint mismatch reasons
    TinyLM<D> other(1);
    require(load_checkpoint(other.net.layers(), file, read, &error), "another batch size loads: " + error);
    EmbeddingLayer<D> t2(17, 8); PositionEmbeddingLayer<D> p2(4, 8); TransformerLayer<D> b2(8, 8, 16, 4, 2, 2, true); LayerNormLayer<D> n2(8); TiedLinearLayer<D> h2(t2);
    EmbeddingLayer<D> t3(19, 8); PositionEmbeddingLayer<D> p3(4, 8); TiedLinearLayer<D> h3(t3);
    std::list<Layer<D>*> fewer{&h2, &b2, &p2, &t2}, wrong_vocab{&h3, &n2, &b2, &p3, &t3}, wrong_type{&h2, &b2, &n2, &p2, &t2};
    require(!load_checkpoint(fewer, file, read, &error), "fewer layers: load refused");            std::cout << "  ckpt, fewer layers  : " << error << "\n";
    require(!load_checkpoint(wrong_vocab, file, read, &error), "other vocab: load refused");       std::cout << "  ckpt, other vocab   : " << error << "\n";
    require(!load_checkpoint(wrong_type, file, read, &error), "other order: load refused");        std::cout << "  ckpt, other order   : " << error << "\n";
}

// ---------------------------------------------------------------- 5. initialisation and LayerNorm
static void conv_init_checks();   // below; uses the build's ConvLayer type

template <class D> static void init_checks() {
    auto draws = [](const char* name, const std::function<void()>& make, const std::function<void()>& replay) {
        reseed<D>(5); make();   const float a = host(Matrix<D>::randn(1, 1)).elem(0, 0);
        reseed<D>(5); replay(); const float b = host(Matrix<D>::randn(1, 1)).elem(0, 0);
        std::cout << "  draws " << name << (a == b ? ": MATCH" : ": DIFFERENT") << "\n";
        require(a == b, std::string("draws ") + name);
    };
    auto R = [](size_t m, size_t n) { (void)Matrix<D>::randn(m, n); };
    draws("Layer/LinearLayer/ReluLayer(4,3,2) == randn(4,3)", [] { Layer<D> a(4, 3, 2); }, [&] { R(4, 3); });
    draws("LinearLayer(4,3,2, Init::normal(0)) == nothing", [] { LinearLayer<D> a(4, 3, 2, Init::normal(0)); }, [] {});
    draws("TransformerLayer(4,4,8,2,1,2) == (4,4)x4, (8,4), (4,8)", [] { TransformerLayer<D> t(4, 4, 8, 2, 1, 2, true); },
          [&] { R(4, 4); R(4, 4); R(4, 4); R(4, 4); R(8, 4); R(4, 8); });
    if constexpr (conv_is<D>)
        draws("ConvLayer(2,1,4,4,2,3,3,1,1) == randn(18,1)", [] { ConvLayer c(2, 1, 4, 4, 2, 3, 3, 1, 1); }, [&] { R(18, 1); });
    else
        std::cout << "  draws ConvLayer: not in this suite (the build's ConvLayer holds another element type)\n";
    draws("Embedding(7,3) == randn(7,3)", [] { EmbeddingLayer<D> e(7, 3); }, [&] { R(7, 3); });
    {   // the original keeps this Embedding in a function-local static; here it is an ordinary local,
        // so that no matrix outlives Juzhen::shutdown()
        EmbeddingLayer<D> owner(7, 3);
        draws("TiedLinear == nothing", [&] { reseed<D>(5); TiedLinearLayer<D> h(owner); }, [] {});
    }
    // the check can tell a difference: one extra draw must change the next number
    {
        reseed<D>(5); R(4, 3);          const float a = host(Matrix<D>::randn(1, 1)).elem(0, 0);
        reseed<D>(5); R(4, 3); R(1, 1); const float b = host(Matrix<D>::randn(1, 1)).elem(0, 0);
        require(a != b, "an extra draw changes the stream (the draws check is sensitive)");
    }

    reseed<D>(7);
    LinearLayer<D> lin(256, 128, 1); ReluLayer<D> rel(256, 128, 1); Layer<D> tanh_layer(256, 128, 1);
    LinearLayer<D> old(256, 128, 1, Init::normal(.001f));
    EmbeddingLayer<D> emb(1000, 64); PositionEmbeddingLayer<D> pos(64, 64);
    const double s_lin = stddev_of(host(lin.W())), s_tanh = stddev_of(host(tanh_layer.W())), s_rel = stddev_of(host(rel.W())),
                 s_old = stddev_of(host(old.W())), s_emb = stddev_of(host(emb.W())), s_pos = stddev_of(host(pos.W()));
    const float bias_max = max_abs_diff(host(lin.b()), H::zeros(256, 1));
    std::cout << "  std Linear(256,128)        " << s_lin << "  (1/sqrt(128) = " << std::sqrt(1. / 128) << "), bias max " << bias_max << "\n";
    std::cout << "  std Layer tanh (256,128)   " << s_tanh << "\n";
    std::cout << "  std ReluLayer(256,128)     " << s_rel << "  (sqrt(2/128) = " << std::sqrt(2. / 128) << ")\n";
    std::cout << "  std Init::normal(.001)     " << s_old << "\n";
    if constexpr (conv_is<D>) conv_init_checks();
    std::cout << "  std Embedding, Position    " << s_emb << ", " << s_pos << "\n";
    // the original only prints these; here they must be within 5 % (32768 draws: about 0.4 % noise; 4096 for positions)
    auto within = [](double got, double want, double rel) { return std::fabs(got - want) <= rel * want; };
    require(within(s_lin, std::sqrt(1. / 128), .05) && within(s_tanh, std::sqrt(1. / 128), .05), "Linear and tanh layers start at N(0, 1/fan_in)");
    require(within(s_rel, std::sqrt(2. / 128), .05), "a ReLU layer starts at N(0, 2/fan_in)");
    require(within(s_old, .001, .05), "Init::normal(.001) gives std .001");
    require(within(s_emb, .02, .05) && within(s_pos, .02, .08), "Embedding and PositionEmbedding start at N(0, 0.02^2)");
    require(bias_max == 0, "biases start at zero");

    // a deep ReLU network no longer starts dead: output spread over 64 inputs
    for (int depth : {4, 8}) {
        reseed<D>(11);
        std::vector<std::unique_ptr<Layer<D>>> owned;
        std::list<Layer<D>*> net;
        owned.push_back(std::make_unique<ReluLayer<D>>(256, 32, 64)); net.push_front(owned.back().get());
        for (int k = 1; k < depth; ++k) { owned.push_back(std::make_unique<ReluLayer<D>>(256, 256, 64)); net.push_front(owned.back().get()); }
        owned.push_back(std::make_unique<LinearLayer<D>>(10, 256, 64)); net.push_front(owned.back().get());
        auto x = Matrix<D>::randn(32, 64);
        const auto& out = forward(net, x);
        std::cout << "  " << depth << " hidden ReLU layers: output std " << stddev_of(host(out)) << "\n";
    }

    // LayerNorm follows the number of columns it is given
    LayerNormLayer<D> ln(5);
    auto refs = ln.parameters();
    AdamW<D> opt(refs);
    // the float layer with the same gamma and beta gives the reference values
    LayerNormLayer<float> ref_ln(5);
    auto ref_refs = ref_ln.parameters();
    AdamW<float> ref_opt(ref_refs);
    for (size_t k = 0; k < refs.size(); ++k) {
        H v = pattern(refs[k].value->num_row(), refs[k].value->num_col(), int(k) + 4);
        if (refs[k].name.find("gamma") != std::string::npos) v = v + 1.0;
        *refs[k].value = dev<D>(v);
        *ref_refs[k].value = v;
    }
    float ln_worst = 0;
    for (size_t n : {4, 6, 1, 6}) {
        H xh = pattern(5, n, int(n)) * 7.0, uph = pattern(5, n, 3), tangent_h = pattern(5, n, 9);
        Matrix<D> x = dev<D>(xh);
        ln.eval(x);
        require(ln.value().num_col() == n, "LayerNorm output columns");
        opt.zero_grad();
        auto dx = ln.backward(x, dev<D>(uph));
        require(dx.num_col() == n && refs[0].has_grad(), "LayerNorm backward at this size");
        // values, input gradient, parameter gradients and tangent against the float layer
        ref_ln.eval(xh);
        ref_opt.zero_grad();
        const H ref_dx = ref_ln.backward(xh, H(uph));
        const std::string at = " with " + std::to_string(n) + " columns";
        const float d_out = max_abs_diff(host(ln.value()), ref_ln.value()), d_dx = max_abs_diff(host(dx), ref_dx);
        float d_par = 0;
        for (size_t k = 0; k < refs.size(); ++k) d_par = std::max(d_par, max_abs_diff(host(refs[k].grad()), ref_refs[k].grad()));
        const float d_jvp = max_abs_diff(host(ln.jvp(x, dev<D>(tangent_h))), ref_ln.jvp(xh, tangent_h));
        ln_worst = std::max({ln_worst, d_out, d_dx, d_par, d_jvp});
        require(d_out <= 2e-5f, "LayerNorm output equals the float layer" + at + " max_abs=" + std::to_string(d_out));
        require(d_dx <= 2e-5f, "LayerNorm input gradient equals the float layer" + at + " max_abs=" + std::to_string(d_dx));
        require(d_par <= 2e-5f, "LayerNorm gamma/beta gradients equal the float layer" + at + " max_abs=" + std::to_string(d_par));
        require(d_jvp <= 2e-5f, "LayerNorm jvp equals the float layer" + at + " max_abs=" + std::to_string(d_jvp));
    }
    std::cout << "  LayerNormLayer(5) ran with 4, 6, 1 and 6 columns; worst |diff| to the float layer " << ln_worst << "\n";
    std::cout << "  LayerNorm wrong rows: " << message([&] { Matrix<D> bad = dev<D>(pattern(4, 3, 1)); ln.eval(bad); }) << "\n";
    require(message([&] { Matrix<D> bad = dev<D>(pattern(4, 3, 1)); ln.eval(bad); }) != "NO EXCEPTION", "LayerNorm rejects a wrong feature count");
}

// the convolution lines of the original init checks, for the element type of the build's ConvLayer
static void conv_init_checks() {
    ConvLayer conv(2, 16, 8, 8, 32, 3, 3, 1, 1, true), conv_plain(2, 16, 8, 8, 32, 3, 3, 1, 1, false);
    convtransLayer up(2, 16, 8, 8, 8, 4, 4, 1, 2, true);
    const double s_conv = stddev_of(host(conv.W())), s_plain = stddev_of(host(conv_plain.W())), s_up = stddev_of(host(up.W()));
    std::cout << "  std Conv 16->32 k3 relu    " << s_conv << "  (sqrt(2/144) = " << std::sqrt(2. / 144) << "), without relu " << s_plain << "  (sqrt(1/144) = " << std::sqrt(1. / 144) << ")\n";
    std::cout << "  std ConvTrans 16->8 k4 s2  " << s_up << "  (sqrt(2/(16*16/4)) = " << std::sqrt(2. / 64) << ")\n";
    // 4608 and 2048 draws: about 1 % and 1.6 % noise
    auto within = [](double got, double want, double rel) { return std::fabs(got - want) <= rel * want; };
    require(within(s_conv, std::sqrt(2. / 144), .06) && within(s_plain, std::sqrt(1. / 144), .06), "convolution starts at N(0, 2/fan_in) with relu, N(0, 1/fan_in) without");
    require(within(s_up, std::sqrt(2. / 64), .08), "transposed convolution starts at N(0, 2 stride^2 / (C_in kH kW))");
    require(max_abs_diff(host(conv.b()), H::zeros(32, 1)) == 0 && max_abs_diff(host(up.b()), H::zeros(8, 1)) == 0, "convolution biases start at zero");
}

// ---------------------------------------------------------------- 6. the device type against float
// The same inputs through Matrix<D> and through Matrix<float>: lookup and scatter-add must be equal
// bit for bit (the proposal's claim), losses and the whole model within float32 rounding.
template <class D> static void parity_checks() {
    const size_t vocab = 257, dim = 24, tokens = 96;
    H table("table", vocab, dim), up("up", dim, tokens);
    for (size_t v = 0; v < vocab; ++v) for (size_t i = 0; i < dim; ++i) table.elem(v, i) = std::sin(.37f * float(v) + 1.3f * float(i)) * .7f;
    for (size_t j = 0; j < tokens; ++j) for (size_t i = 0; i < dim; ++i) up.elem(i, j) = std::cos(.11f * float(j * dim + i)) * 1e-2f * float(1 + (i + j) % 7);
    Matrix<int> ids("ids", 12, 8);   // 8 sequences of 12 tokens, many repeats
    size_t distinct = 0;
    {
        std::vector<bool> seen(vocab, false);
        for (size_t k = 0; k < tokens; ++k) {
            const int id = int((k * k * 7 + k * 3) % 41 * 6 % vocab);
            ids.elem(k % 12, k / 12) = id;
            if (!seen[id]) { seen[id] = true; ++distinct; }
        }
    }
    EmbeddingLayer<float> cpu(vocab, dim);
    EmbeddingLayer<D> gpu(vocab, dim);
    cpu.W() = table; gpu.W() = dev<D>(table);
    auto cpu_refs = cpu.parameters(); auto gpu_refs = gpu.parameters();
    AdamW<float> cpu_opt(cpu_refs); AdamW<D> gpu_opt(gpu_refs);
    cpu.eval_ids(ids); gpu.eval_ids(ids);
    close(host(gpu.value()), cpu.value(), "lookup equals the float loop bit for bit (257 x 24 table, 96 tokens)", 0);
    cpu_opt.zero_grad(); gpu_opt.zero_grad();
    cpu.backward_ids(ids, H(up)); gpu.backward_ids(ids, dev<D>(up));
    close(host(gpu_refs[0].grad()), cpu_refs[0].grad(), "scatter-add equals the float loop bit for bit", 0);
    cpu.backward_ids(ids, H(up)); gpu.backward_ids(ids, dev<D>(up));
    close(host(gpu_refs[0].grad()), cpu_refs[0].grad(), "scatter-add twice equals the float loop twice bit for bit", 0);
    std::cout << "  lookup and scatter-add against the float loops: " << tokens << " tokens over " << distinct << " distinct ids, max|diff| "
              << max_abs_diff(host(gpu.value()), cpu.value()) << " and " << max_abs_diff(host(gpu_refs[0].grad()), cpu_refs[0].grad()) << "\n";

    // class-number losses: 257 classes, 96 columns, 22 labels ignored
    H zh("logits", vocab, tokens), wh("w", 1, tokens);
    Matrix<int> labels("labels", 12, 8);
    size_t counted = 0;
    for (size_t j = 0; j < tokens; ++j) {
        for (size_t i = 0; i < vocab; ++i) zh.elem(i, j) = 3.f * std::sin(.013f * float(i * 31 + j * 17)) + .002f * float(i % 13);
        const bool skip = j % 9 == 2 || j % 9 == 7;
        labels.elem(j % 12, j / 12) = skip ? -100 : int((j * 37 + 11) % vocab);
        counted += !skip;
        wh.elem(0, j) = float(j % 5) * .01f;
    }
    Matrix<D> z = dev<D>(zh);
    auto mean_cpu = softmax_cross_entropy(zh, labels);
    auto mean_gpu = softmax_cross_entropy(z, labels);
    auto sum_cpu = masked_softmax_cross_entropy(zh, labels, wh);
    auto sum_gpu = masked_softmax_cross_entropy(z, labels, dev<D>(wh));
    const float dv = std::fabs(host(mean_gpu.value).elem(0, 0) - mean_cpu.value.elem(0, 0)), dg = max_abs_diff(host(mean_gpu.grad), mean_cpu.grad);
    const float dv2 = std::fabs(host(sum_gpu.value).elem(0, 0) - sum_cpu.value.elem(0, 0)), dg2 = max_abs_diff(host(sum_gpu.grad), sum_cpu.grad);
    std::cout << "  class-number loss against the float loops (" << vocab << " classes, " << counted << " of " << tokens << " labels counted): value "
              << mean_cpu.value.elem(0, 0) << ", |value diff| " << dv << ", max|grad diff| " << dg << "; weighted: value " << sum_cpu.value.elem(0, 0)
              << ", |value diff| " << dv2 << ", max|grad diff| " << dg2 << "\n";
    require(dv <= 1e-5f, "class-number loss value equals the float loop, |diff| " + std::to_string(dv));
    require(dg <= 1e-6f, "class-number loss gradient equals the float loop, max_abs " + std::to_string(dg));
    require(dv2 <= 1e-5f, "weighted class-number loss value equals the float loop, |diff| " + std::to_string(dv2));
    require(dg2 <= 1e-6f, "weighted class-number loss gradient equals the float loop, max_abs " + std::to_string(dg2));
    const float de = std::fabs(host(error_rate(z, labels)).elem(0, 0) - error_rate(zh, labels).elem(0, 0));
    require(de == 0, "error_rate with class numbers equals the float result");

    // the whole model with the same weights: loss and every parameter gradient
    TinyLM<float> cpu_model;
    TinyLM<D> gpu_model;
    AdamWOptions options{.002f, .9f, .97f, 1e-8f, .03f};
    AdamW<float> co(cpu_model.net.parameters(), options);
    AdamW<D> go(gpu_model.net.parameters(), options);
    initialize(co.parameters()); initialize(go.parameters());
    Matrix<int> x("ids", 4, 2), y("targets", 4, 2);
    const int xs[8] = {1, 1, 3, 7, 4, 5, 1, 2}, ys[8] = {2, -100, 4, 1, 8, 7, -100, 6};
    for (int k = 0; k < 8; ++k) { x.elem(k % 4, k / 4) = xs[k]; y.elem(k % 4, k / 4) = ys[k]; }
    float worst_grad = 0, worst_weight = 0, loss_diff = 0;
    std::string worst_name;
    for (int step = 0; step < 5; ++step) {
        co.zero_grad(); go.zero_grad();
        auto rc = softmax_cross_entropy(cpu_model.net.eval(x), y);
        auto rg = softmax_cross_entropy(gpu_model.net.eval(x), y);
        loss_diff = std::max(loss_diff, std::fabs(host(rg.value).elem(0, 0) - rc.value.elem(0, 0)));
        cpu_model.net.backward(std::move(rc.grad));
        gpu_model.net.backward(std::move(rg.grad));
        for (size_t k = 0; k < co.parameters().size(); ++k) {
            const float d = max_abs_diff(host(go.parameters()[k].grad()), co.parameters()[k].grad());
            if (d > worst_grad) { worst_grad = d; worst_name = co.parameters()[k].name; }
        }
        co.clip_grad_norm(.35); go.clip_grad_norm(.35);
        co.step(); go.step();
        for (size_t k = 0; k < co.parameters().size(); ++k)
            worst_weight = std::max(worst_weight, max_abs_diff(host(*go.parameters()[k].value), *co.parameters()[k].value));
    }
    std::cout << "  whole model against float over 5 steps: worst |loss diff| " << loss_diff << ", worst |gradient diff| " << worst_grad << " (" << worst_name
              << "), worst |weight diff| " << worst_weight << "\n";
    require(loss_diff <= 2e-5f, "model loss equals the float model over 5 steps, |diff| " + std::to_string(loss_diff));
    require(worst_grad <= 2e-5f, "model gradients equal the float model over 5 steps, max_abs " + std::to_string(worst_grad));
    require(worst_weight <= 2e-4f, "model weights equal the float model after 5 steps, max_abs " + std::to_string(worst_weight));
}

template <class D> static bool suite(const std::string& backend) {
    checks = failures = 0;
    reseed<D>(1);  // on CUDA this also creates the generator the layer constructors draw from
    std::cout << "===== " << backend << " =====\n";
    auto section = [&](const char* name, const std::function<void()>& run) {
        std::cout << name << std::endl;
        try { run(); } catch (const std::exception& e) {
            ++failures;
            std::cout << "  [FAILED] " << name << " stopped by an exception: " << e.what() << std::endl;
        }
    };
    reseed<D>(919);
    section("embedding", [] { embedding_checks<D>(); });
    section("positions", [] { position_checks<D>(); });
    section("losses", [] { loss_checks<D>(); });
    section("model", [&] { model_checks<D>(backend); });
    section("init", [] { init_checks<D>(); });
    if constexpr (!std::is_same_v<D, float>) section("device against float [added]", [] { parity_checks<D>(); });
    if (failures) std::cout << "[FAIL] " << backend << " token layers: " << failures << " of " << checks << " checks failed\n";
    else std::cout << "[PASS] " << backend << " token layers: " << checks << " checks\n";
    return failures == 0;
}

int compute() {
    try {
        bool ok = suite<float>("CPU");
#ifdef CUDA
        ok = suite<CUDAfloat>("CUDA") && ok;
#endif
#ifdef ROCM_HIP
        ok = suite<ROCMfloat>("ROCm") && ok;
#endif
#ifdef APPLE_SILICON
        ok = suite<MPSfloat>("Metal") && ok;
#endif
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
