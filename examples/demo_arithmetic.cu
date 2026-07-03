/**
 * @file demo_arithmetic.cu
 * @brief Transformer that learns integer addition, char by char.
 *
 * Each problem is rendered as a fixed-width string whose width is set by
 * n_digits (shown here for 3-digit operands):
 *
 *     "012+037=9400"
 *      \__/ \__/ \__/
 *       a    b   sum, digits REVERSED (least-significant first)
 *
 * Why reverse the answer? A causal transformer emits the answer left-to-right,
 * one digit at a time. Written normally (most-significant first) the very first
 * output digit depends on carries from all lower digits -- it would have to look
 * ahead. Reversed, each answer digit depends only on the aligned operand digits
 * plus the carry from the digit it JUST produced, which matches causal decoding.
 * This is the standard trick that makes addition learnable for such models.
 *
 * Training: teacher-forced next-char prediction over the whole string (reusing
 *           the framework's TransformerLayer + LogisticLayer), warmup + cosine
 *           LR schedule, keeping the checkpoint with the best validation
 *           exact-match. The trained model is saved under res/ and reloaded on
 *           later runs (delete the file or set ARITH_RETRAIN=1 to retrain).
 * Eval:     BATCHED AUTOREGRESSIVE decoding -- the model consumes its own
 *           predicted digits -- scored as exact-match on HELD-OUT (a,b) pairs
 *           never seen in training, plus carry-chain robustness buckets and an
 *           edge-case showcase. So the reported accuracy is real generalization
 *           to unseen sums, not memorization.
 *
 * Empirical notes from scaling this demo (see git history):
 *  - Required capacity grows quickly with n_digits: 3 digits train fine with
 *    d_model=128 / 2 blocks, 4 digits need 256 / 3, 5 digits need 512 / 4.
 *    An undersized model plateaus at a loss above the entropy floor with one
 *    or more answer digits collapsed to a constant guess, and more training
 *    steps do NOT rescue it; a sufficient model shows a sharp phase transition
 *    to ~100% within the first few thousand steps.
 *  - The logged loss cannot reach 0: operand digits are random, so its floor
 *    is (#operand digit positions x ln 10)/seq_len -- e.g. 9*ln(10)/17 = 1.219
 *    nats/position for 5 digits. Loss at the floor means the task is solved.
 *  - Generalization holds across unseen pairs and every carry-chain length,
 *    but NOT far outside the training distribution: with operands sampled
 *    uniformly over VALUES, near-all-zero problems like 1+1 (probability
 *    ~1e-8) came out wrong -- the learned circuit is an algorithm, but an
 *    interpolative one. The demo therefore samples operands with a UNIFORM
 *    DIGIT-LENGTH (L ~ U{1..n_digits}, then a value of exactly that length),
 *    which covers every zero-padding pattern; the edge-case showcase
 *    verifies 1+1.
 */

#include "../ml/layer.hpp"
#include "../cpp/juzhen.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
using namespace Juzhen;

#ifdef CUDA
#define FLOAT CUDAfloat
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
#else
#define FLOAT float
#endif

// Fixed vocabulary: digits 0-9, '+', '='.
static const string VOCAB = "0123456789+=";

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

static void copy_tf_weights(TransformerLayer<FLOAT>& dst,
                            const TransformerLayer<FLOAT>& src) {
    dst.set_Wq(Matrix<FLOAT>(as_host(src.get_Wq())));
    dst.set_Wk(Matrix<FLOAT>(as_host(src.get_Wk())));
    dst.set_Wv(Matrix<FLOAT>(as_host(src.get_Wv())));
    dst.set_Wo(Matrix<FLOAT>(as_host(src.get_Wo())));
    dst.set_bo(Matrix<FLOAT>(as_host(src.get_bo())));
    dst.set_W1(Matrix<FLOAT>(as_host(src.get_W1())));
    dst.set_b1(Matrix<FLOAT>(as_host(src.get_b1())));
    dst.set_W2(Matrix<FLOAT>(as_host(src.get_W2())));
    dst.set_b2(Matrix<FLOAT>(as_host(src.get_b2())));
    dst.set_ln1_gamma(Matrix<FLOAT>(as_host(src.get_ln1_gamma())));
    dst.set_ln1_beta(Matrix<FLOAT>(as_host(src.get_ln1_beta())));
    dst.set_ln2_gamma(Matrix<FLOAT>(as_host(src.get_ln2_gamma())));
    dst.set_ln2_beta(Matrix<FLOAT>(as_host(src.get_ln2_beta())));
}

static void copy_linear_weights(LinearLayer<FLOAT>& dst, LinearLayer<FLOAT>& src) {
    dst.W() = Matrix<FLOAT>(as_host(src.W()));
    dst.b() = Matrix<FLOAT>(as_host(src.b()));
}

// ── Model (de)serialization ──────────────────────────────────────────────────
// dumpweights()/loadweights() only handle the base-class weights, which are
// dummies for TransformerLayer, so the demo serializes all parameters itself.
static void write_mat(FILE* fp, const Matrix<float>& m) {
    const int32_t r = (int32_t)m.num_row(), c = (int32_t)m.num_col();
    fwrite(&r, sizeof(int32_t), 1, fp);
    fwrite(&c, sizeof(int32_t), 1, fp);
    for (int32_t j = 0; j < c; ++j)
        for (int32_t i = 0; i < r; ++i) {
            const float v = m.elem(i, j);
            fwrite(&v, sizeof(float), 1, fp);
        }
}

static bool read_mat(FILE* fp, Matrix<float>& out) {
    int32_t r = 0, c = 0;
    if (fread(&r, sizeof(int32_t), 1, fp) != 1 ||
        fread(&c, sizeof(int32_t), 1, fp) != 1 || r <= 0 || c <= 0)
        return false;
    Matrix<float> m("loaded", r, c);
    for (int32_t j = 0; j < c; ++j)
        for (int32_t i = 0; i < r; ++i) {
            float v;
            if (fread(&v, sizeof(float), 1, fp) != 1) return false;
            m.elem(i, j) = v;
        }
    out = std::move(m);
    return true;
}

static int ipow10(int e) { int r = 1; while (e-- > 0) r *= 10; return r; }

// Render "a..a+b..b=<reversed sum>" for the given operands (zero-padded).
static string make_example(int a, int b, int n_digits, int ans_digits) {
    string s;
    for (int i = n_digits - 1; i >= 0; --i) s += char('0' + (a / ipow10(i)) % 10);
    s += '+';
    for (int i = n_digits - 1; i >= 0; --i) s += char('0' + (b / ipow10(i)) % 10);
    s += '=';
    const int sum = a + b;
    for (int i = 0; i < ans_digits; ++i) s += char('0' + (sum / ipow10(i)) % 10); // reversed
    return s;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(1);
#endif
    global_rand_gen.seed(1);

    // ── Problem definition ──────────────────────────────────────────────
    const int n_digits   = 5;                       // operands in [0, 99999]
    const int ans_digits = 6;                       // sum in [0, 199998]
    const int prompt_len = 2 * n_digits + 2;        // "a..a+b..b=" length
    const int L          = prompt_len + ans_digits; // full string length
    const int seq_len    = L - 1;                   // next-char prediction positions

    const int V = (int)VOCAB.size();
    int c2i[256] = {};
    vector<char> idx_to_char(VOCAB.begin(), VOCAB.end());
    for (int i = 0; i < V; ++i) c2i[(unsigned char)VOCAB[i]] = i;
    const int in_dim = V + seq_len;  // one-hot char + one-hot absolute position

    // ── Model / training config ─────────────────────────────────────────
    // Sized for n_digits = 5; smaller problems train with smaller models
    // (3 digits: 128d/2 blocks, 4 digits: 256d/3 blocks) -- see file header.
    const int d_model = 512, d_k = 512, d_ff = 1024, num_heads = 8, num_blocks = 4;
    const int batch = 256;
    const int steps = 12000, log_every = 1000;
    const int N = seq_len * batch;

    const float peak_lr = 1e-3f, min_lr = 1e-4f;
    const int   warmup_steps = 300;
    const double PI = 3.14159265358979323846;
    auto lr_at = [&](int step) -> float {
        if (step < warmup_steps) return peak_lr * (float)(step + 1) / (float)warmup_steps;
        const double t = (double)(step - warmup_steps) / (double)(steps - warmup_steps);
        return (float)(min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + std::cos(PI * t)));
    };

    const int max_operand = ipow10(n_digits);
    // long long: at 5+ digits a*max_operand+b overflows 32-bit int
    auto key = [&](int a, int b) { return (long long)a * max_operand + b; };

    // Operand sampler uniform over digit-lengths: L ~ U{1..n_digits}, then a
    // value OF THAT LENGTH (leading digit nonzero for L >= 2; L = 1 includes
    // 0), so every length is exactly equally likely. Under plain uniform-value
    // sampling, short zero-padded operands ("00001") have probability
    // ~10^-(n_digits-1) and are effectively never trained on, so the model
    // fails 1+1 despite ~100% in-distribution accuracy (see git history).
    // `rnd` (uniform over all values) is kept for the carry-chain stress
    // buckets, which need mostly-long operands.
    uniform_int_distribution<int> rnd(0, max_operand - 1);
    uniform_int_distribution<int> rnd_len(1, n_digits);
    auto sample_operand = [&]() {
        const int len = rnd_len(global_rand_gen);
        const int lo = len == 1 ? 0 : ipow10(len - 1);
        return uniform_int_distribution<int>(lo, ipow10(len) - 1)(global_rand_gen);
    };

    // ── Held-out test set of unseen (a,b) pairs ─────────────────────────
    // Sampled from the same length-uniform distribution as training.
    const int test_size = 4096;
    unordered_set<long long> test_keys;
    vector<pair<int,int>> test_pairs;
    while ((int)test_pairs.size() < test_size) {
        int a = sample_operand(), b = sample_operand();
        if (test_keys.insert(key(a, b)).second) test_pairs.push_back({a, b});
    }

    // ── Adversarial carry-chain slices ──────────────────────────────────
    // Bucket unseen pairs by longest consecutive carry run when adding
    // (0 = carry-free, n_digits = carry ripples through every position).
    // Excluded from training via test_keys, like the main test set.
    const int n_buckets = n_digits + 1;
    vector<vector<pair<int,int>>> carry_sets(n_buckets);
    auto max_carry_run = [&](int a, int b) {
        int run = 0, best = 0, carry = 0;
        for (int i = 0; i < n_digits; ++i) {
            const int s = (a / ipow10(i)) % 10 + (b / ipow10(i)) % 10 + carry;
            carry = s >= 10 ? 1 : 0;
            run = carry ? run + 1 : 0;
            best = run > best ? run : best;
        }
        return best;
    };
    {
        int remaining = n_buckets;
        while (remaining > 0) {
            const int a = rnd(global_rand_gen), b = rnd(global_rand_gen);
            const int L = max_carry_run(a, b);
            if ((int)carry_sets[L].size() >= batch) continue;
            if (!test_keys.insert(key(a, b)).second) continue;
            carry_sets[L].push_back({a, b});
            if ((int)carry_sets[L].size() == batch) --remaining;
        }
    }

    cout << "=== Transformer Arithmetic Demo (learning a+b) ===\n";
    cout << "up-to-" << n_digits << "-digit operands (digit-length ~ U{1.." << n_digits
         << "}), format e.g. \"" << make_example(12, 37, n_digits, ans_digits)
         << "\" (answer digits reversed)\n";
    cout << "Vocabulary (" << V << "): " << VOCAB << "\n";
    cout << "Network: embed(" << in_dim << "->" << d_model << ") -> Transformer("
         << d_model << "," << d_k << "," << d_ff << "," << num_heads << "h) [x" << num_blocks
         << "] -> proj(" << d_model << "->" << V << ")\n";
    cout << "Training: mini-batch=" << batch << " x seq_len=" << seq_len << ", steps=" << steps
         << "; held-out test pairs=" << test_size << "\n\n";

    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < num_blocks; ++i)
            v.push_back(make_unique<TF>(d_model, d_k, d_ff, seq_len, b, num_heads));
        return v;
    };

    LinearLayer<FLOAT> embed(d_model, in_dim, N);
    auto blocks = make_blocks(batch);
    LinearLayer<FLOAT> proj(V, d_model, N);

    LinearLayer<FLOAT> best_embed(d_model, in_dim, N);
    auto best_blocks = make_blocks(batch);
    LinearLayer<FLOAT> best_proj(V, d_model, N);

    auto build_net = [&](vector<unique_ptr<TF>>& bl,
                         LinearLayer<FLOAT>& emb, LinearLayer<FLOAT>& pr) {
        list<Layer<FLOAT>*> net;
        net.push_back(&pr);
        for (int i = num_blocks - 1; i >= 0; --i) net.push_back(bl[i].get());
        net.push_back(&emb);
        return net;
    };
    list<Layer<FLOAT>*> trainnn = build_net(blocks, embed, proj);

    auto save_best = [&]() {
        copy_linear_weights(best_embed, embed);
        copy_linear_weights(best_proj, proj);
        for (int i = 0; i < num_blocks; ++i) copy_tf_weights(*best_blocks[i], *blocks[i]);
    };
    auto restore_best = [&]() {
        copy_linear_weights(embed, best_embed);
        copy_linear_weights(proj, best_proj);
        for (int i = 0; i < num_blocks; ++i) copy_tf_weights(*blocks[i], *best_blocks[i]);
    };
    save_best();

    // Save/load every parameter of embed + all transformer blocks + proj.
    // Header records the config so a stale file can't be loaded silently.
    const char MODEL_MAGIC[9] = "ARITHTF1";
    auto save_model = [&](const string& path) {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) { cerr << "Cannot write model to " << path << "\n"; return; }
        const int32_t cfg[8] = {n_digits, ans_digits, d_model, d_k,
                                d_ff, num_heads, num_blocks, seq_len};
        fwrite(MODEL_MAGIC, 1, 8, fp);
        fwrite(cfg, sizeof(int32_t), 8, fp);
        write_mat(fp, as_host(embed.W()));
        write_mat(fp, as_host(embed.b()));
        for (auto& blk : blocks) {
            write_mat(fp, as_host(blk->get_Wq()));
            write_mat(fp, as_host(blk->get_Wk()));
            write_mat(fp, as_host(blk->get_Wv()));
            write_mat(fp, as_host(blk->get_Wo()));
            write_mat(fp, as_host(blk->get_bo()));
            write_mat(fp, as_host(blk->get_W1()));
            write_mat(fp, as_host(blk->get_b1()));
            write_mat(fp, as_host(blk->get_W2()));
            write_mat(fp, as_host(blk->get_b2()));
            write_mat(fp, as_host(blk->get_ln1_gamma()));
            write_mat(fp, as_host(blk->get_ln1_beta()));
            write_mat(fp, as_host(blk->get_ln2_gamma()));
            write_mat(fp, as_host(blk->get_ln2_beta()));
        }
        write_mat(fp, as_host(proj.W()));
        write_mat(fp, as_host(proj.b()));
        fclose(fp);
    };
    auto load_model = [&](const string& path) -> bool {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return false;
        char magic[8] = {0};
        int32_t cfg[8] = {0};
        const int32_t want[8] = {n_digits, ans_digits, d_model, d_k,
                                 d_ff, num_heads, num_blocks, seq_len};
        bool ok = fread(magic, 1, 8, fp) == 8 &&
                  memcmp(magic, MODEL_MAGIC, 8) == 0 &&
                  fread(cfg, sizeof(int32_t), 8, fp) == 8 &&
                  memcmp(cfg, want, sizeof(cfg)) == 0;
        if (!ok) {
            if (magic[0]) cerr << "Ignoring " << path << ": config mismatch\n";
            fclose(fp);
            return false;
        }
        Matrix<float> m("m", 1, 1);
        auto rd = [&](auto&& setter) {
            if (ok && (ok = read_mat(fp, m))) setter(Matrix<FLOAT>(m));
        };
        rd([&](Matrix<FLOAT>&& w) { embed.W() = std::move(w); });
        rd([&](Matrix<FLOAT>&& w) { embed.b() = std::move(w); });
        for (auto& blk : blocks) {
            rd([&](const Matrix<FLOAT>& w) { blk->set_Wq(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_Wk(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_Wv(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_Wo(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_bo(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_W1(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_b1(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_W2(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_b2(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_ln1_gamma(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_ln1_beta(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_ln2_gamma(w); });
            rd([&](const Matrix<FLOAT>& w) { blk->set_ln2_beta(w); });
        }
        rd([&](Matrix<FLOAT>&& w) { proj.W() = std::move(w); });
        rd([&](Matrix<FLOAT>&& w) { proj.b() = std::move(w); });
        fclose(fp);
        if (!ok) cerr << "Ignoring " << path << ": truncated file\n";
        return ok;
    };

    auto set_all_lr = [&](float lr) {
        embed.adamWstate().alpha = lr; embed.adambstate().alpha = lr;
        proj.adamWstate().alpha  = lr; proj.adambstate().alpha  = lr;
        for (int i = 0; i < num_blocks; ++i) blocks[i]->set_lr(lr);
    };

    // Teacher-forced next-char batch from a list of (a,b) pairs.
    auto fill_batch = [&](const vector<pair<int,int>>& probs, Matrix<float>& X, Matrix<float>& Y) {
        X.zeros(); Y.zeros();
        for (int b = 0; b < (int)probs.size(); ++b) {
            const string S = make_example(probs[b].first, probs[b].second, n_digits, ans_digits);
            for (int s = 0; s < seq_len; ++s) {
                const int col = b * seq_len + s;
                X.elem(c2i[(unsigned char)S[s]], col) = 1.0f;   // current char
                X.elem(V + s, col) = 1.0f;                       // position feature
                Y.elem(c2i[(unsigned char)S[s + 1]], col) = 1.0f;// next char
            }
        }
    };

    // Batched autoregressive solve: feed the model its own digits.
    // Returns {per-digit accuracy %, exact-match %}; optionally fills predicted sums.
    auto eval_solve = [&](const vector<pair<int,int>>& probs, vector<int>* out_pred) {
        const int B = (int)probs.size();
        auto X = Matrix<float>::zeros(in_dim, seq_len * B);
        vector<string> S(B);
        for (int b = 0; b < B; ++b) {
            S[b] = make_example(probs[b].first, probs[b].second, n_digits, ans_digits);
            for (int s = 0; s < seq_len; ++s) X.elem(V + s, b * seq_len + s) = 1.0f;
            for (int s = 0; s < prompt_len; ++s)
                X.elem(c2i[(unsigned char)S[b][s]], b * seq_len + s) = 1.0f;
        }
        vector<string> pred(B);
        for (int k = 0; k < ans_digits; ++k) {
            auto net = build_net(blocks, embed, proj);
            forward(net, Matrix<FLOAT>(X));
            auto logits = as_host(proj.value());
            const int out_pos = prompt_len - 1 + k;   // predicts answer digit k
            const int in_pos  = prompt_len + k;        // where to feed it back
            for (int b = 0; b < B; ++b) {
                const int col = b * seq_len + out_pos;
                int best = 0;
                for (int r = 1; r < V; ++r)
                    if (logits.elem(r, col) > logits.elem(best, col)) best = r;
                const char c = best < 10 ? char('0' + best) : '#'; // answer must be a digit
                pred[b] += c;
                if (best < 10 && in_pos <= seq_len - 1)
                    X.elem(best, b * seq_len + in_pos) = 1.0f;
            }
        }
        int exact = 0, dcorrect = 0, dtotal = 0;
        for (int b = 0; b < B; ++b) {
            const string truth = S[b].substr(prompt_len);   // reversed true answer
            bool all = true;
            for (int i = 0; i < ans_digits; ++i) {
                dtotal++;
                if (pred[b][i] == truth[i]) dcorrect++; else all = false;
            }
            if (all) exact++;
            if (out_pred) {
                int val = 0, mul = 1;
                for (int i = 0; i < ans_digits; ++i) {
                    if (pred[b][i] >= '0' && pred[b][i] <= '9') val += (pred[b][i] - '0') * mul;
                    mul *= 10;
                }
                out_pred->push_back(val);
            }
        }
        return make_pair(100.0f * dcorrect / dtotal, 100.0f * exact / B);
    };

    // Fixed eval batches: a train slice (freshly sampled, excluded from test) and
    // the first `batch` held-out test pairs.
    vector<pair<int,int>> train_eval(batch), val_eval(test_pairs.begin(), test_pairs.begin() + batch);
    for (auto& p : train_eval) {
        int a, b;
        do { a = sample_operand(); b = sample_operand(); } while (test_keys.count(key(a, b)));
        p = {a, b};
    }

    auto X_h = Matrix<float>::zeros(in_dim, N);
    auto Y_h = Matrix<float>::zeros(V, N);
    vector<pair<int,int>> offs(batch);

    // Reuse a previously trained model when one with this exact config exists.
    // Delete the file or set ARITH_RETRAIN=1 to train from scratch.
    // "_ulen" marks weights trained on the length-uniform operand distribution
    // (incompatible with models trained on uniform-value sampling).
    const string model_path = string(PROJECT_DIR) + "/res/arithmetic_tf_"
        + to_string(n_digits) + "digit_" + to_string(d_model) + "d_"
        + to_string(num_blocks) + "blk_ulen.bin";
    const bool force_retrain = getenv("ARITH_RETRAIN") != nullptr;
    if (!force_retrain && load_model(model_path)) {
        cout << "Loaded trained model from " << model_path
             << "\n(set ARITH_RETRAIN=1 to train from scratch)\n";
    } else {

    float best_em = -1.0f;
    int   best_step = 0;

    for (int step = 0; step < steps; ++step) {
        set_all_lr(lr_at(step));

        for (int b = 0; b < batch; ++b) {
            int a, bb;
            do { a = sample_operand(); bb = sample_operand(); } while (test_keys.count(key(a, bb)));
            offs[b] = {a, bb};
        }
        fill_batch(offs, X_h, Y_h);

        auto X = Matrix<FLOAT>(X_h);
        forward(trainnn, X);
        LogisticLayer<FLOAT> loss(N, Matrix<FLOAT>(Y_h));
        trainnn.push_front(&loss);
        backprop(trainnn, X);
        trainnn.pop_front();

        if (step % log_every == 0 || step == steps - 1) {
            // Mean next-char NLL (nats/position) of this batch, pre-update
            // logits. Its floor is not 0: the operand digits themselves are
            // unpredictable (with uniform-value operands it is exactly
            // (#operand digit positions * ln 10)/seq_len; length-uniform
            // operands have less entropy, so the floor sits a bit lower).
            loss.eval(proj.value());
            const float nll = item(as_host(loss.value()));
            auto [tr_dig, tr_em] = eval_solve(train_eval, nullptr);
            auto [va_dig, va_em] = eval_solve(val_eval, nullptr);
            // >= : on ties (e.g. repeated 100% on the small val slice) keep the
            // LATEST checkpoint -- more training on rare cases, lower LR.
            const bool improved = va_em >= best_em;
            if (improved) { best_em = va_em; best_step = step; save_best(); }
            printf("step %5d   loss = %.4f   train_exact = %5.1f%%   val_exact = %5.1f%%   "
                   "val_digit = %5.1f%%   lr = %.2e%s\n",
                   step, nll, tr_em, va_em, va_dig, lr_at(step), improved ? "   *best" : "");
            fflush(stdout);
        }
    }

    restore_best();
    printf("\nBest val exact-match %.1f%% at step %d\n", best_em, best_step);

    save_model(model_path);
    cout << "Saved trained model to " << model_path << "\n";

    } // end train-from-scratch branch

    // ── Final held-out evaluation over the full test set (in batches) ────
    int total_exact = 0, total = 0;
    for (int i = 0; i + batch <= (int)test_pairs.size(); i += batch) {
        vector<pair<int,int>> chunk(test_pairs.begin() + i, test_pairs.begin() + i + batch);
        auto [dig, em] = eval_solve(chunk, nullptr);
        total_exact += (int)llround(em / 100.0 * batch);
        total += batch;
    }
    printf("Final: solved %d / %d held-out problems exactly  (%.1f%%)\n",
           total_exact, total, 100.0f * total_exact / total);

    // ── Carry-chain robustness (all pairs unseen in training) ───────────
    cout << "\n--- Carry-chain robustness (256 unseen pairs per bucket) ---\n";
    for (int L = 0; L < n_buckets; ++L) {
        auto [dig, em] = eval_solve(carry_sets[L], nullptr);
        const auto& ex = carry_sets[L][0];
        printf("  max carry run = %d :  exact = %5.1f%%   digit = %5.1f%%   (e.g. %d+%d=%d)\n",
               L, em, dig, ex.first, ex.second, ex.first + ex.second);
    }

    // ── Edge-case showcase (padded to a full eval batch with random pairs) ──
    // Under uniform-value sampling these near-all-zero problems (1+1, 0+0, ...)
    // were out-of-distribution and failed; with length-uniform operand
    // sampling every padding pattern is trained on, so they should pass.
    {
        vector<pair<int,int>> fun = {
            {1, 1}, {0, 0}, {0, 1}, {9, 9}, {99999, 1},
            {99999, 99999}, {50000, 50000}, {12345, 54321}};
        const int n_fun = (int)fun.size();
        while ((int)fun.size() < batch)
            fun.push_back({rnd(global_rand_gen), rnd(global_rand_gen)});
        vector<int> fun_pred;
        eval_solve(fun, &fun_pred);
        cout << "\n--- Edge cases ---\n";
        for (int i = 0; i < n_fun; ++i) {
            const int truth = fun[i].first + fun[i].second;
            printf("  %5d + %5d = %-6d  model: %-6d  [%s]\n",
                   fun[i].first, fun[i].second, truth, fun_pred[i],
                   fun_pred[i] == truth ? "OK" : "X");
        }
    }

    // ── Show a few solved problems (autoregressive, held-out) ───────────
    cout << "\n--- Sample held-out problems ---\n";
    vector<pair<int,int>> show(test_pairs.begin(), test_pairs.begin() + batch);
    vector<int> preds;
    eval_solve(show, &preds);
    for (int i = 0; i < 12; ++i) {
        const int a = show[i].first, b = show[i].second, truth = a + b, got = preds[i];
        printf("  %3d + %3d = %-4d  model: %-4d  [%s]\n",
               a, b, truth, got, got == truth ? "OK" : "X");
    }
    return 0;
}
