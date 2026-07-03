/**
 * @file demo_transformer.cu
 * @brief Character language model trained on real-world text.
 *
 *   - corpus loaded from examples/enwik8 (100 MB Wikipedia) if present, else
 *     examples/corpus.txt, else an embedded public-domain passage (Alice)
 *   - byte/char-level: one-hot character input plus one-hot position features
 *   - embed -> two TransformerLayer blocks -> projection
 *   - MINI-BATCH next-character training with the framework LogisticLayer
 *     (random windows sampled each step, so the corpus size is unbounded and
 *      memory per step is constant)
 *   - 90/10 train/validation split; reported val loss/acc measure
 *     generalization, and the train-vs-val gap reveals memorization
 *   - sample predictions and autoregressive generation
 */

#include "../ml/layer.hpp"
#include "../cpp/juzhen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <list>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
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

static void copy_linear_weights(LinearLayer<FLOAT>& dst,
                                LinearLayer<FLOAT>& src) {
    dst.W() = Matrix<FLOAT>(as_host(src.W()));
    dst.b() = Matrix<FLOAT>(as_host(src.b()));
}

// The demo trains on enwik8 (the standard 100 MB char-LM benchmark: the first
// 1e8 bytes of an English Wikipedia dump). Fetch it once into examples/:
//
//     curl -L https://mattmahoney.net/dc/enwik8.zip -o examples/enwik8.zip
//     unzip -o examples/enwik8.zip -d examples/        # -> examples/enwik8
//
// load_corpus() prefers examples/enwik8, then examples/corpus.txt (e.g.
// tiny-shakespeare), then this small embedded public-domain fallback (the
// opening of Lewis Carroll's "Alice's Adventures in Wonderland", 1865).
static string embedded_corpus() {
    return
        "Alice was beginning to get very tired of sitting by her sister on the "
        "bank, and of having nothing to do: once or twice she had peeped into the "
        "book her sister was reading, but it had no pictures or conversations in "
        "it, and what is the use of a book, thought Alice, without pictures or "
        "conversations? So she was considering in her own mind, as well as she "
        "could, for the hot day made her feel very sleepy and stupid, whether the "
        "pleasure of making a daisy-chain would be worth the trouble of getting up "
        "and picking the daisies, when suddenly a White Rabbit with pink eyes ran "
        "close by her. There was nothing so very remarkable in that; nor did Alice "
        "think it so very much out of the way to hear the Rabbit say to itself, Oh "
        "dear! Oh dear! I shall be late! But when the Rabbit actually took a watch "
        "out of its waistcoat-pocket, and looked at it, and then hurried on, Alice "
        "started to her feet, for it flashed across her mind that she had never "
        "before seen a rabbit with either a waistcoat-pocket, or a watch to take "
        "out of it, and burning with curiosity, she ran across the field after it, "
        "and fortunately was just in time to see it pop down a large rabbit-hole "
        "under the hedge. In another moment down went Alice after it, never once "
        "considering how in the world she was to get out again. The rabbit-hole "
        "went straight on like a tunnel for some way, and then dipped suddenly "
        "down, so suddenly that Alice had not a moment to think about stopping "
        "herself before she found herself falling down a very deep well. Either "
        "the well was very deep, or she fell very slowly, for she had plenty of "
        "time as she went down to look about her and to wonder what was going to "
        "happen next. First, she tried to look down and make out what she was "
        "coming to, but it was too dark to see anything; then she looked at the "
        "sides of the well, and noticed that they were filled with cupboards and "
        "book-shelves; here and there she saw maps and pictures hung upon pegs. "
        "She took down a jar from one of the shelves as she passed; it was labelled "
        "ORANGE MARMALADE, but to her great disappointment it was empty: she did "
        "not like to drop the jar for fear of killing somebody underneath, so "
        "managed to put it into one of the cupboards as she fell past it. ";
}

static string load_corpus(string& source) {
    const string base = string(PROJECT_DIR) + "/examples/";
    for (const char* name : {"enwik8", "corpus.txt"}) {
        ifstream f(base + name, ios::binary);
        if (f) {
            stringstream ss;
            ss << f.rdbuf();
            string s = ss.str();
            if (s.size() > 64) {
                source = base + name;
                return s;
            }
        }
    }
    source = "embedded (Alice in Wonderland)";
    return embedded_corpus();
}

// Fill a (in_dim, seq_len*B) input matrix and (V, seq_len*B) target matrix from
// a list of window start offsets. One-hot char + one-hot position features.
static void fill_batch(Matrix<float>& X, Matrix<float>& Y,
                       const string& corpus, const vector<int>& offs,
                       int seq_len, int V, const int* c2i) {
    X.zeros();
    Y.zeros();
    for (int b = 0; b < (int)offs.size(); ++b) {
        const int off = offs[b];
        for (int s = 0; s < seq_len; ++s) {
            const int col = b * seq_len + s;
            X.elem(c2i[(unsigned char)corpus[off + s]], col) = 1.0f;
            X.elem(V + s, col) = 1.0f;
            Y.elem(c2i[(unsigned char)corpus[off + s + 1]], col) = 1.0f;
        }
    }
}

static int sample_top_k(const Matrix<float>& logits,
                        int col,
                        const vector<char>& idx_to_char,
                        const string& recent_text,
                        float temperature,
                        int top_k) {
    vector<pair<float, int>> ranked;
    ranked.reserve(idx_to_char.size());

    const int recent_window = 12;
    const int start = recent_text.size() > recent_window
        ? (int)recent_text.size() - recent_window
        : 0;

    for (int v = 0; v < (int)idx_to_char.size(); ++v) {
        float score = logits.elem(v, col);
        for (int i = start; i < (int)recent_text.size(); ++i) {
            if (recent_text[i] == idx_to_char[v]) score -= 0.02f;
        }
        ranked.push_back({score, v});
    }

    top_k = min(top_k, (int)ranked.size());
    partial_sort(ranked.begin(), ranked.begin() + top_k, ranked.end(),
        [](const pair<float, int>& a, const pair<float, int>& b) {
            return a.first > b.first;
        });

    const float best = ranked[0].first;
    vector<float> cdf(top_k, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        const float prob = expf((ranked[i].first - best) / temperature);
        sum += prob;
        cdf[i] = sum;
    }

    const float draw = std::uniform_real_distribution<float>(0.0f, sum)(global_rand_gen);
    for (int i = 0; i < top_k; ++i) {
        if (draw <= cdf[i]) return ranked[i].second;
    }
    return ranked[top_k - 1].second;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(42);
#endif
    global_rand_gen.seed(42);

    string corpus_source;
    const string corpus = load_corpus(corpus_source);
    const int clen = (int)corpus.size();

    set<char> charset(corpus.begin(), corpus.end());
    vector<char> idx_to_char(charset.begin(), charset.end());
    const int V = (int)idx_to_char.size();
    int c2i[256] = {};
    for (int i = 0; i < V; ++i) c2i[(unsigned char)idx_to_char[i]] = i;

    const int seq_len = 64;
    const int d_model = 256;
    const int d_k = 256;
    const int d_ff = 1024;
    const int num_heads = 8;
    const int num_blocks = 4;
    const int steps = 30000;       // hard cap; early stopping usually ends sooner
    const int log_every = 1000;    // also the early-stopping evaluation interval
    const int patience = 6;        // stop after this many evals w/o val improvement
    const int in_dim = V + seq_len;

    // Learning-rate schedule: linear warmup then cosine decay to 10% of peak.
    const float peak_lr = 5e-4f;
    const int   warmup_steps = 1000;
    const float min_lr = peak_lr * 0.1f;
    const double PI = 3.14159265358979323846;
    auto lr_at = [&](int step) -> float {
        if (step < warmup_steps)
            return peak_lr * (float)(step + 1) / (float)warmup_steps;
        const double t = (double)(step - warmup_steps) / (double)(steps - warmup_steps);
        return (float)(min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + std::cos(PI * t)));
    };

    // Train/validation split: reserve the last 10% of the corpus for held-out
    // evaluation. Valid window START positions form two ranges; we sample from
    // them directly (no per-window index vector), so this scales to enwik8-sized
    // corpora with no extra memory. Train windows lie entirely in the first 90%,
    // val windows in the last 10%; the boundary band is simply never sampled.
    const int split    = (int)((long long)clen * 9 / 10);
    const int train_lo = 0,     train_hi = split - seq_len;   // start range [lo, hi)
    const int val_lo   = split,  val_hi  = clen  - seq_len;
    if (train_hi <= train_lo || val_hi <= val_lo) {
        cout << "Corpus too small for seq_len=" << seq_len << " (needs a train/val split)\n";
        return 1;
    }
    const long long train_windows = (long long)train_hi - train_lo;
    const long long val_windows   = (long long)val_hi   - val_lo;

    int batch = 64;                       // mini-batch size (sequences per step)
    batch = min<long long>(batch, min(train_windows, val_windows));
    const int N = seq_len * batch;

    cout << "=== Transformer Char-LM Demo ===\n";
    cout << "Corpus source: " << corpus_source << "\n";
    cout << "Corpus size: " << clen << " chars (train/val split at " << split << ", 90/10)\n";
    cout << "Windows: " << train_windows << " train, " << val_windows << " val\n";
    cout << "Preview: \"" << corpus.substr(0, min(clen, 80)) << "...\"\n";
    cout << "Vocabulary (" << V << " symbols): ";
    for (int i = 0; i < V; ++i) {
        unsigned char ch = (unsigned char)idx_to_char[i];
        if (ch == ' ') cout << '_';
        else if (ch == '\n') cout << '/';
        else if (ch >= 33 && ch < 127) cout << (char)ch;
        else cout << '.';                              // non-printable byte
    }
    cout << "\n";
    cout << "Network: embed(" << in_dim << "->" << d_model << ") -> Transformer("
         << d_model << "," << d_k << "," << d_ff << "," << num_heads << "h) [x" << num_blocks
         << "] -> proj(" << d_model << "->" << V << ")\n";
    cout << "Training: mini-batch=" << batch << " x seq_len=" << seq_len
         << " (" << N << " tokens/step), steps=" << steps
         << ", lr=warmup(" << warmup_steps << ")->" << peak_lr << "->cos->" << min_lr << "\n\n";

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

    // Core network as a layer list: front = last layer (proj), back = first
    // (embed); forward()/backprop() evaluate from back() to front(), so data
    // flows embed -> block[0] -> ... -> block[num_blocks-1] -> proj.
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

    // Drive the Adam learning rate across all parameter groups (embed/proj +
    // every transformer block) from the schedule above.
    auto set_all_lr = [&](float lr) {
        embed.adamWstate().alpha = lr; embed.adambstate().alpha = lr;
        proj.adamWstate().alpha  = lr; proj.adambstate().alpha  = lr;
        for (int i = 0; i < num_blocks; ++i) blocks[i]->set_lr(lr);
    };

    // Fixed train and val eval batches (evenly spaced windows) for stable,
    // comparable logging. The train-vs-val gap reveals memorization.
    auto make_fixed = [&](int lo, long long count, Matrix<float>& X, Matrix<float>& Y) {
        vector<int> off_(batch);
        for (int b = 0; b < batch; ++b) off_[b] = lo + (int)((long long)b * count / batch);
        fill_batch(X, Y, corpus, off_, seq_len, V, c2i);
    };
    auto Xt_h = Matrix<float>::zeros(in_dim, N);   // fixed train eval batch
    auto Yt_h = Matrix<float>::zeros(V, N);
    auto Xe_h = Matrix<float>::zeros(in_dim, N);   // fixed val eval batch
    auto Ye_h = Matrix<float>::zeros(V, N);
    make_fixed(train_lo, train_windows, Xt_h, Yt_h);
    make_fixed(val_lo,   val_windows,   Xe_h, Ye_h);

    // Evaluate a fixed batch (no weight update): returns {loss, accuracy%}.
    auto eval_batch = [&](const Matrix<float>& Xh, const Matrix<float>& Yh) {
        auto Xb = Matrix<FLOAT>(Xh);
        LogisticLayer<FLOAT> el(N, Matrix<FLOAT>(Yh));
        auto net = build_net(blocks, embed, proj);
        net.push_front(&el);
        const float loss = as_host(forward(net, Xb)).elem(0, 0);
        auto logits = as_host(proj.value());
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            int pred = 0, truth = 0;
            for (int r = 1; r < V; ++r) {
                if (logits.elem(r, i) > logits.elem(pred, i)) pred = r;
                if (Yh.elem(r, i) > Yh.elem(truth, i)) truth = r;
            }
            if (pred == truth) correct++;
        }
        return make_pair(loss, 100.0f * (float)correct / (float)N);
    };

    auto X_h = Matrix<float>::zeros(in_dim, N);
    auto Y_h = Matrix<float>::zeros(V, N);

    uniform_int_distribution<int> pick(train_lo, train_hi - 1);
    vector<int> offs(batch);

    float best_loss = numeric_limits<float>::infinity();
    int   best_step = 0;
    int   stale = 0;               // evals since the last val-loss improvement

    for (int step = 0; step < steps; ++step) {
        set_all_lr(lr_at(step));

        // Sample a fresh random mini-batch from the training region only.
        for (int b = 0; b < batch; ++b) offs[b] = pick(global_rand_gen);
        fill_batch(X_h, Y_h, corpus, offs, seq_len, V, c2i);

        auto X = Matrix<FLOAT>(X_h);
        forward(trainnn, X);
        LogisticLayer<FLOAT> loss(N, Matrix<FLOAT>(Y_h));
        trainnn.push_front(&loss);
        backprop(trainnn, X);
        trainnn.pop_front();

        if (step % log_every == 0 || step == steps - 1) {
            // Held-out evaluation: track the best model on VAL loss, and print
            // train loss alongside so the generalization gap is visible.
            auto [train_l, train_a] = eval_batch(Xt_h, Yt_h);
            auto [val_l,   val_a]   = eval_batch(Xe_h, Ye_h);

            if (val_l < best_loss) {
                best_loss = val_l;
                best_step = step;
                save_best();
                stale = 0;
            } else {
                stale++;
            }

            printf("step %5d   train_loss = %.4f   val_loss = %.4f   val_ppl = %6.2f   "
                   "val_acc = %5.1f%%   lr = %.2e%s\n",
                   step, train_l, val_l, expf(val_l), val_a, lr_at(step),
                   stale == 0 ? "   *best" : "");
            fflush(stdout);

            // Early stopping: val loss has not improved for `patience` evals.
            if (stale >= patience) {
                printf("Early stopping at step %d: no val improvement for %d evals "
                       "(best val_loss = %.4f at step %d).\n",
                       step, patience, best_loss, best_step);
                break;
            }
        }
    }

    // Restore best weights for inference.
    restore_best();

    // Single-sequence inference network (batch = 1).
    LinearLayer<FLOAT> g_embed(d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, d_model, seq_len);
    g_embed.W() = Matrix<FLOAT>(as_host(embed.W()));
    g_embed.b() = Matrix<FLOAT>(as_host(embed.b()));
    g_proj.W() = Matrix<FLOAT>(as_host(proj.W()));
    g_proj.b() = Matrix<FLOAT>(as_host(proj.b()));
    for (int i = 0; i < num_blocks; ++i) copy_tf_weights(*g_blocks[i], *blocks[i]);

    list<Layer<FLOAT>*> gennn = build_net(g_blocks, g_embed, g_proj);
    freeze(gennn);

    cout << "\n--- Sample predictions (held-out val text) ---\n";
    for (int k = 0; k < 3; ++k) {
        const int off = val_lo + (int)((long long)k * val_windows / 3);
        auto enc_h = Matrix<float>::zeros(in_dim, seq_len);
        for (int s = 0; s < seq_len; ++s) {
            enc_h.elem(c2i[(unsigned char)corpus[off + s]], s) = 1.0f;
            enc_h.elem(V + s, s) = 1.0f;
        }
        forward(gennn, Matrix<FLOAT>(enc_h));
        auto logits = as_host(g_proj.value());

        string inp, tgt, pred;
        for (int s = 0; s < seq_len; ++s) {
            inp += corpus[off + s];
            tgt += corpus[off + s + 1];
            int best = 0;
            for (int r = 1; r < V; ++r) {
                if (logits.elem(r, s) > logits.elem(best, s)) best = r;
            }
            pred += idx_to_char[best];
        }
        auto clean = [](string s) {
            for (auto& ch : s) {
                unsigned char u = (unsigned char)ch;
                if (u == '\n') ch = '/';
                else if (u < 32 || u >= 127) ch = '.';   // keep terminal output readable
            }
            return s;
        };
        printf("  in:   \"%s\"\n  want: \"%s\"\n  got:  \"%s\"\n\n",
               clean(inp).c_str(), clean(tgt).c_str(), clean(pred).c_str());
    }

    // Autoregressive generation seeded from the start of the corpus.
    string window = corpus.substr(0, seq_len);
    string generated = window;
    const int gen_len = 200;

    for (int i = 0; i < gen_len; ++i) {
        auto enc_h = Matrix<float>::zeros(in_dim, seq_len);
        for (int s = 0; s < seq_len; ++s) {
            enc_h.elem(c2i[(unsigned char)window[s]], s) = 1.0f;
            enc_h.elem(V + s, s) = 1.0f;
        }
        forward(gennn, Matrix<FLOAT>(enc_h));
        auto out = as_host(g_proj.value());
        const int next = sample_top_k(out, seq_len - 1, idx_to_char, generated, 0.8f, 5);
        generated += idx_to_char[next];
        window = window.substr(1) + idx_to_char[next];
    }

    cout << "--- Generated text ---\n";
    printf("Seed:      \"%s\"\n", corpus.substr(0, seq_len).c_str());
    printf("Generated: \"%s\"\n", generated.c_str());
    return 0;
}
