/**
 * @file demo_transformer.cu
 * @brief Character language model trained on real-world text.
 *
 *   - corpus loaded from examples/corpus.txt if present, else an embedded
 *     public-domain passage (Alice's Adventures in Wonderland)
 *   - one-hot character input plus one-hot position features
 *   - embed -> two TransformerLayer blocks -> projection
 *   - MINI-BATCH next-character training with the framework LogisticLayer
 *     (random windows sampled each step, so the corpus size is unbounded and
 *      memory per step is constant)
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

// A public-domain fallback corpus: the opening of Lewis Carroll's
// "Alice's Adventures in Wonderland" (1865). Drop a larger file at
// examples/corpus.txt (e.g. tiny-shakespeare) to train on your own text.
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
    const string path = string(PROJECT_DIR) + "/examples/corpus.txt";
    ifstream f(path, ios::binary);
    if (f) {
        stringstream ss;
        ss << f.rdbuf();
        string s = ss.str();
        if (s.size() > 64) {
            source = path;
            return s;
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
    const int steps = 30000;
    const int log_every = 2000;
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

    // Every valid window start (predicting one char ahead needs off+seq_len).
    vector<int> window_starts;
    for (int pos = 0; pos + seq_len < clen; ++pos) window_starts.push_back(pos);
    if (window_starts.empty()) {
        cout << "Corpus too small for seq_len=" << seq_len << "\n";
        return 1;
    }

    int batch = 64;                       // mini-batch size (sequences per step)
    batch = min(batch, (int)window_starts.size());
    const int N = seq_len * batch;

    cout << "=== Transformer Char-LM Demo ===\n";
    cout << "Corpus source: " << corpus_source << "\n";
    cout << "Corpus size: " << clen << " chars, " << window_starts.size() << " windows\n";
    cout << "Preview: \"" << corpus.substr(0, min(clen, 80)) << "...\"\n";
    cout << "Vocabulary (" << V << " chars): ";
    for (int i = 0; i < V; ++i) {
        char ch = idx_to_char[i];
        cout << (ch == ' ' ? '_' : (ch == '\n' ? '/' : ch));
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

    // Fixed evaluation batch (evenly spaced windows) for stable loss/acc logging.
    vector<int> eval_offs(batch);
    for (int b = 0; b < batch; ++b)
        eval_offs[b] = window_starts[(size_t)b * window_starts.size() / batch];
    auto Xe_h = Matrix<float>::zeros(in_dim, N);
    auto Ye_h = Matrix<float>::zeros(V, N);
    fill_batch(Xe_h, Ye_h, corpus, eval_offs, seq_len, V, c2i);

    auto X_h = Matrix<float>::zeros(in_dim, N);
    auto Y_h = Matrix<float>::zeros(V, N);

    uniform_int_distribution<int> pick(0, (int)window_starts.size() - 1);
    vector<int> offs(batch);

    float best_loss = numeric_limits<float>::infinity();

    for (int step = 0; step < steps; ++step) {
        set_all_lr(lr_at(step));

        // Sample a fresh random mini-batch of windows.
        for (int b = 0; b < batch; ++b) offs[b] = window_starts[pick(global_rand_gen)];
        fill_batch(X_h, Y_h, corpus, offs, seq_len, V, c2i);

        auto X = Matrix<FLOAT>(X_h);
        forward(trainnn, X);
        LogisticLayer<FLOAT> loss(N, Matrix<FLOAT>(Y_h));
        trainnn.push_front(&loss);
        backprop(trainnn, X);
        trainnn.pop_front();

        if (step % log_every == 0 || step == steps - 1) {
            // Evaluate on the fixed eval batch (no weight update here).
            auto Xe = Matrix<FLOAT>(Xe_h);
            LogisticLayer<FLOAT> eval_loss(N, Matrix<FLOAT>(Ye_h));
            auto evalnet = build_net(blocks, embed, proj);
            evalnet.push_front(&eval_loss);
            const float eval_l = as_host(forward(evalnet, Xe)).elem(0, 0);

            auto logits = as_host(proj.value());
            int correct = 0;
            for (int i = 0; i < N; ++i) {
                int pred = 0, truth = 0;
                for (int r = 1; r < V; ++r) {
                    if (logits.elem(r, i) > logits.elem(pred, i)) pred = r;
                    if (Ye_h.elem(r, i) > Ye_h.elem(truth, i)) truth = r;
                }
                if (pred == truth) correct++;
            }
            const float acc = 100.0f * (float)correct / (float)N;

            if (eval_l < best_loss) {
                best_loss = eval_l;
                save_best();
            }

            printf("step %5d   eval_loss = %.4f   ppl = %7.2f   acc = %5.1f%%   lr = %.2e\n",
                   step, eval_l, expf(eval_l), acc, lr_at(step));
            fflush(stdout);
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

    cout << "\n--- Sample predictions ---\n";
    for (int k = 0; k < 3; ++k) {
        const int off = window_starts[(size_t)k * window_starts.size() / 3];
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
            for (auto& ch : s) if (ch == '\n') ch = '/';
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
