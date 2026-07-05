/**
 * @file demo_transformer_web.cu
 * @brief Web demo: character-level Transformer language model with an
 *        interactive text-completion page (see examples/web/transformer.html).
 *
 * Training is identical to demo_transformer.cu. The trained (batch=1)
 * inference network is cached in transformer_web.weights at the project root,
 * so only the first launch pays the training cost; later launches load the
 * cache and serve immediately. Delete the file to retrain.
 *
    Copyright (C) 2022 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "../ml/layer.hpp"
#include "../cpp/juzhen.hpp"
#include "../external/httplib.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
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
    const string base = string(PROJECT_DIR) + "/datasets/";
    for (const char* name : {"text8", "enwik8", "corpus.txt"}) {
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

// ---------------------------------------------------------------------------
// Weight cache for the batch=1 inference network.
// ---------------------------------------------------------------------------

static const int WEIGHTS_MAGIC = 0x4a5a5457;  // "JZTW"

struct ModelConfig {
    int seq_len, d_model, d_k, d_ff, num_heads, num_blocks;
};

// Try to read just the header; on success returns true and fills vocab/stats.
static bool read_cache_header(FILE* fp, const ModelConfig& cfg,
                              vector<char>& vocab, float& val_loss, float& val_acc) {
    int header[8] = {};
    if (fread(header, sizeof(int), 8, fp) != 8) return false;
    if (header[0] != WEIGHTS_MAGIC) return false;
    const int V = header[1];
    if (header[2] != cfg.seq_len || header[3] != cfg.d_model ||
        header[4] != cfg.d_k || header[5] != cfg.d_ff ||
        header[6] != cfg.num_heads || header[7] != cfg.num_blocks) return false;
    if (V <= 0 || V > 256) return false;
    vocab.resize(V);
    if (fread(vocab.data(), sizeof(char), V, fp) != (size_t)V) return false;
    if (fread(&val_loss, sizeof(float), 1, fp) != 1) return false;
    if (fread(&val_acc, sizeof(float), 1, fp) != 1) return false;
    return true;
}

static void load_cached_weights(FILE* fp, int num_blocks,
                                LinearLayer<FLOAT>& emb,
                                vector<unique_ptr<TransformerLayer<FLOAT>>>& blocks,
                                LinearLayer<FLOAT>& proj) {
    Matrix<float> M("cache", 1, 1);
    auto next = [&]() -> Matrix<float>& { read(fp, M); return M; };
    emb.W() = Matrix<FLOAT>(next());
    emb.b() = Matrix<FLOAT>(next());
    for (int i = 0; i < num_blocks; ++i) {
        auto& l = *blocks[i];
        l.set_Wq(Matrix<FLOAT>(next()));
        l.set_Wk(Matrix<FLOAT>(next()));
        l.set_Wv(Matrix<FLOAT>(next()));
        l.set_Wo(Matrix<FLOAT>(next()));
        l.set_bo(Matrix<FLOAT>(next()));
        l.set_W1(Matrix<FLOAT>(next()));
        l.set_b1(Matrix<FLOAT>(next()));
        l.set_W2(Matrix<FLOAT>(next()));
        l.set_b2(Matrix<FLOAT>(next()));
        l.set_ln1_gamma(Matrix<FLOAT>(next()));
        l.set_ln1_beta(Matrix<FLOAT>(next()));
        l.set_ln2_gamma(Matrix<FLOAT>(next()));
        l.set_ln2_beta(Matrix<FLOAT>(next()));
    }
    proj.W() = Matrix<FLOAT>(next());
    proj.b() = Matrix<FLOAT>(next());
}

static void save_cached_weights(const string& path, const ModelConfig& cfg,
                                const vector<char>& vocab,
                                float val_loss, float val_acc,
                                LinearLayer<FLOAT>& emb,
                                vector<unique_ptr<TransformerLayer<FLOAT>>>& blocks,
                                LinearLayer<FLOAT>& proj) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        cout << "WARNING: cannot write weight cache " << path << endl;
        return;
    }
    int header[8] = {WEIGHTS_MAGIC, (int)vocab.size(), cfg.seq_len, cfg.d_model,
                     cfg.d_k, cfg.d_ff, cfg.num_heads, cfg.num_blocks};
    fwrite(header, sizeof(int), 8, fp);
    fwrite(vocab.data(), sizeof(char), vocab.size(), fp);
    fwrite(&val_loss, sizeof(float), 1, fp);
    fwrite(&val_acc, sizeof(float), 1, fp);
    auto put = [&](const Matrix<float>& M) { write(fp, M); };
    put(as_host(emb.W()));
    put(as_host(emb.b()));
    for (int i = 0; i < (int)blocks.size(); ++i) {
        auto& l = *blocks[i];
        put(as_host(l.get_Wq()));
        put(as_host(l.get_Wk()));
        put(as_host(l.get_Wv()));
        put(as_host(l.get_Wo()));
        put(as_host(l.get_bo()));
        put(as_host(l.get_W1()));
        put(as_host(l.get_b1()));
        put(as_host(l.get_W2()));
        put(as_host(l.get_b2()));
        put(as_host(l.get_ln1_gamma()));
        put(as_host(l.get_ln1_beta()));
        put(as_host(l.get_ln2_gamma()));
        put(as_host(l.get_ln2_beta()));
    }
    put(as_host(proj.W()));
    put(as_host(proj.b()));
    fclose(fp);
    cout << "Saved weight cache to " << path << endl;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(42);
#endif
    global_rand_gen.seed(42);

    string corpus_source;
    const string corpus = load_corpus(corpus_source);
    const int clen = (int)corpus.size();

    // Model config, scaled up from demo_transformer's default (seq 64, d_model
    // 256, 4 blocks, batch 64, ~4 min) to saturate a 24 GB RTX 4090 over a
    // multi-hour budget: 2x context, 2x width, 2x depth, 2x batch. Enable TF32
    // tensor-core GEMMs with NVIDIA_TF32=1 for a large throughput boost.
    const ModelConfig cfg = {/*seq_len*/ 128, /*d_model*/ 512, /*d_k*/ 512,
                             /*d_ff*/ 2048, /*num_heads*/ 8, /*num_blocks*/ 12};
    const int seq_len = cfg.seq_len;
    const string cachepath = string(PROJECT_DIR) + "/transformer_web.weights";

    // Vocabulary: from the weight cache if present (it must match the trained
    // embedding), otherwise derived from the corpus.
    vector<char> idx_to_char;
    float val_loss = -1, val_acc = -1;
    FILE* cachefp = fopen(cachepath.c_str(), "rb");
    bool cached = false;
    if (cachefp) {
        cached = read_cache_header(cachefp, cfg, idx_to_char, val_loss, val_acc);
        if (!cached) {
            fclose(cachefp);
            cachefp = nullptr;
            cout << "Weight cache " << cachepath << " is stale or invalid; retraining." << endl;
        }
    }
    if (!cached) {
        set<char> charset(corpus.begin(), corpus.end());
        idx_to_char.assign(charset.begin(), charset.end());
    }
    const int V = (int)idx_to_char.size();
    int c2i[256] = {};
    for (int i = 0; i < V; ++i) c2i[(unsigned char)idx_to_char[i]] = i;
    const int in_dim = V + seq_len;

    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < cfg.num_blocks; ++i)
            v.push_back(make_unique<TF>(cfg.d_model, cfg.d_k, cfg.d_ff, seq_len, b, cfg.num_heads));
        return v;
    };

    // Single-sequence inference network (batch = 1); filled from the cache or
    // from a fresh training run below.
    LinearLayer<FLOAT> g_embed(cfg.d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, cfg.d_model, seq_len);

    if (cached) {
        load_cached_weights(cachefp, cfg.num_blocks, g_embed, g_blocks, g_proj);
        fclose(cachefp);
        cout << "Loaded weight cache " << cachepath
             << " (val_loss = " << val_loss << ", val_acc = " << val_acc << "%)." << endl;
        cout << "Delete the file and restart to retrain." << endl;
    } else {
        // --------------- training, identical to demo_transformer ---------------
        int steps = 100000;
        if (const char* s = getenv("STEPS")) steps = max(1, atoi(s));
        const int log_every = 1000;
        const int patience = 15;

        const float peak_lr = 5e-4f;
        const int   warmup_steps = 2000;
        const float min_lr = peak_lr * 0.1f;
        const double PI = 3.14159265358979323846;
        auto lr_at = [&](int step) -> float {
            if (step < warmup_steps)
                return peak_lr * (float)(step + 1) / (float)warmup_steps;
            const double t = (double)(step - warmup_steps) / (double)(steps - warmup_steps);
            return (float)(min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + std::cos(PI * t)));
        };

        const int split    = (int)((long long)clen * 9 / 10);
        const int train_lo = 0,     train_hi = split - seq_len;
        const int val_lo   = split,  val_hi  = clen  - seq_len;
        if (train_hi <= train_lo || val_hi <= val_lo) {
            cout << "Corpus too small for seq_len=" << seq_len << " (needs a train/val split)\n";
            return 1;
        }
        const long long train_windows = (long long)train_hi - train_lo;
        const long long val_windows   = (long long)val_hi   - val_lo;

        int batch = 128;
        batch = min<long long>(batch, min(train_windows, val_windows));
        const int N = seq_len * batch;

        cout << "=== Transformer Char-LM Web Demo: training ===\n";
        cout << "Corpus source: " << corpus_source << " (" << clen << " chars)\n";
        cout << "Vocabulary: " << V << " symbols\n";
        cout << "Network: embed(" << in_dim << "->" << cfg.d_model << ") -> Transformer("
             << cfg.d_model << "," << cfg.d_k << "," << cfg.d_ff << "," << cfg.num_heads
             << "h) [x" << cfg.num_blocks << "] -> proj(" << cfg.d_model << "->" << V << ")\n";
        cout << "Training: mini-batch=" << batch << " x seq_len=" << seq_len
             << ", steps=" << steps << "\n\n";

        LinearLayer<FLOAT> embed(cfg.d_model, in_dim, N);
        auto blocks = make_blocks(batch);
        LinearLayer<FLOAT> proj(V, cfg.d_model, N);

        // batch=1: the snapshot only ever stores weights (never runs a pass), so
        // skip the large per-batch activation caches a layer allocates.
        LinearLayer<FLOAT> best_embed(cfg.d_model, in_dim, 1);
        auto best_blocks = make_blocks(1);
        LinearLayer<FLOAT> best_proj(V, cfg.d_model, 1);

        auto build_net = [&](vector<unique_ptr<TF>>& bl,
                             LinearLayer<FLOAT>& emb, LinearLayer<FLOAT>& pr) {
            list<Layer<FLOAT>*> net;
            net.push_back(&pr);
            for (int i = cfg.num_blocks - 1; i >= 0; --i) net.push_back(bl[i].get());
            net.push_back(&emb);
            return net;
        };

        list<Layer<FLOAT>*> trainnn = build_net(blocks, embed, proj);

        auto save_best = [&]() {
            copy_linear_weights(best_embed, embed);
            copy_linear_weights(best_proj, proj);
            for (int i = 0; i < cfg.num_blocks; ++i) copy_tf_weights(*best_blocks[i], *blocks[i]);
        };
        auto restore_best = [&]() {
            copy_linear_weights(embed, best_embed);
            copy_linear_weights(proj, best_proj);
            for (int i = 0; i < cfg.num_blocks; ++i) copy_tf_weights(*blocks[i], *best_blocks[i]);
        };
        save_best();

        auto set_all_lr = [&](float lr) {
            embed.adamWstate().alpha = lr; embed.adambstate().alpha = lr;
            proj.adamWstate().alpha  = lr; proj.adambstate().alpha  = lr;
            for (int i = 0; i < cfg.num_blocks; ++i) blocks[i]->set_lr(lr);
        };

        auto make_fixed = [&](int lo, long long count, Matrix<float>& X, Matrix<float>& Y) {
            vector<int> off_(batch);
            for (int b = 0; b < batch; ++b) off_[b] = lo + (int)((long long)b * count / batch);
            fill_batch(X, Y, corpus, off_, seq_len, V, c2i);
        };
        auto Xt_h = Matrix<float>::zeros(in_dim, N);
        auto Yt_h = Matrix<float>::zeros(V, N);
        auto Xe_h = Matrix<float>::zeros(in_dim, N);
        auto Ye_h = Matrix<float>::zeros(V, N);
        make_fixed(train_lo, train_windows, Xt_h, Yt_h);
        make_fixed(val_lo,   val_windows,   Xe_h, Ye_h);

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
        float best_acc  = 0;
        int   best_step = 0;
        int   stale = 0;
        int   last_ckpt_step = 0;   // periodic checkpoint of the best snapshot

        for (int step = 0; step < steps; ++step) {
            set_all_lr(lr_at(step));

            for (int b = 0; b < batch; ++b) offs[b] = pick(global_rand_gen);
            fill_batch(X_h, Y_h, corpus, offs, seq_len, V, c2i);

            auto X = Matrix<FLOAT>(X_h);
            forward(trainnn, X);
            LogisticLayer<FLOAT> loss(N, Matrix<FLOAT>(Y_h));
            trainnn.push_front(&loss);
            backprop(trainnn, X);
            trainnn.pop_front();

            if (step % log_every == 0 || step == steps - 1) {
                auto [train_l, train_a] = eval_batch(Xt_h, Yt_h);
                auto [val_l,   val_a]   = eval_batch(Xe_h, Ye_h);

                if (val_l < best_loss) {
                    best_loss = val_l;
                    best_acc  = val_a;
                    best_step = step;
                    save_best();
                    stale = 0;
                    // Periodic crash-safe checkpoint: persist the best snapshot
                    // so an interrupted run can still serve (or be inspected).
                    if (step - last_ckpt_step >= 5000) {
                        save_cached_weights(cachepath, cfg, idx_to_char, best_loss,
                                            best_acc, best_embed, best_blocks, best_proj);
                        last_ckpt_step = step;
                    }
                } else {
                    stale++;
                }

                printf("step %5d   train_loss = %.4f   val_loss = %.4f   val_ppl = %6.2f   "
                       "val_acc = %5.1f%%   lr = %.2e%s\n",
                       step, train_l, val_l, expf(val_l), val_a, lr_at(step),
                       stale == 0 ? "   *best" : "");
                fflush(stdout);

                if (stale >= patience) {
                    printf("Early stopping at step %d: no val improvement for %d evals "
                           "(best val_loss = %.4f at step %d).\n",
                           step, patience, best_loss, best_step);
                    break;
                }
            }
        }

        restore_best();
        val_loss = best_loss;
        val_acc  = best_acc;

        // copy the trained weights into the batch=1 inference network + cache
        g_embed.W() = Matrix<FLOAT>(as_host(embed.W()));
        g_embed.b() = Matrix<FLOAT>(as_host(embed.b()));
        g_proj.W() = Matrix<FLOAT>(as_host(proj.W()));
        g_proj.b() = Matrix<FLOAT>(as_host(proj.b()));
        for (int i = 0; i < cfg.num_blocks; ++i) copy_tf_weights(*g_blocks[i], *blocks[i]);

        save_cached_weights(cachepath, cfg, idx_to_char, val_loss, val_acc,
                            g_embed, g_blocks, g_proj);
        // ------------------------------------------------------------------------
    }

    list<Layer<FLOAT>*> gennn;
    gennn.push_back(&g_proj);
    for (int i = cfg.num_blocks - 1; i >= 0; --i) gennn.push_back(g_blocks[i].get());
    gennn.push_back(&g_embed);
    freeze(gennn);

    // map arbitrary user text into the model vocabulary
    bool in_vocab[256] = {};
    for (int i = 0; i < V; ++i) in_vocab[(unsigned char)idx_to_char[i]] = true;
    const char fallback_char = in_vocab[(unsigned char)' '] ? ' ' : idx_to_char[0];
    auto sanitize = [&](const string& s) {
        string out = s;
        for (auto& ch : out) {
            unsigned char u = (unsigned char)ch;
            if (!in_vocab[u]) {
                unsigned char lo = (unsigned char)tolower(u);
                ch = in_vocab[lo] ? (char)lo : fallback_char;
            }
        }
        return out;
    };

    // in-distribution left context used when the user seed is shorter than seq_len
    const string pad_context = sanitize(corpus.substr(0, seq_len));

    // the matrix library is not thread-safe; serialize generations
    auto predict_mutex = make_shared<mutex>();

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        ifstream f(PROJECT_DIR + string("/examples/web/transformer.html"));
        if (!f) {
            res.status = 500;
            res.set_content("examples/web/transformer.html not found", "text/plain");
            return;
        }
        stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    svr.Get("/info", [&](const httplib::Request&, httplib::Response& res) {
        ostringstream ss;
        ss << "{\"vocab\":" << V
           << ",\"seq_len\":" << seq_len
           << ",\"d_model\":" << cfg.d_model
           << ",\"num_blocks\":" << cfg.num_blocks
           << ",\"num_heads\":" << cfg.num_heads
           << ",\"val_loss\":" << val_loss
           << ",\"val_acc\":" << val_acc
           << ",\"corpus\":\"" << corpus_source << "\"}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /generate?length=200&temperature=0.8&topk=5, body = seed text (UTF-8).
    // Streams generated characters back as they are sampled.
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        int length = 200;
        float temperature = 0.8f;
        int topk = 5;
        if (req.has_param("length")) length = atoi(req.get_param_value("length").c_str());
        if (req.has_param("temperature")) temperature = atof(req.get_param_value("temperature").c_str());
        if (req.has_param("topk")) topk = atoi(req.get_param_value("topk").c_str());
        length = max(1, min(length, 2000));
        temperature = max(0.05f, min(temperature, 5.0f));
        topk = max(1, min(topk, V));

        string seed = sanitize(req.body);
        // window = last seq_len chars of (pad_context + seed)
        string context = pad_context + seed;
        auto state = make_shared<string>(context.substr(context.size() - seq_len));
        auto recent = make_shared<string>(seed);

        res.set_chunked_content_provider(
            "text/plain; charset=utf-8",
            [&, state, recent, length, temperature, topk](size_t, httplib::DataSink& sink) {
                lock_guard<mutex> lock(*predict_mutex);
                string& window = *state;
                for (int i = 0; i < length; ++i) {
                    auto enc_h = Matrix<float>::zeros(in_dim, seq_len);
                    for (int s = 0; s < seq_len; ++s) {
                        enc_h.elem(c2i[(unsigned char)window[s]], s) = 1.0f;
                        enc_h.elem(V + s, s) = 1.0f;
                    }
                    forward(gennn, Matrix<FLOAT>(enc_h));
                    auto out = as_host(g_proj.value());
                    const int next = sample_top_k(out, seq_len - 1, idx_to_char,
                                                  *recent, temperature, topk);
                    const char c = idx_to_char[next];
                    *recent += c;
                    window = window.substr(1) + c;
                    if (!sink.write(&c, 1)) break;  // client disconnected
                }
                sink.done();
                return true;
            });
    });

    int port = 8090;
    if (const char* p = getenv("PORT")) port = atoi(p);
    cout << "Serving Transformer web demo on http://localhost:" << port << endl;
    if (!svr.listen("0.0.0.0", port)) {
        cout << "Failed to listen on port " << port << endl;
        return 1;
    }

    return 0;
}
