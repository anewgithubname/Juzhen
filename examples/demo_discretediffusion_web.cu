/**
 * @file demo_discretediffusion_web.cu
 * @brief Web demo: masked discrete diffusion char-LM with an interactive
 *        fill-in-the-blank page (see examples/web/diffusion.html).
 *
 * Training is identical to demo_discretediffusion.cu. The trained (batch=1)
 * denoiser is cached in discretediffusion_web.weights at the project root, so
 * only the first launch pays the ~15 min training cost; later launches load
 * the cache and serve immediately. Delete the file to retrain.
 *
 * The /fill endpoint takes text where '_' marks a blank, runs MaskGIT-style
 * progressive decoding, and streams every denoising round back as NDJSON so
 * the page can animate the blanks being resolved.
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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
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

// Column-wise max of a (V, N) matrix -> (1, N); used for a stable softmax.
template <class D>
static Matrix<D> col_max(const Matrix<D>& input) {
    return reduce(
        [] __GPU_CPU__(float* v, float* vdes, int lenv, int) {
            float m = -1e30f;
            for (int i = 0; i < lenv; i++) m = m > v[i] ? m : v[i];
            vdes[0] = m;
        }, input, 0, 1);
}

// Masked, 1/t-weighted softmax cross-entropy (see demo_discretediffusion.cu).
template <class D>
class MaskedCELayer : public Layer<D> {
    Matrix<D> output;   // (V, N) one-hot true characters
    Matrix<D> wrow;     // (1, N) per-column weights (0 for unmasked positions)
    Matrix<D> oneK1;    // (V, 1) ones, for broadcasting a row over all V rows

public:
    MaskedCELayer(size_t nb, Matrix<D>&& target, Matrix<D>&& weights)
        : Layer<D>(2, 2, nb),
          output(std::move(target)),
          wrow(std::move(weights)),
          oneK1("oneK1", output.num_row(), 1) {
        oneK1.ones();
    }

    Matrix<D> grad(const Matrix<D>& input) const override {
        auto mx = col_max(input);
        auto E = exp(input - oneK1 * mx);
        auto Z = oneK1 * sum(E, 0);
        auto softmax = E / std::move(Z);
        return hadmd(softmax - output, oneK1 * wrow);
    }

    void eval(const Matrix<D>& input) override {
        auto mx = col_max(input);
        auto lse = log(sum(exp(input - oneK1 * mx), 0)) + mx;
        auto dot = sum(hadmd(input, output), 0);
        auto ce = lse - dot;
        Layer<D>::val = sum(hadmd(wrow, ce), 1);
    }
};

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
        "considering how in the world she was to get out again. ";
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

static string json_escape(const string& s) {
    string out;
    out.reserve(s.size() + 8);
    for (unsigned char u : s) {
        switch (u) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (u < 32 || u >= 127) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", u);
                    out += buf;
                } else {
                    out += (char)u;
                }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Weight cache for the batch=1 denoiser network.
// ---------------------------------------------------------------------------

static const int WEIGHTS_MAGIC = 0x4a5a4444;  // "JZDD"

struct ModelConfig {
    int seq_len, d_model, d_k, d_ff, num_heads, num_blocks;
};

static bool read_cache_header(FILE* fp, const ModelConfig& cfg,
                              vector<char>& vocab,
                              float& val_loss, float& val_bpc, float& val_acc) {
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
    if (fread(&val_bpc, sizeof(float), 1, fp) != 1) return false;
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
                                float val_loss, float val_bpc, float val_acc,
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
    fwrite(&val_bpc, sizeof(float), 1, fp);
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
    uniform_real_distribution<float> u01(0.0f, 1.0f);

    string corpus_source;
    const string corpus = load_corpus(corpus_source);
    const int clen = (int)corpus.size();

    // model config, identical to demo_discretediffusion
    const ModelConfig cfg = {/*seq_len*/ 128, /*d_model*/ 512, /*d_k*/ 512,
                             /*d_ff*/ 2048, /*num_heads*/ 8, /*num_blocks*/ 8};
    const int seq_len = cfg.seq_len;
    const char* wenv = getenv("DIFFUSION_WEIGHTS");
    const string cachepath = wenv ? string(wenv)
                                  : string(PROJECT_DIR) + "/discretediffusion_web.weights";

    // Vocabulary: from the weight cache if present, else from the corpus.
    vector<char> idx_to_char;
    float val_loss = -1, val_bpc = -1, val_acc = -1;
    FILE* cachefp = fopen(cachepath.c_str(), "rb");
    bool cached = false;
    if (cachefp) {
        cached = read_cache_header(cachefp, cfg, idx_to_char, val_loss, val_bpc, val_acc);
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
    const int V = (int)idx_to_char.size();   // real vocabulary size
    int c2i[256] = {};
    for (int i = 0; i < V; ++i) c2i[(unsigned char)idx_to_char[i]] = i;
    const int MASK = V;                      // extra input-only [MASK] token
    const int V1 = V + 1;
    const int in_dim = V1 + seq_len;
    const float t_min = 1e-3f;

    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < cfg.num_blocks; ++i)
            v.push_back(make_unique<TF>(cfg.d_model, cfg.d_k, cfg.d_ff, seq_len, b,
                                        cfg.num_heads, /*causal=*/false));
        return v;
    };

    // Single-sequence denoiser (batch = 1); filled from the cache or trained.
    LinearLayer<FLOAT> g_embed(cfg.d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, cfg.d_model, seq_len);

    if (cached) {
        load_cached_weights(cachefp, cfg.num_blocks, g_embed, g_blocks, g_proj);
        fclose(cachefp);
        cout << "Loaded weight cache " << cachepath << " (val_bpc = " << val_bpc
             << ", val_acc = " << val_acc << "%)." << endl;
        cout << "Delete the file and restart to retrain." << endl;
    } else {
        // ------------- training, identical to demo_discretediffusion -----------
        int steps = 20000;
        if (const char* s = getenv("STEPS")) steps = max(1, atoi(s));
        const int log_every = 500;
        const int patience = 12;

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

        const int split    = (int)((long long)clen * 9 / 10);
        const int train_lo = 0,     train_hi = split - seq_len;
        const int val_lo   = split, val_hi   = clen  - seq_len;
        if (train_hi <= train_lo || val_hi <= val_lo) {
            cout << "Corpus too small for seq_len=" << seq_len << "\n";
            return 1;
        }
        const long long train_windows = (long long)train_hi - train_lo;
        const long long val_windows   = (long long)val_hi   - val_lo;

        int batch = 128;
        batch = min<long long>(batch, min(train_windows, val_windows));
        const int N = seq_len * batch;

        cout << "=== Masked Discrete Diffusion Web Demo: training ===\n";
        cout << "Corpus source: " << corpus_source << " (" << clen << " chars)\n";
        cout << "Vocabulary: " << V << " real symbols + 1 [MASK]\n";
        cout << "Network: embed(" << in_dim << "->" << cfg.d_model << ") -> Transformer("
             << cfg.d_model << "," << cfg.d_k << "," << cfg.d_ff << "," << cfg.num_heads
             << "h,bidir) [x" << cfg.num_blocks << "] -> proj(" << cfg.d_model << "->" << V << ")\n";
        cout << "Training: mini-batch=" << batch << " x seq_len=" << seq_len
             << ", steps=" << steps << "\n\n";

        LinearLayer<FLOAT> embed(cfg.d_model, in_dim, N);
        auto blocks = make_blocks(batch);
        LinearLayer<FLOAT> proj(V, cfg.d_model, N);

        // batch=1: weights only, avoids duplicating the big activation caches
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

        auto fill_batch = [&](const vector<int>& offs, const vector<float>& tlvl,
                              Matrix<float>& X, Matrix<float>& Y, Matrix<float>& W,
                              vector<int>& true_idx, vector<char>& masked) {
            X.zeros(); Y.zeros(); W.zeros();
            for (int b = 0; b < (int)offs.size(); ++b) {
                const int off = offs[b];
                const float t = tlvl[b];
                for (int s = 0; s < seq_len; ++s) {
                    const int col = b * seq_len + s;
                    const int c = c2i[(unsigned char)corpus[off + s]];
                    const bool m = (u01(global_rand_gen) < t);
                    X.elem(m ? MASK : c, col) = 1.0f;
                    X.elem(V1 + s, col) = 1.0f;
                    Y.elem(c, col) = 1.0f;
                    W.elem(0, col) = m ? (1.0f / t) / (float)N : 0.0f;
                    true_idx[col] = c;
                    masked[col] = m ? 1 : 0;
                }
            }
        };

        auto stratified_t = [&](vector<float>& t) {
            const float u = u01(global_rand_gen);
            for (int b = 0; b < (int)t.size(); ++b)
                t[b] = t_min + (1.0f - t_min) * ((float)b + u) / (float)t.size();
            shuffle(t.begin(), t.end(), global_rand_gen);
        };

        auto Xh = Matrix<float>::zeros(in_dim, N);
        auto Yh = Matrix<float>::zeros(V, N);
        auto Wh = Matrix<float>::zeros(1, N);
        vector<int> true_idx(N);
        vector<char> masked(N);
        vector<int> offs(batch);
        vector<float> tlvl(batch);

        auto eval_batch = [&](int lo, long long count, int draws) {
            for (int b = 0; b < batch; ++b) offs[b] = lo + (int)((long long)b * count / batch);
            float loss_sum = 0.0f; long long correct = 0, total = 0;
            for (int d = 0; d < draws; ++d) {
                stratified_t(tlvl);
                fill_batch(offs, tlvl, Xh, Yh, Wh, true_idx, masked);

                auto Xb = Matrix<FLOAT>(Xh);
                MaskedCELayer<FLOAT> el(N, Matrix<FLOAT>(Yh), Matrix<FLOAT>(Wh));
                auto net = build_net(blocks, embed, proj);
                net.push_front(&el);
                loss_sum += as_host(forward(net, Xb)).elem(0, 0);

                auto logits = as_host(proj.value());
                for (int i = 0; i < N; ++i) {
                    if (!masked[i]) continue;
                    int pred = 0;
                    for (int r = 1; r < V; ++r)
                        if (logits.elem(r, i) > logits.elem(pred, i)) pred = r;
                    if (pred == true_idx[i]) correct++;
                    total++;
                }
            }
            const float loss = loss_sum / draws;
            const float bpc = loss / (float)log(2.0);
            const float acc = total ? 100.0f * (float)correct / (float)total : 0.0f;
            return make_tuple(loss, bpc, acc);
        };

        uniform_int_distribution<int> pick(train_lo, train_hi - 1);

        float best_loss = numeric_limits<float>::infinity();
        float best_bpc_v = -1, best_acc_v = -1;
        int   best_step = 0, stale = 0;

        for (int step = 0; step < steps; ++step) {
            set_all_lr(lr_at(step));

            for (int b = 0; b < batch; ++b) offs[b] = pick(global_rand_gen);
            stratified_t(tlvl);
            fill_batch(offs, tlvl, Xh, Yh, Wh, true_idx, masked);

            auto X = Matrix<FLOAT>(Xh);
            forward(trainnn, X);
            MaskedCELayer<FLOAT> loss(N, Matrix<FLOAT>(Yh), Matrix<FLOAT>(Wh));
            trainnn.push_front(&loss);
            backprop(trainnn, X);
            trainnn.pop_front();

            if (step % log_every == 0 || step == steps - 1) {
                auto [tr_l, tr_bpc, tr_a] = eval_batch(train_lo, train_windows, 4);
                auto [va_l, va_bpc, va_a] = eval_batch(val_lo,   val_windows,   4);
                if (va_l < best_loss) {
                    best_loss = va_l; best_bpc_v = va_bpc; best_acc_v = va_a;
                    best_step = step; save_best(); stale = 0;
                } else {
                    stale++;
                }
                printf("step %5d   train_bpc = %.3f   val_bpc = %.3f   "
                       "val_loss = %.4f   val_acc = %5.1f%%   lr = %.2e%s\n",
                       step, tr_bpc, va_bpc, va_l, va_a, lr_at(step),
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
        val_bpc  = best_bpc_v;
        val_acc  = best_acc_v;

        copy_linear_weights(g_embed, embed);
        copy_linear_weights(g_proj, proj);
        for (int i = 0; i < cfg.num_blocks; ++i) copy_tf_weights(*g_blocks[i], *blocks[i]);

        save_cached_weights(cachepath, cfg, idx_to_char, val_loss, val_bpc, val_acc,
                            g_embed, g_blocks, g_proj);
        // ------------------------------------------------------------------------
    }

    list<Layer<FLOAT>*> gennn;
    gennn.push_back(&g_proj);
    for (int i = cfg.num_blocks - 1; i >= 0; --i) gennn.push_back(g_blocks[i].get());
    gennn.push_back(&g_embed);
    freeze(gennn);

    // Run the denoiser once on a length-seq_len token array (tok[s] in [0,V],
    // V == [MASK]); returns host logits (V, seq_len).
    auto denoise = [&](const vector<int>& tok) {
        auto enc = Matrix<float>::zeros(in_dim, seq_len);
        for (int s = 0; s < seq_len; ++s) {
            enc.elem(tok[s], s) = 1.0f;
            enc.elem(V1 + s, s) = 1.0f;
        }
        forward(gennn, Matrix<FLOAT>(enc));
        return as_host(g_proj.value());
    };

    auto sample_col = [&](const Matrix<float>& logits, int col, float temp,
                          int& tok_out, float& conf_out) {
        float mx = -1e30f;
        for (int r = 0; r < V; ++r) mx = max(mx, logits.elem(r, col));
        vector<float> p(V);
        float Z = 0.0f;
        for (int r = 0; r < V; ++r) { p[r] = expf((logits.elem(r, col) - mx) / temp); Z += p[r]; }
        float draw = u01(global_rand_gen) * Z, acc = 0.0f;
        int chosen = V - 1;
        for (int r = 0; r < V; ++r) { acc += p[r]; if (draw <= acc) { chosen = r; break; } }
        tok_out = chosen;
        conf_out = p[chosen] / Z;
    };

    auto gumbel = [&]() {
        float u = u01(global_rand_gen);
        return -logf(-logf(u > 1e-9f ? u : 1e-9f));
    };

    // MaskGIT-style progressive decoding (see demo_discretediffusion.cu),
    // reporting the sequence state after every round via `on_round`.
    const double PI = 3.14159265358979323846;
    auto tok_to_text = [&](const vector<int>& tok) {
        string out;
        for (int s = 0; s < seq_len; ++s) out += (tok[s] == MASK) ? '?' : idx_to_char[tok[s]];
        return out;
    };
    auto decode = [&](vector<int> tok, int T, float temp, float noise_scale,
                      const function<bool(int, const string&)>& on_round) {
        int n_masked = 0;
        for (int s = 0; s < seq_len; ++s) if (tok[s] == MASK) n_masked++;
        const int initial = n_masked;
        for (int k = 1; k <= T && n_masked > 0; ++k) {
            auto logits = denoise(tok);
            vector<int>   cand(seq_len, 0);
            vector<float> score(seq_len, -1e30f);
            const float anneal = noise_scale * (float)(T - k) / (float)T;
            for (int s = 0; s < seq_len; ++s) {
                if (tok[s] != MASK) continue;
                float conf; sample_col(logits, s, temp, cand[s], conf);
                score[s] = logf(conf > 1e-9f ? conf : 1e-9f) + anneal * gumbel();
            }

            int target = (k >= T) ? 0
                : (int)floor(initial * cos(0.5 * PI * (double)k / (double)T));
            target = min(target, n_masked);
            int reveal = n_masked - target;

            vector<int> order;
            for (int s = 0; s < seq_len; ++s) if (tok[s] == MASK) order.push_back(s);
            partial_sort(order.begin(), order.begin() + reveal, order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
            for (int j = 0; j < reveal; ++j) { tok[order[j]] = cand[order[j]]; n_masked--; }

            if (!on_round(k, tok_to_text(tok))) break;   // client disconnected
        }
        return tok_to_text(tok);
    };

    // map arbitrary user text into the model vocabulary (fold case if needed)
    bool in_vocab[256] = {};
    for (int i = 0; i < V; ++i) in_vocab[(unsigned char)idx_to_char[i]] = true;
    const char fallback_char = in_vocab[(unsigned char)' '] ? ' ' : idx_to_char[0];
    auto sanitize_char = [&](unsigned char u) -> char {
        if (in_vocab[u]) return (char)u;
        unsigned char lo = (unsigned char)tolower(u);
        return in_vocab[lo] ? (char)lo : fallback_char;
    };

    // the matrix library is not thread-safe; serialize decodes
    auto predict_mutex = make_shared<mutex>();

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        ifstream f(PROJECT_DIR + string("/examples/web/diffusion.html"));
        if (!f) {
            res.status = 500;
            res.set_content("examples/web/diffusion.html not found", "text/plain");
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
           << ",\"val_bpc\":" << val_bpc
           << ",\"val_acc\":" << val_acc
           << ",\"corpus\":\"" << json_escape(corpus_source) << "\"}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /fill?steps=64&temperature=0.7&noise=1.0, body = text where '_' is a
    // blank. The text is clipped to seq_len chars; anything beyond the text up
    // to seq_len is also treated as blank. Streams NDJSON: one line per
    // denoising round ({"round":k,"text":"..."}), then {"done":true,...}.
    svr.Post("/fill", [&](const httplib::Request& req, httplib::Response& res) {
        int T = 64;
        float temperature = 0.7f;
        float noise = 1.0f;
        if (req.has_param("steps")) T = atoi(req.get_param_value("steps").c_str());
        if (req.has_param("temperature")) temperature = atof(req.get_param_value("temperature").c_str());
        if (req.has_param("noise")) noise = atof(req.get_param_value("noise").c_str());
        T = max(1, min(T, 256));
        temperature = max(0.05f, min(temperature, 5.0f));
        noise = max(0.0f, min(noise, 5.0f));

        auto tok = make_shared<vector<int>>(seq_len, MASK);
        const string& body = req.body;
        for (int s = 0; s < seq_len && s < (int)body.size(); ++s) {
            const unsigned char u = (unsigned char)body[s];
            (*tok)[s] = (u == '_') ? MASK : c2i[(unsigned char)sanitize_char(u)];
        }

        res.set_chunked_content_provider(
            "application/x-ndjson",
            [&, tok, T, temperature, noise](size_t, httplib::DataSink& sink) {
                lock_guard<mutex> lock(*predict_mutex);
                auto emit = [&](const string& line) {
                    return sink.write(line.c_str(), line.size());
                };
                emit("{\"round\":0,\"text\":\"" + json_escape(tok_to_text(*tok)) + "\"}\n");
                string final_text = decode(*tok, T, temperature, noise,
                    [&](int k, const string& text) {
                        return emit("{\"round\":" + to_string(k) + ",\"text\":\""
                                    + json_escape(text) + "\"}\n");
                    });
                emit("{\"done\":true,\"text\":\"" + json_escape(final_text) + "\"}\n");
                sink.done();
                return true;
            });
    });

    int port = 8095;
    if (const char* p = getenv("PORT")) port = atoi(p);
    cout << "Serving discrete-diffusion web demo on http://localhost:" << port << endl;
    if (!svr.listen("0.0.0.0", port)) {
        cout << "Failed to listen on port " << port << endl;
        return 1;
    }

    return 0;
}
