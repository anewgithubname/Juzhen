/**
 * @file demo_tinystories.cu
 * @brief BPE-token autoregressive transformer trained on TinyStories.
 *
 * The first token-level (rather than character-level) language model in this
 * codebase. Data preparation is offline (scripts/tinystories_prepare.py):
 * a BPE-4096 tokenizer encodes the ~2.2 GB TinyStories V2 corpus into flat
 * uint16 id files. This program:
 *
 *   - loads train.bin / val.bin / tokenizer.txt from the external scratch
 *     disk (TS_DATA env overrides the default path)
 *   - builds (token one-hot + position one-hot) inputs ON THE GPU from a
 *     small (1, N) matrix of token ids — with V=4096 a host-side one-hot
 *     would mean hundreds of MB of PCIe traffic per step; instead a tiny
 *     kernel scatters the ones directly in device memory and the embedding
 *     is the existing LinearLayer GEMM (no new trainable layers)
 *   - trains embed -> N x causal TransformerLayer -> proj with the framework
 *     LogisticLayer (softmax CE over the 4096-token vocabulary)
 *   - caches weights (TS_WEIGHTS env overrides; checkpointed every >=5000
 *     steps during training, so an interrupted run still yields a model)
 *   - after training (or from the cache), generates sample stories seeded
 *     with validation-story openings
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <memory>
#include <random>
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

// ── GPU one-hot builders ─────────────────────────────────────────────────────
// ids arrive as a (1, N) float matrix (token ids < 4096 are exact in fp32).
#ifdef CUDA
__global__ void ts_onehot_input_kernel(float* X, const float* ids,
                                       int in_dim, int V, int seq, size_t N) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float* col = X + j * in_dim;
    col[(int)ids[j]] = 1.0f;              // token one-hot
    col[V + (int)(j % seq)] = 1.0f;       // position one-hot
}

__global__ void ts_onehot_target_kernel(float* Y, const float* ids,
                                        int V, size_t N) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    Y[j * V + (int)ids[j]] = 1.0f;
}
#endif

// ── token / tokenizer IO ─────────────────────────────────────────────────────
static string data_dir() {
    if (const char* p = getenv("TS_DATA")) return p;
    return "/mnt/external_hdd/data/nlp/tinystories";
}

static string weights_path() {
    if (const char* p = getenv("TS_WEIGHTS")) return p;
    return data_dir() + "/tinystories.weights";
}

static vector<uint16_t> load_bin(const string& path) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f) { cout << "cannot open " << path << endl; exit(1); }
    const streamsize n = f.tellg();
    f.seekg(0);
    vector<uint16_t> v(n / 2);
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// vocab strings from tokenizer.txt ("VOCAB n" header, one escaped token/line)
static vector<string> load_vocab(const string& path) {
    ifstream f(path);
    if (!f) { cout << "cannot open " << path << endl; exit(1); }
    string word;
    int n;
    f >> word >> n;
    f.ignore();
    vector<string> vocab(n);
    for (int i = 0; i < n; ++i) {
        string line, out;
        getline(f, line);
        for (size_t k = 0; k < line.size(); ++k) {
            if (line[k] == '\\' && k + 1 < line.size()) {
                const char c = line[++k];
                out += (c == 'n') ? '\n' : (c == 'r') ? '\r' : c;
            } else {
                out += line[k];
            }
        }
        vocab[i] = out;
    }
    return vocab;
}

// Metaspace decode: "▁" (U+2581) marks a leading space.
static string decode_tokens(const vector<int>& ids, const vector<string>& vocab) {
    string s;
    for (int id : ids)
        if (id >= 0 && id < (int)vocab.size()) s += vocab[id];
    string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i + 2 < s.size() && (unsigned char)s[i] == 0xE2 &&
            (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
            out += ' ';
            i += 2;
        } else {
            out += s[i];
        }
    }
    if (!out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
}

// ── weight cache (same scheme as demo_transformer_web) ──────────────────────
static const int WEIGHTS_MAGIC = 0x4a5a5453;  // "JZTS"

struct ModelConfig {
    int seq_len, d_model, d_k, d_ff, num_heads, num_blocks;
};

static bool read_cache_header(FILE* fp, const ModelConfig& cfg, int V,
                              float& val_loss, float& val_acc) {
    int header[8] = {};
    if (fread(header, sizeof(int), 8, fp) != 8) return false;
    if (header[0] != WEIGHTS_MAGIC) return false;
    if (header[1] != V || header[2] != cfg.seq_len || header[3] != cfg.d_model ||
        header[4] != cfg.d_k || header[5] != cfg.d_ff ||
        header[6] != cfg.num_heads || header[7] != cfg.num_blocks) return false;
    if (fread(&val_loss, sizeof(float), 1, fp) != 1) return false;
    if (fread(&val_acc, sizeof(float), 1, fp) != 1) return false;
    return true;
}

static void load_cached_weights(FILE* fp, int num_blocks,
                                LinearLayer<FLOAT>& emb,
                                vector<unique_ptr<TransformerLayer<FLOAT>>>& blocks,
                                LinearLayer<FLOAT>& proj) {
    Matrix<float> M("cache", 1, 1);
    auto next = [&]() -> Matrix<float> { read(fp, M); return M; };
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

static void save_cached_weights(const string& path, const ModelConfig& cfg, int V,
                                float val_loss, float val_acc,
                                LinearLayer<FLOAT>& emb,
                                vector<unique_ptr<TransformerLayer<FLOAT>>>& blocks,
                                LinearLayer<FLOAT>& proj) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        cout << "WARNING: cannot write weight cache " << path << endl;
        return;
    }
    int header[8] = {WEIGHTS_MAGIC, V, cfg.seq_len, cfg.d_model,
                     cfg.d_k, cfg.d_ff, cfg.num_heads, cfg.num_blocks};
    fwrite(header, sizeof(int), 8, fp);
    fwrite(&val_loss, sizeof(float), 1, fp);
    fwrite(&val_acc, sizeof(float), 1, fp);
    auto put = [&](const Matrix<float>& M) { write(fp, M); };
    put(as_host(emb.W()));
    put(as_host(emb.b()));
    for (auto& bl : blocks) {
        auto& l = *bl;
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

    const string base = data_dir();
    const vector<string> vocab = load_vocab(base + "/tokenizer.txt");
    const int V = (int)vocab.size();
    int eot_id = 0;
    for (int i = 0; i < V; ++i)
        if (vocab[i] == "<|endoftext|>") { eot_id = i; break; }

    const ModelConfig cfg = {/*seq_len*/ 256, /*d_model*/ 512, /*d_k*/ 512,
                             /*d_ff*/ 2048, /*num_heads*/ 8, /*num_blocks*/ 8};
    const int seq_len = cfg.seq_len;
    const int in_dim = V + seq_len;

    cout << "TinyStories token LM: vocab " << V << " (BPE), seq_len " << seq_len << endl;

    // (1, n) host matrix of ids -> (in_dim, n) one-hot input on device
    auto make_X = [&](const Matrix<float>& ids_h) {
        const size_t n = ids_h.num_col();
#ifdef CUDA
        auto ids = Matrix<CUDAfloat>(ids_h);
        auto X = Matrix<CUDAfloat>::zeros(in_dim, n);
        const int threads = 256;
        const int nblocks = (int)((n + threads - 1) / threads);
        ts_onehot_input_kernel<<<nblocks, threads>>>(
            const_cast<float*>(reinterpret_cast<const float*>(X.data())),
            reinterpret_cast<const float*>(ids.data()),
            in_dim, V, seq_len, n);
        CudaErrorCheck(cudaGetLastError());
        return X;
#else
        Matrix<float> X("X", in_dim, n);
        X.zeros();
        for (size_t j = 0; j < n; ++j) {
            X.elem((int)ids_h.elem(0, j), j) = 1.0f;
            X.elem(V + (j % seq_len), j) = 1.0f;
        }
        return Matrix<FLOAT>(X);   // host->device (no-op copy when FLOAT==float)
#endif
    };
    auto make_Y = [&](const Matrix<float>& ids_h) {
        const size_t n = ids_h.num_col();
#ifdef CUDA
        auto ids = Matrix<CUDAfloat>(ids_h);
        auto Y = Matrix<CUDAfloat>::zeros(V, n);
        const int threads = 256;
        const int nblocks = (int)((n + threads - 1) / threads);
        ts_onehot_target_kernel<<<nblocks, threads>>>(
            const_cast<float*>(reinterpret_cast<const float*>(Y.data())),
            reinterpret_cast<const float*>(ids.data()), V, n);
        CudaErrorCheck(cudaGetLastError());
        return Y;
#else
        Matrix<float> Y("Y", V, n);
        Y.zeros();
        for (size_t j = 0; j < n; ++j) Y.elem((int)ids_h.elem(0, j), j) = 1.0f;
        return Matrix<FLOAT>(Y);   // host->device (no-op copy when FLOAT==float)
#endif
    };

    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < cfg.num_blocks; ++i)
            v.push_back(make_unique<TF>(cfg.d_model, cfg.d_k, cfg.d_ff, seq_len, b,
                                        cfg.num_heads));   // causal by default
        return v;
    };
    auto build_net = [&](vector<unique_ptr<TF>>& bl,
                         LinearLayer<FLOAT>& emb, LinearLayer<FLOAT>& pr) {
        list<Layer<FLOAT>*> net;
        net.push_back(&pr);
        for (int i = cfg.num_blocks - 1; i >= 0; --i) net.push_back(bl[i].get());
        net.push_back(&emb);
        return net;
    };

    // Inference network (batch = 1), filled from the cache or training below.
    LinearLayer<FLOAT> g_embed(cfg.d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, cfg.d_model, seq_len);

    float val_loss = -1, val_acc = -1;
    FILE* cachefp = fopen(weights_path().c_str(), "rb");
    bool cached = false;
    if (cachefp) {
        cached = read_cache_header(cachefp, cfg, V, val_loss, val_acc);
        if (!cached) {
            fclose(cachefp);
            cachefp = nullptr;
            cout << "Weight cache is stale or invalid; retraining." << endl;
        }
    }

    if (cached) {
        load_cached_weights(cachefp, cfg.num_blocks, g_embed, g_blocks, g_proj);
        fclose(cachefp);
        cout << "Loaded weight cache " << weights_path() << " (val_loss = " << val_loss
             << ", val_acc = " << val_acc << "%). Delete it to retrain." << endl;
    } else {
        // ─────────────────────────── training ───────────────────────────
        const vector<uint16_t> train_ids = load_bin(base + "/train.bin");
        const vector<uint16_t> val_ids = load_bin(base + "/val.bin");
        cout << "tokens: " << train_ids.size() << " train, " << val_ids.size()
             << " val" << endl;

        int steps = 30000;
        if (const char* s = getenv("STEPS")) steps = max(1, atoi(s));
        int batch = 64;
        if (const char* s = getenv("BATCH")) batch = max(1, atoi(s));
        const int log_every = 500;
        const int patience = 12;
        const int N = seq_len * batch;

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

        cout << "Network: embed(" << in_dim << "->" << cfg.d_model << ") -> Transformer("
             << cfg.d_model << "," << cfg.d_k << "," << cfg.d_ff << "," << cfg.num_heads
             << "h,causal) [x" << cfg.num_blocks << "] -> proj(" << cfg.d_model
             << "->" << V << ")" << endl;
        cout << "Training: batch " << batch << " x seq " << seq_len << " = " << N
             << " tokens/step, steps " << steps << endl << endl;

        LinearLayer<FLOAT> embed(cfg.d_model, in_dim, N);
        auto blocks = make_blocks(batch);
        LinearLayer<FLOAT> proj(V, cfg.d_model, N);

        // batch=1: snapshots store weights only, skip big activation caches
        LinearLayer<FLOAT> best_embed(cfg.d_model, in_dim, 1);
        auto best_blocks = make_blocks(1);
        LinearLayer<FLOAT> best_proj(V, cfg.d_model, 1);

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

        Matrix<float> xids("xids", 1, N), yids("yids", 1, N);
        auto fill_ids = [&](const vector<uint16_t>& corpus, const vector<long>& offs) {
            for (int b = 0; b < (int)offs.size(); ++b) {
                const long off = offs[b];
                for (int s = 0; s < seq_len; ++s) {
                    const int col = b * seq_len + s;
                    xids.elem(0, col) = (float)corpus[off + s];
                    yids.elem(0, col) = (float)corpus[off + s + 1];
                }
            }
        };

        const long train_hi = (long)train_ids.size() - seq_len - 1;
        const long val_hi = (long)val_ids.size() - seq_len - 1;
        if (train_hi <= 0 || val_hi <= 0) { cout << "corpus too small" << endl; return 1; }
        uniform_int_distribution<long> pick(0, train_hi - 1);
        vector<long> offs(batch);

        // fixed evenly-spaced val batch; returns {loss, top-1 accuracy%}
        auto eval_val = [&]() {
            for (int b = 0; b < batch; ++b) offs[b] = (long)((double)b * val_hi / batch);
            fill_ids(val_ids, offs);
            auto X = make_X(xids);
            auto Y = make_Y(yids);
            LogisticLayer<FLOAT> el(N, std::move(Y));
            auto net = build_net(blocks, embed, proj);
            net.push_front(&el);
            const float loss = as_host(forward(net, X)).elem(0, 0);
            auto logits = as_host(proj.value());
            long correct = 0;
            for (int i = 0; i < N; ++i) {
                int pred = 0;
                for (int r = 1; r < V; ++r)
                    if (logits.elem(r, i) > logits.elem(pred, i)) pred = r;
                if (pred == (int)yids.elem(0, i)) correct++;
            }
            return make_pair(loss, 100.0f * (float)correct / (float)N);
        };

        float best_loss = numeric_limits<float>::infinity();
        float best_acc = 0;
        int best_step = 0, stale = 0, last_ckpt_step = 0;

        for (int step = 0; step < steps; ++step) {
            set_all_lr(lr_at(step));

            for (int b = 0; b < batch; ++b) offs[b] = pick(global_rand_gen);
            fill_ids(train_ids, offs);
            auto X = make_X(xids);
            auto Y = make_Y(yids);

            forward(trainnn, X);
            LogisticLayer<FLOAT> loss(N, std::move(Y));
            trainnn.push_front(&loss);
            backprop(trainnn, X);
            trainnn.pop_front();

            if (step % log_every == 0 || step == steps - 1) {
                auto [va_l, va_a] = eval_val();
                if (va_l < best_loss) {
                    best_loss = va_l;
                    best_acc = va_a;
                    best_step = step;
                    save_best();
                    stale = 0;
                    // crash-safe checkpoint of the best snapshot
                    if (step - last_ckpt_step >= 5000) {
                        save_cached_weights(weights_path(), cfg, V, best_loss, best_acc,
                                            best_embed, best_blocks, best_proj);
                        last_ckpt_step = step;
                    }
                } else {
                    stale++;
                }
                printf("step %5d   val_loss = %.4f   val_ppl = %7.2f   val_acc = %5.1f%%   "
                       "lr = %.2e%s\n",
                       step, va_l, expf(va_l), va_a, lr_at(step),
                       stale == 0 ? "   *best" : "");
                fflush(stdout);
                if (stale >= patience) {
                    printf("Early stopping at step %d (best val_loss = %.4f at step %d).\n",
                           step, best_loss, best_step);
                    break;
                }
            }
        }

        restore_best();
        val_loss = best_loss;
        val_acc = best_acc;

        copy_linear_weights(g_embed, embed);
        copy_linear_weights(g_proj, proj);
        for (int i = 0; i < cfg.num_blocks; ++i) copy_tf_weights(*g_blocks[i], *blocks[i]);
        save_cached_weights(weights_path(), cfg, V, val_loss, val_acc,
                            g_embed, g_blocks, g_proj);
        // ──────────────────────────────────────────────────────────────────
    }

    list<Layer<FLOAT>*> gennn = build_net(g_blocks, g_embed, g_proj);
    freeze(gennn);

    // ── sample stories: seed with the openings of validation stories ───────
    const vector<uint16_t> val_ids = load_bin(base + "/val.bin");
    uniform_real_distribution<float> u01(0.0f, 1.0f);

    // autoregressive top-k sampling with an EOT-padded sliding window
    auto generate = [&](const vector<int>& ctx, int gen_len, float temp, int topk) {
        vector<int> out = ctx;
        Matrix<float> ids_h("gen_ids", 1, seq_len);
        for (int i = 0; i < gen_len; ++i) {
            for (int s = 0; s < seq_len; ++s) {
                const int pos = (int)out.size() - seq_len + s;
                ids_h.elem(0, s) = (float)(pos >= 0 ? out[pos] : eot_id);
            }
            forward(gennn, make_X(ids_h));
            auto logits = as_host(g_proj.value());

            vector<pair<float, int>> ranked(V);
            for (int r = 0; r < V; ++r) ranked[r] = {logits.elem(r, seq_len - 1), r};
            const int k = min(topk, V);
            partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
            const float mx = ranked[0].first;
            float Z = 0;
            vector<float> cdf(k);
            for (int r = 0; r < k; ++r) {
                Z += expf((ranked[r].first - mx) / temp);
                cdf[r] = Z;
            }
            const float draw = u01(global_rand_gen) * Z;
            int next = ranked[k - 1].second;
            for (int r = 0; r < k; ++r)
                if (draw <= cdf[r]) { next = ranked[r].second; break; }

            if (next == eot_id) break;   // story finished
            out.push_back(next);
        }
        return vector<int>(out.begin() + ctx.size(), out.end());
    };

    cout << "\n--- Sample stories (seeded from validation openings) ---\n";
    for (int k = 0; k < 3; ++k) {
        // find a story start: the token right after an EOT
        long p = (long)((double)(k + 1) * val_ids.size() / 4);
        while (p + 9 < (long)val_ids.size() && val_ids[p] != eot_id) p++;
        p++;
        vector<int> seed(val_ids.begin() + p, val_ids.begin() + p + 8);
        auto gen = generate(seed, 250, 0.8f, 40);
        printf("[seed]  %s\n[story] %s\n\n",
               decode_tokens(seed, vocab).c_str(),
               decode_tokens(gen, vocab).c_str());
    }

    return 0;
}
