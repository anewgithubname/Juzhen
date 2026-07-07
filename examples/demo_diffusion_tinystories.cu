/**
 * @file demo_diffusion_tinystories.cu
 * @brief Masked (absorbing-state) discrete diffusion LM on TinyStories BPE tokens.
 *
 * This combines the two existing lines of work in this codebase:
 *   - the TOKEN-LEVEL data pipeline of demo_tinystories.cu (BPE-4096 ids from
 *     scripts/tinystories_prepare.py, loaded from the external scratch disk;
 *     TS_DATA env overrides the default path), and
 *   - the DIFFUSION training recipe of demo_discretediffusion.cu (absorbing
 *     state D3PM / MDLM: corrupt a clean window by replacing a random fraction
 *     of tokens with [MASK], train a bidirectional transformer to recover the
 *     originals with a 1/t-weighted masked cross-entropy).
 *
 * Differences from the char-level diffusion demo:
 *   - vocabulary is the 4096-token BPE vocab plus one [MASK] id (4096), so
 *     one-hot inputs are built ON THE GPU exactly as in demo_tinystories.cu
 *     (a host-side one-hot would be hundreds of MB of PCIe traffic per step)
 *   - the eval batch uses a FIXED noise seed, so successive evaluations are
 *     directly comparable and early stopping is not driven by eval noise
 *   - weights are cached/checkpointed to the external disk (TSDIFF_WEIGHTS
 *     env overrides; checkpointed every >=5000 steps during training, so an
 *     interrupted run still yields a model)
 *   - generation: MaskGIT-style progressive decoding — unconditional from
 *     all-[MASK], and prompt-conditioned with openings of validation stories
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
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
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

// ────────────────────────────────────────────────────────────────────────────
// Custom loss: masked, 1/t-weighted softmax cross-entropy (same as the char
// diffusion demo). Columns are (position, sequence) pairs; w=0 for unmasked
// tokens, w=(1/t)/N for a token masked at level t — the MDLM continuous-time
// ELBO weight (linear schedule), per-token normalized.
//   loss   = sum_c  w_c * ( logsumexp(z_c) - <z_c, y_c> )
//   dloss/dz_c = w_c * ( softmax(z_c) - y_c )
// ────────────────────────────────────────────────────────────────────────────
template <class D>
static Matrix<D> col_max(const Matrix<D>& input) {
    return reduce(
        [] __GPU_CPU__(float* v, float* vdes, int lenv, int) {
            float m = -1e30f;
            for (int i = 0; i < lenv; i++) m = m > v[i] ? m : v[i];
            vdes[0] = m;
        }, input, 0, 1);
}

template <class D>
class MaskedCELayer : public Layer<D> {
    Matrix<D> output;   // (V, N) one-hot true tokens
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
        auto mx = col_max(input);              // (1, N)
        auto E = exp(input - oneK1 * mx);      // (V, N)
        auto Z = oneK1 * sum(E, 0);            // (V, N) column sums broadcast
        auto softmax = E / std::move(Z);       // (V, N)
        return hadmd(softmax - output, oneK1 * wrow);
    }

    void eval(const Matrix<D>& input) override {
        auto mx = col_max(input);                               // (1, N)
        auto lse = log(sum(exp(input - oneK1 * mx), 0)) + mx;   // (1, N)
        auto dot = sum(hadmd(input, output), 0);                // (1, N) <z, y>
        auto ce = lse - dot;                                    // (1, N)
        Layer<D>::val = sum(hadmd(wrow, ce), 1);                // (1, 1)
    }
};

// ── GPU one-hot builders ─────────────────────────────────────────────────────
// ids arrive as a (1, N) float matrix. Input ids may equal MASK (= V), so the
// token block of the input one-hot has V+1 slots; targets are always real
// tokens (V slots).
#ifdef CUDA
__global__ void dd_onehot_input_kernel(float* X, const float* ids,
                                       int in_dim, int V1, int seq, size_t N) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float* col = X + j * in_dim;
    col[(int)ids[j]] = 1.0f;               // token one-hot (index V == MASK)
    col[V1 + (int)(j % seq)] = 1.0f;       // position one-hot
}

__global__ void dd_onehot_target_kernel(float* Y, const float* ids,
                                        int V, size_t N) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    Y[j * V + (int)ids[j]] = 1.0f;
}
#endif

// ── token / tokenizer IO (same as demo_tinystories.cu) ──────────────────────
static string data_dir() {
    if (const char* p = getenv("TS_DATA")) return p;
    return "/mnt/external_hdd/data/nlp/tinystories";
}

static string weights_path() {
    if (const char* p = getenv("TSDIFF_WEIGHTS")) return p;
    return data_dir() + "/tinystories_diffusion.weights";
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

// ── BPE encoder (same as demo_tinystories_web.cu) for GPT-Eval prompts ──────
struct Tokenizer {
    vector<string> vocab;                       // id -> token string (may contain ▁)
    unordered_map<string, int> str2id;          // token string -> id
    // merge (a,b) -> {rank, merged_id}; lower rank = higher priority
    map<pair<int, int>, pair<int, int>> merges;

    static string unescape(const string& line) {
        string out;
        for (size_t k = 0; k < line.size(); ++k) {
            if (line[k] == '\\' && k + 1 < line.size()) {
                const char c = line[++k];
                out += (c == 'n') ? '\n' : (c == 'r') ? '\r' : c;
            } else out += line[k];
        }
        return out;
    }

    void load(const string& path) {
        ifstream f(path);
        if (!f) { cout << "cannot open " << path << endl; exit(1); }
        string key; int n;
        f >> key >> n; f.ignore();              // "VOCAB n"
        vocab.resize(n);
        for (int i = 0; i < n; ++i) {
            string line; getline(f, line);
            vocab[i] = unescape(line);
            str2id[vocab[i]] = i;
        }
        f >> key >> n; f.ignore();              // "MERGES m"
        for (int r = 0; r < n; ++r) {
            int a, b, c; f >> a >> b >> c;
            merges[{a, b}] = {r, c};
        }
    }

    // split a UTF-8 string into a vector of single-character substrings
    static vector<string> utf8_chars(const string& s) {
        vector<string> out;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];
            int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            out.push_back(s.substr(i, len));
            i += len;
        }
        return out;
    }

    // Metaspace + BPE encode: user text -> token ids.
    vector<int> encode(const string& text) const {
        string ms = "\xE2\x96\x81";             // leading ▁
        for (char ch : text) {
            if (ch == ' ' || ch == '\n' || ch == '\t') ms += "\xE2\x96\x81";
            else ms += ch;
        }
        vector<int> syms;
        for (const string& ch : utf8_chars(ms)) {
            auto it = str2id.find(ch);
            if (it != str2id.end()) syms.push_back(it->second);
        }
        // greedy BPE: repeatedly merge the adjacent pair of lowest rank
        while (syms.size() >= 2) {
            int best_rank = INT32_MAX, best_i = -1, best_merged = -1;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = merges.find({syms[i], syms[i + 1]});
                if (it != merges.end() && it->second.first < best_rank) {
                    best_rank = it->second.first;
                    best_i = (int)i;
                    best_merged = it->second.second;
                }
            }
            if (best_i < 0) break;
            syms[best_i] = best_merged;
            syms.erase(syms.begin() + best_i + 1);
        }
        return syms;
    }
};

static string json_escape(const string& s) {
    string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, 8, "\\u%04x", c); o += buf; }
                else o += (char)c;
        }
    }
    return o;
}

// ── weight cache (same scheme as demo_tinystories, different magic) ─────────
static const int WEIGHTS_MAGIC = 0x4a5a4444;  // "JZDD"

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
    uniform_real_distribution<float> u01(0.0f, 1.0f);

    const string base = data_dir();
    const vector<string> vocab = load_vocab(base + "/tokenizer.txt");
    const int V = (int)vocab.size();
    int eot_id = 0;
    for (int i = 0; i < V; ++i)
        if (vocab[i] == "<|endoftext|>") { eot_id = i; break; }

    // The denoiser reads a (V+1)-way token one-hot: the extra slot V is the
    // [MASK] token. It only ever predicts the V real tokens.
    const int MASK = V;
    const int V1 = V + 1;

    const ModelConfig cfg = {/*seq_len*/ 256, /*d_model*/ 512, /*d_k*/ 512,
                             /*d_ff*/ 2048, /*num_heads*/ 8, /*num_blocks*/ 8};
    const int seq_len = cfg.seq_len;
    const int in_dim = V1 + seq_len;
    const float t_min = 1e-3f;     // floor on the mask level (bounds the 1/t weight)

    cout << "TinyStories masked-diffusion LM: vocab " << V << " (BPE) + 1 [MASK], seq_len "
         << seq_len << endl;

    // (1, n) host matrix of ids (may include MASK) -> (in_dim, n) one-hot input
    auto make_X = [&](const Matrix<float>& ids_h) {
        const size_t n = ids_h.num_col();
#ifdef CUDA
        auto ids = Matrix<CUDAfloat>(ids_h);
        auto X = Matrix<CUDAfloat>::zeros(in_dim, n);
        const int threads = 256;
        const int nblocks = (int)((n + threads - 1) / threads);
        dd_onehot_input_kernel<<<nblocks, threads>>>(
            const_cast<float*>(reinterpret_cast<const float*>(X.data())),
            reinterpret_cast<const float*>(ids.data()),
            in_dim, V1, seq_len, n);
        CudaErrorCheck(cudaGetLastError());
        return X;
#else
        Matrix<float> X("X", in_dim, n);
        X.zeros();
        for (size_t j = 0; j < n; ++j) {
            X.elem((int)ids_h.elem(0, j), j) = 1.0f;
            X.elem(V1 + (j % seq_len), j) = 1.0f;
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
        dd_onehot_target_kernel<<<nblocks, threads>>>(
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
                                        cfg.num_heads, /*causal=*/false));
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

        // Hyperparameters (steps, batch, LR schedule, eval cadence, patience)
        // deliberately match demo_tinystories.cu so AR-vs-diffusion numbers on
        // the same data are directly comparable.
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
             << "h,bidir) [x" << cfg.num_blocks << "] -> proj(" << cfg.d_model
             << "->" << V << ")" << endl;
        cout << "Noising: replace each token with [MASK] at prob t~U(" << t_min
             << ",1); loss = (1/t)-weighted CE on masked tokens" << endl;
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

        // Per-batch host buffers. xids carries the CORRUPTED ids (MASK where
        // noised), yids the clean ids, Wh the per-column ELBO weights.
        Matrix<float> xids("xids", 1, N), yids("yids", 1, N);
        auto Wh = Matrix<float>::zeros(1, N);
        vector<char> masked(N);
        vector<long> offs(batch);
        vector<float> tlvl(batch);

        // Antithetic / stratified sampling of the per-sequence mask level t
        // (one sample per stratum of (t_min,1], shared jitter) — the standard
        // variance reduction for the high-variance 1/t ELBO weight.
        auto stratified_t = [&](vector<float>& t, mt19937& rng) {
            uniform_real_distribution<float> u(0.0f, 1.0f);
            const float jitter = u(rng);
            for (int b = 0; b < (int)t.size(); ++b)
                t[b] = t_min + (1.0f - t_min) * ((float)b + jitter) / (float)t.size();
            shuffle(t.begin(), t.end(), rng);   // decorrelate t from window
        };

        auto fill_batch = [&](const vector<uint16_t>& corpus, mt19937& rng) {
            uniform_real_distribution<float> u(0.0f, 1.0f);
            for (int b = 0; b < batch; ++b) {
                const long off = offs[b];
                const float t = tlvl[b];
                for (int s = 0; s < seq_len; ++s) {
                    const int col = b * seq_len + s;
                    const int c = (int)corpus[off + s];
                    const bool m = (u(rng) < t);
                    xids.elem(0, col) = (float)(m ? MASK : c);
                    yids.elem(0, col) = (float)c;
                    Wh.elem(0, col) = m ? (1.0f / t) / (float)N : 0.0f;
                    masked[col] = m ? 1 : 0;
                }
            }
        };

        const long train_hi = (long)train_ids.size() - seq_len;
        const long val_hi = (long)val_ids.size() - seq_len;
        if (train_hi <= 0 || val_hi <= 0) { cout << "corpus too small" << endl; return 1; }
        uniform_int_distribution<long> pick(0, train_hi - 1);

        // Fixed evenly-spaced val batch with a FIXED noise seed, so evals are
        // deterministic and comparable across steps (early stopping is then
        // driven by the model, not by eval-noise luck). Averages `draws`
        // independent noise realizations. Returns {loss, masked accuracy%}.
        auto eval_val = [&](int draws) {
            mt19937 eval_rng(12345);
            float loss_sum = 0.0f;
            long long correct = 0, total = 0;
            for (int d = 0; d < draws; ++d) {
                for (int b = 0; b < batch; ++b) offs[b] = (long)((double)b * val_hi / batch);
                stratified_t(tlvl, eval_rng);
                fill_batch(val_ids, eval_rng);
                auto X = make_X(xids);
                MaskedCELayer<FLOAT> el(N, make_Y(yids), Matrix<FLOAT>(Wh));
                auto net = build_net(blocks, embed, proj);
                net.push_front(&el);
                loss_sum += as_host(forward(net, X)).elem(0, 0);
                auto logits = as_host(proj.value());
                for (int i = 0; i < N; ++i) {
                    if (!masked[i]) continue;
                    int pred = 0;
                    for (int r = 1; r < V; ++r)
                        if (logits.elem(r, i) > logits.elem(pred, i)) pred = r;
                    if (pred == (int)yids.elem(0, i)) correct++;
                    total++;
                }
            }
            const float loss = loss_sum / draws;   // per-token NLL bound (nats)
            const float acc = total ? 100.0f * (float)correct / (float)total : 0.0f;
            return make_pair(loss, acc);
        };

        float best_loss = numeric_limits<float>::infinity();
        float best_acc = 0;
        int best_step = 0, stale = 0, last_ckpt_step = 0;

        for (int step = 0; step < steps; ++step) {
            set_all_lr(lr_at(step));

            for (int b = 0; b < batch; ++b) offs[b] = pick(global_rand_gen);
            stratified_t(tlvl, global_rand_gen);
            fill_batch(train_ids, global_rand_gen);
            auto X = make_X(xids);

            forward(trainnn, X);
            MaskedCELayer<FLOAT> loss(N, make_Y(yids), Matrix<FLOAT>(Wh));
            trainnn.push_front(&loss);
            backprop(trainnn, X);
            trainnn.pop_front();

            if (step % log_every == 0 || step == steps - 1) {
                auto [va_l, va_a] = eval_val(2);
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
                printf("step %5d   val_loss = %.4f   val_ppl = %8.2f   masked_acc = %5.1f%%   "
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

    // ── MaskGIT-style progressive decoding ──────────────────────────────────
    // Run the denoiser once on a length-seq_len token array (tok[s] in [0,V]
    // with V = [MASK]); returns host logits (V, seq_len).
    auto denoise = [&](const vector<int>& tok) {
        Matrix<float> ids_h("gen_ids", 1, seq_len);
        for (int s = 0; s < seq_len; ++s) ids_h.elem(0, s) = (float)tok[s];
        forward(gennn, make_X(ids_h));
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

    // Over T rounds, predict all masked positions, then permanently commit a
    // growing fraction per a cosine schedule until none remain. Commits are
    // ranked by log-confidence plus ANNEALED Gumbel noise: early rounds
    // explore, late rounds fall back to greedy confidence.
    const double PI = 3.14159265358979323846;
    auto decode = [&](vector<int> tok, int T, float temp, float noise_scale) {
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
        }
        return tok;
    };

    // Trim for display: drop everything from the first <|endoftext|> onward.
    auto trim_at_eot = [&](const vector<int>& tok, int from) {
        vector<int> out;
        for (int s = from; s < (int)tok.size(); ++s) {
            if (tok[s] == eot_id) break;
            out.push_back(tok[s]);
        }
        return out;
    };

    const int T_steps = 64;   // ~4 tokens committed per round over seq_len=256

    // ── GPT-Eval batch mode ─────────────────────────────────────────────────
    // GPTEVAL_PROMPTS = text file, one story opening per line. Each prompt is
    // BPE-encoded, kept fixed as context, and the rest of the window is decoded
    // from [MASK]; completions are written as JSON compatible with
    // scripts/tinystories_gpteval.py's grade phase (GPTEVAL_OUT overrides the
    // output path). Skips the demo sample sections.
    if (const char* pf = getenv("GPTEVAL_PROMPTS")) {
        const string outpath = getenv("GPTEVAL_OUT")
            ? getenv("GPTEVAL_OUT") : base + "/gpteval_completions_diffusion.json";
        Tokenizer tok;
        tok.load(base + "/tokenizer.txt");
        ifstream pfile(pf);
        if (!pfile) { cout << "cannot open " << pf << endl; return 1; }
        vector<string> prompts;
        string line;
        while (getline(pfile, line)) if (!line.empty()) prompts.push_back(line);
        ofstream jf(outpath);
        jf << "[\n";
        for (size_t i = 0; i < prompts.size(); ++i) {
            vector<int> ids = tok.encode(prompts[i]);
            const int plen = min((int)ids.size(), seq_len / 2);
            vector<int> t(seq_len, MASK);
            for (int s = 0; s < plen; ++s) t[s] = ids[s];
            t = decode(t, T_steps, 0.8f, 1.0f);
            const string completion = decode_tokens(trim_at_eot(t, plen), vocab);
            jf << "  {\"prompt\": \"" << json_escape(prompts[i])
               << "\", \"completion\": \"" << json_escape(completion) << "\"}"
               << (i + 1 < prompts.size() ? "," : "") << "\n";
            printf("[%zu/%zu] %.50s... -> %zu chars\n",
                   i + 1, prompts.size(), prompts[i].c_str(), completion.size());
            fflush(stdout);
        }
        jf << "]\n";
        cout << "saved " << prompts.size() << " completions to " << outpath << endl;
        return 0;
    }

    cout << "\n--- Unconditional generation (from all-[MASK], " << T_steps
         << " denoising steps) ---\n";
    for (int k = 0; k < 2; ++k) {
        vector<int> tok(seq_len, MASK);
        tok = decode(tok, T_steps, 1.0f, 2.0f);
        printf("[%d] %s\n\n", k, decode_tokens(trim_at_eot(tok, 0), vocab).c_str());
    }

    // Prompt-conditioned: keep the opening tokens of a validation story as
    // fixed context, mask the rest, and let the model write the story.
    const vector<uint16_t> val_ids = load_bin(base + "/val.bin");
    const int prompt_len = 16;
    cout << "--- Prompt-conditioned generation (openings of validation stories) ---\n";
    for (int k = 0; k < 3; ++k) {
        // find a story start: the token right after an EOT
        long p = (long)((double)(k + 1) * val_ids.size() / 4);
        while (p + prompt_len + 1 < (long)val_ids.size() && val_ids[p] != eot_id) p++;
        p++;
        vector<int> tok(seq_len, MASK);
        vector<int> seed(prompt_len);
        for (int s = 0; s < prompt_len; ++s) { seed[s] = val_ids[p + s]; tok[s] = seed[s]; }
        tok = decode(tok, T_steps, 0.8f, 1.0f);   // context constrains: low temp/noise
        printf("[seed]  %s\n[story] %s\n\n",
               decode_tokens(seed, vocab).c_str(),
               decode_tokens(trim_at_eot(tok, prompt_len), vocab).c_str());
    }

    return 0;
}
