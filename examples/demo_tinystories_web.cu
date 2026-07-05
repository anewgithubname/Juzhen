/**
 * @file demo_tinystories_web.cu
 * @brief Web demo for the TinyStories BPE language model: type the opening of a
 *        story and the model streams a continuation, token by token.
 *
 * Loads the trained weight cache produced by demo_tinystories.cu (no training
 * here — it errors out if the cache is missing). The one new piece relative to
 * the trainer is a C++ BPE *encoder* (bpe_encode) that turns arbitrary user
 * text into token ids using the merge rules exported in tokenizer.txt; decoding,
 * the GPU one-hot embedding, the batch=1 network and top-k sampling are all
 * shared with demo_tinystories.cu.
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
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

#ifdef CUDA
__global__ void ts_onehot_input_kernel(float* X, const float* ids,
                                       int in_dim, int V, int seq, size_t N) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float* col = X + j * in_dim;
    col[(int)ids[j]] = 1.0f;
    col[V + (int)(j % seq)] = 1.0f;
}
#endif

static string data_dir() {
    if (const char* p = getenv("TS_DATA")) return p;
    return "/mnt/external_hdd/data/nlp/tinystories";
}
static string weights_path() {
    if (const char* p = getenv("TS_WEIGHTS")) return p;
    return data_dir() + "/tinystories.weights";
}

// ── tokenizer.txt: vocab strings + merge rules ──────────────────────────────
struct Tokenizer {
    vector<string> vocab;                       // id -> token string (may contain ▁)
    unordered_map<string, int> str2id;          // token string -> id
    // merge (a,b) -> {rank, merged_id}; lower rank = higher priority
    map<pair<int, int>, pair<int, int>> merges;
    int eot_id = 0;

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
            if (vocab[i] == "<|endoftext|>") eot_id = i;
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
        // Metaspace pre-tokenizer: replace ' ' with ▁ (U+2581) and prepend one.
        string ms = "\xE2\x96\x81";             // leading ▁
        for (char ch : text) {
            if (ch == ' ' || ch == '\n' || ch == '\t') ms += "\xE2\x96\x81";
            else ms += ch;
        }
        // initial symbols = per-character vocab ids (skip chars not in vocab)
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

    // Metaspace decode of a single token: ▁ -> space.
    string decode_token(int id) const {
        if (id < 0 || id >= (int)vocab.size()) return "";
        const string& s = vocab[id];
        string out;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i + 2 < s.size() && (unsigned char)s[i] == 0xE2 &&
                (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
                out += ' '; i += 2;
            } else out += s[i];
        }
        return out;
    }
};

// ── weight cache (matches demo_tinystories.cu) ──────────────────────────────
static const int WEIGHTS_MAGIC = 0x4a5a5453;
struct ModelConfig { int seq_len, d_model, d_k, d_ff, num_heads, num_blocks; };

static bool read_cache_header(FILE* fp, const ModelConfig& cfg, int V,
                              float& val_loss, float& val_acc) {
    int h[8] = {};
    if (fread(h, sizeof(int), 8, fp) != 8) return false;
    if (h[0] != WEIGHTS_MAGIC || h[1] != V || h[2] != cfg.seq_len || h[3] != cfg.d_model ||
        h[4] != cfg.d_k || h[5] != cfg.d_ff || h[6] != cfg.num_heads || h[7] != cfg.num_blocks)
        return false;
    return fread(&val_loss, sizeof(float), 1, fp) == 1 &&
           fread(&val_acc, sizeof(float), 1, fp) == 1;
}

static void load_cached_weights(FILE* fp, int num_blocks, LinearLayer<FLOAT>& emb,
                                vector<unique_ptr<TransformerLayer<FLOAT>>>& blocks,
                                LinearLayer<FLOAT>& proj) {
    Matrix<float> M("cache", 1, 1);
    auto next = [&]() -> Matrix<float>& { read(fp, M); return M; };
    emb.W() = Matrix<FLOAT>(next()); emb.b() = Matrix<FLOAT>(next());
    for (int i = 0; i < num_blocks; ++i) {
        auto& l = *blocks[i];
        l.set_Wq(Matrix<FLOAT>(next())); l.set_Wk(Matrix<FLOAT>(next()));
        l.set_Wv(Matrix<FLOAT>(next())); l.set_Wo(Matrix<FLOAT>(next()));
        l.set_bo(Matrix<FLOAT>(next()));
        l.set_W1(Matrix<FLOAT>(next())); l.set_b1(Matrix<FLOAT>(next()));
        l.set_W2(Matrix<FLOAT>(next())); l.set_b2(Matrix<FLOAT>(next()));
        l.set_ln1_gamma(Matrix<FLOAT>(next())); l.set_ln1_beta(Matrix<FLOAT>(next()));
        l.set_ln2_gamma(Matrix<FLOAT>(next())); l.set_ln2_beta(Matrix<FLOAT>(next()));
    }
    proj.W() = Matrix<FLOAT>(next()); proj.b() = Matrix<FLOAT>(next());
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(1234);
#endif
    global_rand_gen.seed(1234);

    const string base = data_dir();
    Tokenizer tok;
    tok.load(base + "/tokenizer.txt");
    const int V = (int)tok.vocab.size();

    const ModelConfig cfg = {/*seq_len*/ 256, /*d_model*/ 512, /*d_k*/ 512,
                             /*d_ff*/ 2048, /*num_heads*/ 8, /*num_blocks*/ 8};
    const int seq_len = cfg.seq_len;
    const int in_dim = V + seq_len;

    // batch=1 inference network
    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < cfg.num_blocks; ++i)
            v.push_back(make_unique<TF>(cfg.d_model, cfg.d_k, cfg.d_ff, seq_len, b, cfg.num_heads));
        return v;
    };
    LinearLayer<FLOAT> g_embed(cfg.d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, cfg.d_model, seq_len);

    float val_loss = -1, val_acc = -1;
    FILE* fp = fopen(weights_path().c_str(), "rb");
    if (!fp || !read_cache_header(fp, cfg, V, val_loss, val_acc)) {
        cout << "No valid weight cache at " << weights_path()
             << " — train with demo_tinystories first." << endl;
        if (fp) fclose(fp);
        return 1;
    }
    load_cached_weights(fp, cfg.num_blocks, g_embed, g_blocks, g_proj);
    fclose(fp);
    cout << "Loaded model (val_loss " << val_loss << ", val_acc " << val_acc << "%)." << endl;

    list<Layer<FLOAT>*> gennn;
    gennn.push_back(&g_proj);
    for (int i = cfg.num_blocks - 1; i >= 0; --i) gennn.push_back(g_blocks[i].get());
    gennn.push_back(&g_embed);
    freeze(gennn);

    auto make_X = [&](const Matrix<float>& ids_h) {
        const size_t n = ids_h.num_col();
#ifdef CUDA
        auto ids = Matrix<CUDAfloat>(ids_h);
        auto X = Matrix<CUDAfloat>::zeros(in_dim, n);
        const int threads = 256, nblocks = (int)((n + threads - 1) / threads);
        ts_onehot_input_kernel<<<nblocks, threads>>>(
            const_cast<float*>(reinterpret_cast<const float*>(X.data())),
            reinterpret_cast<const float*>(ids.data()), in_dim, V, seq_len, n);
        CudaErrorCheck(cudaGetLastError());
        return X;
#else
        Matrix<float> X("X", in_dim, n); X.zeros();
        for (size_t j = 0; j < n; ++j) {
            X.elem((int)ids_h.elem(0, j), j) = 1.0f;
            X.elem(V + (j % seq_len), j) = 1.0f;
        }
        return Matrix<FLOAT>(X);   // host->device (no-op copy when FLOAT==float)
#endif
    };

    auto predict_mutex = make_shared<mutex>();
    uniform_real_distribution<float> u01(0.0f, 1.0f);

    // one autoregressive step: given the running token list, sample the next id
    auto sample_next = [&](const vector<int>& out, float temp, int topk) {
        Matrix<float> ids_h("gen_ids", 1, seq_len);
        for (int s = 0; s < seq_len; ++s) {
            const int pos = (int)out.size() - seq_len + s;
            ids_h.elem(0, s) = (float)(pos >= 0 ? out[pos] : tok.eot_id);
        }
        forward(gennn, make_X(ids_h));
        auto logits = as_host(g_proj.value());
        vector<pair<float, int>> ranked(V);
        for (int r = 0; r < V; ++r) ranked[r] = {logits.elem(r, seq_len - 1), r};
        const int k = min(topk, V);
        partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                     [](auto& a, auto& b) { return a.first > b.first; });
        const float mx = ranked[0].first;
        float Z = 0; vector<float> cdf(k);
        for (int r = 0; r < k; ++r) { Z += expf((ranked[r].first - mx) / temp); cdf[r] = Z; }
        const float draw = u01(global_rand_gen) * Z;
        for (int r = 0; r < k; ++r) if (draw <= cdf[r]) return ranked[r].second;
        return ranked[k - 1].second;
    };

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        ifstream f(PROJECT_DIR + string("/examples/web/tinystories.html"));
        if (!f) { res.status = 500; res.set_content("tinystories.html not found", "text/plain"); return; }
        stringstream ss; ss << f.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    svr.Get("/info", [&](const httplib::Request&, httplib::Response& res) {
        ostringstream ss;
        ss << "{\"vocab\":" << V << ",\"seq_len\":" << seq_len
           << ",\"d_model\":" << cfg.d_model << ",\"num_blocks\":" << cfg.num_blocks
           << ",\"num_heads\":" << cfg.num_heads
           << ",\"val_loss\":" << val_loss << ",\"val_acc\":" << val_acc << "}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /generate?length=&temperature=&topk=, body = story opening (text).
    // Streams the decoded continuation as it is sampled.
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        int length = 250; float temperature = 0.8f; int topk = 40;
        if (req.has_param("length")) length = atoi(req.get_param_value("length").c_str());
        if (req.has_param("temperature")) temperature = atof(req.get_param_value("temperature").c_str());
        if (req.has_param("topk")) topk = atoi(req.get_param_value("topk").c_str());
        length = max(1, min(length, 1000));
        temperature = max(0.05f, min(temperature, 5.0f));
        topk = max(1, min(topk, V));

        auto ctx = make_shared<vector<int>>(tok.encode(req.body));
        if (ctx->empty()) ctx->push_back(tok.eot_id);

        res.set_chunked_content_provider(
            "text/plain; charset=utf-8",
            [&, ctx, length, temperature, topk](size_t, httplib::DataSink& sink) {
                lock_guard<mutex> lock(*predict_mutex);
                for (int i = 0; i < length; ++i) {
                    const int next = sample_next(*ctx, temperature, topk);
                    if (next == tok.eot_id) break;   // story finished
                    ctx->push_back(next);
                    const string piece = tok.decode_token(next);
                    if (!piece.empty() && !sink.write(piece.c_str(), piece.size())) break;
                }
                sink.done();
                return true;
            });
    });

    int port = 8097;
    if (const char* p = getenv("PORT")) port = atoi(p);
    cout << "Serving TinyStories web demo on http://localhost:" << port << endl;
    if (!svr.listen("0.0.0.0", port)) { cout << "listen failed on port " << port << endl; return 1; }
    return 0;
}
