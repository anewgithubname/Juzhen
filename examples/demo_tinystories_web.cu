/**
 * @file demo_tinystories_web.cu
 * @brief Web demo for the TinyStories BPE language models: type the opening of
 *        a story and a model writes the rest. Two selectable backends:
 *
 *   - autoregressive (demo_tinystories.cu weights): streams the continuation
 *     token by token over POST /generate
 *   - masked discrete diffusion (demo_diffusion_tinystories.cu weights):
 *     MaskGIT-style progressive decoding over POST /diffuse, streaming the
 *     whole window once per denoising round (NDJSON) so the front end can
 *     animate the unmasking; loaded only if its weight cache exists
 *
 * Loads trained weight caches only (no training here — it errors out if the
 * AR cache is missing; the diffusion backend is optional). The one new piece
 * relative to the trainers is a C++ BPE *encoder* (bpe_encode) that turns
 * arbitrary user text into token ids using the merge rules exported in
 * tokenizer.txt; decoding, the GPU one-hot embedding, the batch=1 networks
 * and the samplers are all shared with the trainer demos.
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
#include <functional>
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
static string weights_path_dd() {
    if (const char* p = getenv("TSDIFF_WEIGHTS")) return p;
    return data_dir() + "/tinystories_diffusion.weights";
}

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

// ── weight caches (match demo_tinystories.cu / demo_diffusion_tinystories.cu) ─
static const int WEIGHTS_MAGIC = 0x4a5a5453;      // "JZTS" — autoregressive
static const int DD_WEIGHTS_MAGIC = 0x4a5a4444;   // "JZDD" — masked diffusion
struct ModelConfig { int seq_len, d_model, d_k, d_ff, num_heads, num_blocks; };

static bool read_cache_header(FILE* fp, const ModelConfig& cfg, int V, int magic,
                              float& val_loss, float& val_acc) {
    int h[8] = {};
    if (fread(h, sizeof(int), 8, fp) != 8) return false;
    if (h[0] != magic || h[1] != V || h[2] != cfg.seq_len || h[3] != cfg.d_model ||
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

    // token one-hot slots: AR reads V tokens; diffusion reads V+1 (slot V = [MASK])
    const int MASK = V;
    const int V1 = V + 1;
    const int in_dim_dd = V1 + seq_len;

    // batch=1 inference networks
    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b, bool causal) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < cfg.num_blocks; ++i)
            v.push_back(make_unique<TF>(cfg.d_model, cfg.d_k, cfg.d_ff, seq_len, b,
                                        cfg.num_heads, causal));
        return v;
    };
    LinearLayer<FLOAT> g_embed(cfg.d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1, /*causal=*/true);
    LinearLayer<FLOAT> g_proj(V, cfg.d_model, seq_len);

    float val_loss = -1, val_acc = -1;
    FILE* fp = fopen(weights_path().c_str(), "rb");
    if (!fp || !read_cache_header(fp, cfg, V, WEIGHTS_MAGIC, val_loss, val_acc)) {
        cout << "No valid weight cache at " << weights_path()
             << " — train with demo_tinystories first." << endl;
        if (fp) fclose(fp);
        return 1;
    }
    load_cached_weights(fp, cfg.num_blocks, g_embed, g_blocks, g_proj);
    fclose(fp);
    cout << "Loaded AR model (val_loss " << val_loss << ", val_acc " << val_acc << "%)." << endl;

    list<Layer<FLOAT>*> gennn;
    gennn.push_back(&g_proj);
    for (int i = cfg.num_blocks - 1; i >= 0; --i) gennn.push_back(g_blocks[i].get());
    gennn.push_back(&g_embed);
    freeze(gennn);

    // optional masked-diffusion backend (bidirectional attention, [MASK] slot)
    LinearLayer<FLOAT> dd_embed(cfg.d_model, in_dim_dd, seq_len);
    auto dd_blocks = make_blocks(1, /*causal=*/false);
    LinearLayer<FLOAT> dd_proj(V, cfg.d_model, seq_len);
    float dd_val_loss = -1, dd_val_acc = -1;
    bool have_dd = false;
    if (FILE* dfp = fopen(weights_path_dd().c_str(), "rb")) {
        if (read_cache_header(dfp, cfg, V, DD_WEIGHTS_MAGIC, dd_val_loss, dd_val_acc)) {
            load_cached_weights(dfp, cfg.num_blocks, dd_embed, dd_blocks, dd_proj);
            have_dd = true;
            cout << "Loaded diffusion model (val_loss " << dd_val_loss
                 << ", masked_acc " << dd_val_acc << "%)." << endl;
        } else {
            cout << "Diffusion weight cache at " << weights_path_dd()
                 << " is stale or invalid; serving AR only." << endl;
        }
        fclose(dfp);
    } else {
        cout << "No diffusion weight cache at " << weights_path_dd()
             << " — train with demo_diffusion_tinystories to enable that backend." << endl;
    }
    list<Layer<FLOAT>*> ddnn;
    ddnn.push_back(&dd_proj);
    for (int i = cfg.num_blocks - 1; i >= 0; --i) ddnn.push_back(dd_blocks[i].get());
    ddnn.push_back(&dd_embed);
    freeze(ddnn);

    // idim/vslot parameterize the two input encodings (AR: V, diffusion: V+1)
    auto make_X_dim = [&](const Matrix<float>& ids_h, int idim, int vslot) {
        const size_t n = ids_h.num_col();
#ifdef CUDA
        auto ids = Matrix<CUDAfloat>(ids_h);
        auto X = Matrix<CUDAfloat>::zeros(idim, n);
        const int threads = 256, nblocks = (int)((n + threads - 1) / threads);
        ts_onehot_input_kernel<<<nblocks, threads>>>(
            const_cast<float*>(reinterpret_cast<const float*>(X.data())),
            reinterpret_cast<const float*>(ids.data()), idim, vslot, seq_len, n);
        CudaErrorCheck(cudaGetLastError());
        return X;
#else
        Matrix<float> X("X", idim, n); X.zeros();
        for (size_t j = 0; j < n; ++j) {
            X.elem((int)ids_h.elem(0, j), j) = 1.0f;
            X.elem(vslot + (j % seq_len), j) = 1.0f;
        }
        return Matrix<FLOAT>(X);   // host->device (no-op copy when FLOAT==float)
#endif
    };
    auto make_X = [&](const Matrix<float>& ids_h) { return make_X_dim(ids_h, in_dim, V); };

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

    // ── diffusion decoding (see demo_diffusion_tinystories.cu) ──────────────
    const double PI = 3.14159265358979323846;

    // one denoiser pass over a full window (tok[s] in [0,V], V = [MASK])
    auto dd_denoise = [&](const vector<int>& t) {
        Matrix<float> ids_h("dd_ids", 1, seq_len);
        for (int s = 0; s < seq_len; ++s) ids_h.elem(0, s) = (float)t[s];
        forward(ddnn, make_X_dim(ids_h, in_dim_dd, V1));
        return as_host(dd_proj.value());
    };

    auto dd_sample_col = [&](const Matrix<float>& logits, int col, float temp,
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

    auto dd_gumbel = [&]() {
        float u = u01(global_rand_gen);
        return -logf(-logf(u > 1e-9f ? u : 1e-9f));
    };

    // render the window from `from` on: [MASK] -> □, <|endoftext|> -> ¶
    auto dd_render = [&](const vector<int>& t, int from) {
        string out;
        for (int s = from; s < seq_len; ++s) {
            if (t[s] == MASK) out += "\xE2\x96\xA1";            // □
            else if (t[s] == tok.eot_id) out += "\xC2\xB6";     // ¶
            else out += tok.decode_token(t[s]);
        }
        return out;
    };
    // final text: stop at the first <|endoftext|> after the prompt
    auto dd_final = [&](const vector<int>& t, int from) {
        string out;
        for (int s = from; s < seq_len; ++s) {
            if (t[s] == tok.eot_id) break;
            if (t[s] != MASK) out += tok.decode_token(t[s]);
        }
        return out;
    };

    // MaskGIT-style progressive decoding, reporting the rendered window after
    // every round via on_round (returning false aborts: client disconnected).
    auto dd_decode = [&](vector<int> t, int T, float temp, float noise_scale, int from,
                         const function<bool(int, const string&)>& on_round) {
        int n_masked = 0;
        for (int s = 0; s < seq_len; ++s) if (t[s] == MASK) n_masked++;
        const int initial = n_masked;
        for (int k = 1; k <= T && n_masked > 0; ++k) {
            auto logits = dd_denoise(t);
            vector<int>   cand(seq_len, 0);
            vector<float> score(seq_len, -1e30f);
            const float anneal = noise_scale * (float)(T - k) / (float)T;
            for (int s = 0; s < seq_len; ++s) {
                if (t[s] != MASK) continue;
                float conf; dd_sample_col(logits, s, temp, cand[s], conf);
                score[s] = logf(conf > 1e-9f ? conf : 1e-9f) + anneal * dd_gumbel();
            }
            int target = (k >= T) ? 0
                : (int)floor(initial * cos(0.5 * PI * (double)k / (double)T));
            target = min(target, n_masked);
            const int reveal = n_masked - target;
            vector<int> order;
            for (int s = 0; s < seq_len; ++s) if (t[s] == MASK) order.push_back(s);
            partial_sort(order.begin(), order.begin() + reveal, order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
            for (int j = 0; j < reveal; ++j) { t[order[j]] = cand[order[j]]; n_masked--; }
            if (!on_round(k, dd_render(t, from))) break;
        }
        return t;
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
           << ",\"val_loss\":" << val_loss << ",\"val_acc\":" << val_acc
           << ",\"diffusion\":" << (have_dd ? "true" : "false")
           << ",\"dd_val_loss\":" << dd_val_loss << ",\"dd_val_acc\":" << dd_val_acc << "}";
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

    // POST /diffuse?steps=&temperature=&noise=, body = story opening (text).
    // The prompt is pinned as context; the rest of the 256-token window starts
    // as [MASK] and is progressively decoded. Streams NDJSON: one line per
    // denoising round ({"round":k,"total":T,"text":"..."} with □ for still-
    // masked slots), then {"done":true,"text":"..."} trimmed at end-of-story.
    svr.Post("/diffuse", [&](const httplib::Request& req, httplib::Response& res) {
        if (!have_dd) {
            res.status = 503;
            res.set_content("diffusion backend not available — train demo_diffusion_tinystories first",
                            "text/plain");
            return;
        }
        int T = 64; float temperature = 0.8f, noise = 1.0f;
        if (req.has_param("steps")) T = atoi(req.get_param_value("steps").c_str());
        if (req.has_param("temperature")) temperature = atof(req.get_param_value("temperature").c_str());
        if (req.has_param("noise")) noise = atof(req.get_param_value("noise").c_str());
        T = max(1, min(T, 256));
        temperature = max(0.05f, min(temperature, 5.0f));
        noise = max(0.0f, min(noise, 5.0f));

        auto ctx = make_shared<vector<int>>(seq_len, MASK);
        vector<int> ids = tok.encode(req.body);
        const int plen = min((int)ids.size(), seq_len / 2);
        for (int s = 0; s < plen; ++s) (*ctx)[s] = ids[s];

        res.set_chunked_content_provider(
            "application/x-ndjson",
            [&, ctx, T, temperature, noise, plen](size_t, httplib::DataSink& sink) {
                lock_guard<mutex> lock(*predict_mutex);
                auto emit = [&](const string& line) {
                    return sink.write(line.c_str(), line.size());
                };
                emit("{\"round\":0,\"total\":" + to_string(T) + ",\"text\":\""
                     + json_escape(dd_render(*ctx, plen)) + "\"}\n");
                auto t = dd_decode(*ctx, T, temperature, noise, plen,
                    [&](int k, const string& text) {
                        return emit("{\"round\":" + to_string(k) + ",\"total\":" + to_string(T)
                                    + ",\"text\":\"" + json_escape(text) + "\"}\n");
                    });
                emit("{\"done\":true,\"text\":\"" + json_escape(dd_final(t, plen)) + "\"}\n");
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
