/**
 * @file demo_word2vec.cu
 * @brief word2vec (skip-gram with negative sampling) on text8, expressed
 *        entirely in Juzhen matrix primitives — no ml/layer.hpp, gradients
 *        derived by hand. Design notes: misc/word2vec_demo_plan.md.
 *
 * Objective (batch of B (center, context) pairs sharing K negatives):
 *
 *     V_c = W_in  X_c   (d x B)     U_o = W_out X_o   (d x B)
 *     U_n = W_out X_n   (d x K)
 *     s+  = colsum(V_c . U_o)  (1 x B)     S- = U_n^T V_c  (K x B)
 *     L   = -sum_b log sigma(s+_b) - sum_kb log sigma(-S-_kb)
 *
 * X_c/X_o/X_n are one-hot (GEMM replaces embedding lookup, as in
 * demo_tinystories). Backward, with g+ = sigma(s+)-1 and G- = sigma(S-):
 *
 *     dV_c = U_o diag(g+) + U_n G-      dU_o = V_c diag(g+)
 *     dU_n = V_c G-^T
 *     W_in  -= lr/B * dV_c X_c^T
 *     W_out -= lr/B * (dU_o X_o^T + dU_n X_n^T)
 *
 * Data: datasets/text8 (17M words; already in this repo). For noticeably
 * better vectors train on fil9 (~124M words, same format; built from enwik9
 * with Matt Mahoney's wikifil.pl, http://mattmahoney.net/dc/textdata.html):
 *     curl -L https://mattmahoney.net/dc/enwik9.zip -o enwik9.zip  # 322MB
 *     unzip enwik9.zip && perl wikifil.pl enwik9 > fil9
 *     W2V_DATA=path/to/fil9 VOCAB=50000 STEPS=600000 ./demo_word2vec
 * (measured: ~30 min on an RTX 4090; same 4/5 analogy score but visibly
 *  cleaner nearest neighbours than text8)
 *
 * After training (or when a weight cache exists) it runs a canned analogy
 * check (king - man + woman ~ queen, ...) and, on a terminal, an interactive
 * loop: type `word` for nearest neighbours or `a - b + c` for analogies.
 *
 * Env overrides: STEPS BATCH DIM VOCAB NEG WINDOW LR
 *                LOSS=nce|infonce TAU (softmax temperature, infonce only)
 *                W2V_DATA (corpus path)  W2V_WEIGHTS (cache path)
 *                W2V_INTERACTIVE=0|1 (default: only when stdin is a tty)
 * Exit code 77 when the corpus file is missing (ctest skip convention).
 */

#include "../cpp/juzhen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define STDIN_IS_TTY() _isatty(_fileno(stdin))
#else
#include <unistd.h>
#define STDIN_IS_TTY() isatty(fileno(stdin))
#endif

using namespace std;

#ifdef CUDA
#define FLOAT CUDAfloat
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
#else
#define FLOAT float
#endif
typedef Matrix<FLOAT> MF;

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

static float scalar(const MF& m) { return as_host(m).elem(0, 0); }

static int env_int(const char* name, int def) {
    const char* s = getenv(name);
    return s ? atoi(s) : def;
}
static float env_float(const char* name, float def) {
    const char* s = getenv(name);
    return s ? (float)atof(s) : def;
}

// ── GPU one-hot builder ──────────────────────────────────────────────────────
// ids arrive as a (1, n) float matrix (ids < 2^24 are exact in fp32).
#ifdef CUDA
__global__ void w2v_onehot_kernel(float* X, const float* ids, int V, size_t n) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    X[j * (size_t)V + (size_t)ids[j]] = 1.0f;
}
#endif

// (1, n) host ids -> (V, n) one-hot on the compute device
static MF make_onehot(const Matrix<float>& ids_h, int V) {
    const size_t n = ids_h.num_col();
#ifdef CUDA
    auto ids = Matrix<CUDAfloat>(ids_h);
    auto X = Matrix<CUDAfloat>::zeros(V, n);
    const int threads = 256;
    const int nblocks = (int)((n + threads - 1) / threads);
    w2v_onehot_kernel<<<nblocks, threads>>>(
        const_cast<float*>(reinterpret_cast<const float*>(X.data())),
        reinterpret_cast<const float*>(ids.data()), V, n);
    CudaErrorCheck(cudaGetLastError());
    return X;
#else
    Matrix<float> X("onehot", V, n);
    X.zeros();
    for (size_t j = 0; j < n; ++j) X.elem((size_t)ids_h.elem(0, j), j) = 1.0f;
    return MF(X);  // host->device (plain copy on the CPU backend)
#endif
}

// ── vocabulary / corpus ──────────────────────────────────────────────────────
struct Corpus {
    vector<string> words;         // id -> word; last id is <unk>
    unordered_map<string, int> word2id;
    vector<int64_t> counts;       // id -> corpus count
    vector<int32_t> train_ids;    // encoded + subsampled token stream
    int64_t total_tokens = 0;
};

static string data_path() {
    if (const char* p = getenv("W2V_DATA")) return p;
    return PROJECT_DIR + string("/datasets/text8");
}
static string weights_path() {
    if (const char* p = getenv("W2V_WEIGHTS")) return p;
    return PROJECT_DIR + string("/datasets/word2vec.weights");
}

// Two streaming passes over the raw corpus: (1) count words and pick the top
// V-1 (everything else maps to <unk>), (2) encode with frequency subsampling
// (Mikolov 2013: keep w with prob (sqrt(f/t)+1)*(t/f), f = corpus frequency).
static bool load_corpus(Corpus& C, int V, mt19937_64& rng) {
    ifstream f(data_path(), ios::binary | ios::ate);
    if (!f) return false;
    const streamsize len = f.tellg();
    f.seekg(0);
    string buf((size_t)len, '\0');
    f.read(buf.data(), len);

    unordered_map<string, int64_t> freq;
    freq.reserve(1 << 19);
    string w;
    auto each_word = [&](auto&& fn) {
        size_t i = 0;
        const size_t n = buf.size();
        while (i < n) {
            while (i < n && buf[i] == ' ') ++i;
            size_t j = i;
            while (j < n && buf[j] != ' ' && buf[j] != '\n') ++j;
            if (j > i) fn(buf.substr(i, j - i));
            i = j + 1;
        }
    };
    each_word([&](string tok) { ++freq[tok]; ++C.total_tokens; });

    vector<pair<int64_t, string>> top;
    top.reserve(freq.size());
    for (auto& [word, cnt] : freq) top.push_back({cnt, word});
    const size_t keep = min((size_t)(V - 1), top.size());
    partial_sort(top.begin(), top.begin() + keep, top.end(),
                 [](auto& a, auto& b) { return a.first > b.first; });
    top.resize(keep);

    C.words.reserve(keep + 1);
    C.counts.reserve(keep + 1);
    int64_t known = 0;
    for (auto& [cnt, word] : top) {
        C.word2id[word] = (int)C.words.size();
        C.words.push_back(word);
        C.counts.push_back(cnt);
        known += cnt;
    }
    C.word2id["<unk>"] = (int)C.words.size();
    C.words.push_back("<unk>");
    C.counts.push_back(max<int64_t>(1, C.total_tokens - known));
    const int unk = (int)C.words.size() - 1;

    const double t = 1e-4;
    uniform_real_distribution<double> uni(0.0, 1.0);
    C.train_ids.reserve((size_t)(C.total_tokens / 2));
    each_word([&](string tok) {
        auto it = C.word2id.find(tok);
        const int id = it == C.word2id.end() ? unk : it->second;
        const double fr = (double)C.counts[id] / (double)C.total_tokens;
        const double keep_p = (sqrt(fr / t) + 1.0) * (t / fr);
        if (keep_p >= 1.0 || uni(rng) < keep_p) C.train_ids.push_back(id);
    });
    return true;
}

// ── weight cache ─────────────────────────────────────────────────────────────
static const uint32_t W2V_MAGIC = 0x77327631;  // "w2v1"

static bool load_cache(Corpus& C, Matrix<float>& Win_h, int& d) {
    FILE* fp = fopen(weights_path().c_str(), "rb");
    if (!fp) return false;
    uint32_t magic = 0;
    int32_t V = 0, dim = 0;
    if (fread(&magic, 4, 1, fp) != 1 || magic != W2V_MAGIC ||
        fread(&V, 4, 1, fp) != 1 || fread(&dim, 4, 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    C.words.resize(V);
    for (int i = 0; i < V; ++i) {
        uint16_t l = 0;
        if (fread(&l, 2, 1, fp) != 1) { fclose(fp); return false; }
        C.words[i].resize(l);
        if (l && fread(C.words[i].data(), 1, l, fp) != l) { fclose(fp); return false; }
        C.word2id[C.words[i]] = i;
    }
    d = dim;
    Win_h = Matrix<float>("Win", d, V);
    const size_t n = (size_t)d * V;
    const bool ok = fread((float*)Win_h.data(), sizeof(float), n, fp) == n;
    fclose(fp);
    return ok;
}

static void save_cache(const Corpus& C, const Matrix<float>& Win_h) {
    FILE* fp = fopen(weights_path().c_str(), "wb");
    if (!fp) {
        cout << "cannot write " << weights_path() << endl;
        return;
    }
    const int32_t V = (int32_t)C.words.size(), d = (int32_t)Win_h.num_row();
    fwrite(&W2V_MAGIC, 4, 1, fp);
    fwrite(&V, 4, 1, fp);
    fwrite(&d, 4, 1, fp);
    for (auto& w : C.words) {
        const uint16_t l = (uint16_t)w.size();
        fwrite(&l, 2, 1, fp);
        fwrite(w.data(), 1, l, fp);
    }
    fwrite((const float*)Win_h.data(), sizeof(float), (size_t)d * V, fp);
    fclose(fp);
    cout << "weights cached to " << weights_path() << endl;
}

// ── evaluation (host side) ───────────────────────────────────────────────────
// Wn: (d, V) with L2-normalized columns. Returns ids of the k best cosine
// matches to q (also normalized), skipping ids in `exclude` and <unk>.
static vector<int> nearest(const vector<float>& q, const Matrix<float>& Wn,
                           const vector<int>& exclude, int k) {
    const int d = (int)Wn.num_row(), V = (int)Wn.num_col();
    vector<pair<float, int>> best;
    for (int i = 0; i < V - 1; ++i) {  // V-1: never suggest <unk>
        if (find(exclude.begin(), exclude.end(), i) != exclude.end()) continue;
        float s = 0;
        for (int r = 0; r < d; ++r) s += q[r] * Wn.elem(r, i);
        best.push_back({s, i});
    }
    const int kk = min<int>(k, (int)best.size());
    partial_sort(best.begin(), best.begin() + kk, best.end(),
                 [](auto& a, auto& b) { return a.first > b.first; });
    vector<int> ids(kk);
    for (int i = 0; i < kk; ++i) ids[i] = best[i].second;
    return ids;
}

static vector<float> column(const Matrix<float>& W, int i) {
    vector<float> v(W.num_row());
    for (size_t r = 0; r < W.num_row(); ++r) v[r] = W.elem(r, i);
    return v;
}

static void normalize(vector<float>& v) {
    float n = 0;
    for (float x : v) n += x * x;
    n = sqrtf(n) + 1e-12f;
    for (float& x : v) x /= n;
}

// v(a) - v(b) + v(c), normalized
static vector<float> analogy_query(const Matrix<float>& Wn, int a, int b, int c) {
    vector<float> q = column(Wn, a);
    const auto vb = column(Wn, b), vc = column(Wn, c);
    for (size_t r = 0; r < q.size(); ++r) q[r] += vc[r] - vb[r];
    normalize(q);
    return q;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(42);
#endif
    cout << "word2vec: skip-gram with negative sampling on "
         << data_path() << endl;

    // Defaults follow Mikolov's text8 reference setup (dim 200, window 8,
    // sample 1e-4); measured on text8: 4/5 canned analogies in top-3 with
    // king-man+woman = queen at rank 1 (~9 min on an RTX 4090).
    const int d = env_int("DIM", 200);
    const int V = env_int("VOCAB", 30000);
    const int B = env_int("BATCH", 1024);
    const int K = env_int("NEG", 64);
    const int window = env_int("WINDOW", 8);
    const int steps = env_int("STEPS", 300000);
    const float lr0 = env_float("LR", 10.0f);
    // LOSS=nce (default): SGNS binary logistic loss (word2vec 2013).
    // LOSS=infonce: softmax over [s+; S-] (CPC/SimCLR family), temperature TAU.
    const char* loss_env = getenv("LOSS");
    const bool infonce = loss_env && string(loss_env) == "infonce";
    const float tau = env_float("TAU", 1.0f);
    const int log_every = max(1, steps / 40);

    mt19937_64 rng(42);
    Corpus C;
    Matrix<float> Win_h("Win", 1, 1);
    int d_cached = 0;

    if (load_cache(C, Win_h, d_cached)) {
        cout << "loaded cached embeddings: " << C.words.size() << " words, dim "
             << d_cached << " (" << weights_path() << ")" << endl;
    } else {
        if (!load_corpus(C, V, rng)) {
            cout << "cannot open " << data_path() << endl
                 << "text8: curl -L https://mattmahoney.net/dc/text8.zip -o "
                    "/tmp/text8.zip && unzip /tmp/text8.zip -d datasets/"
                 << endl;
            return 77;  // ctest skip convention
        }
        const int Vr = (int)C.words.size();  // actual vocab (<= V, incl <unk>)
        cout << "corpus: " << C.total_tokens << " tokens, "
             << C.train_ids.size() << " after subsampling, vocab " << Vr
             << endl;
        cout << "model: dim " << d << ", batch " << B << ", " << K
             << " shared negatives, window " << window << ", steps " << steps
             << ", lr " << lr0 << " (mean-gradient SGD, linear decay), loss "
             << (infonce ? "infonce (tau " + to_string(tau) + ")" : "nce")
             << endl
             << endl;

        // unigram^{3/4} cumulative table for negative sampling
        vector<double> cum(Vr);
        double acc = 0;
        for (int i = 0; i < Vr; ++i) {
            acc += pow((double)C.counts[i], 0.75);
            cum[i] = acc;
        }
        uniform_real_distribution<double> uni(0.0, acc);
        uniform_int_distribution<size_t> pos(window,
                                             C.train_ids.size() - window - 1);
        uniform_int_distribution<int> rad(1, window);

        // W_in ~ U(-0.5, 0.5)/d (as in the original word2vec); W_out = 0.
        auto Win = (MF::rand(d, Vr) - 0.5) / (double)d;
        auto Wout = MF::zeros(d, Vr);
        const auto broadcast_rows = MF::ones(d, 1);
        const auto broadcast_negs = MF::ones(K, 1);

        Matrix<float> idc("idc", 1, B), ido("ido", 1, B), idn("idn", 1, K);
        float first_loss = -1, last_loss = -1;

        for (int step = 0; step < steps; ++step) {
            for (int b = 0; b < B; ++b) {
                const size_t i = pos(rng);
                int delta = rad(rng);
                if (rng() & 1) delta = -delta;
                idc.elem(0, b) = (float)C.train_ids[i];
                ido.elem(0, b) = (float)C.train_ids[i + delta];
            }
            for (int k = 0; k < K; ++k) {
                const int id = (int)(upper_bound(cum.begin(), cum.end(),
                                                 uni(rng)) -
                                     cum.begin());
                idn.elem(0, k) = (float)min(id, Vr - 1);
            }

            auto Xc = make_onehot(idc, Vr);
            auto Xo = make_onehot(ido, Vr);
            auto Xn = make_onehot(idn, Vr);

            // forward
            auto Vc = Win * Xc;                       // (d, B)
            auto Uo = Wout * Xo;                      // (d, B)
            auto Un = Wout * Xn;                      // (d, K)
            auto spos = sum(hadmd(Vc, Uo), 0);        // (1, B)
            auto Sneg = Un.T() * Vc;                  // (K, B)

            // The two contrastive losses differ only in the error signals
            //   gpos = dL/ds+  (1, B)      Gneg = dL/dS-  (K, B);
            // the chain rule through the two linear maps below is shared.
            const bool want_loss = step % log_every == 0 || step == steps - 1;
            float loss = 0;
            MF gpos, Gneg;
            if (infonce) {
                // softmax over the K+1 candidates, shifted by s+ for
                // stability: p+ = 1/Z, P- = E/Z with E = exp((S- - s+)/tau),
                // Z = 1 + colsum(E); loss = -log p+ = log Z.
                auto E = exp((Sneg - broadcast_negs * spos) / (double)tau);
                auto Z = 1.0 + sum(E, 0);                       // (1, B)
                if (want_loss) loss = scalar(sum(log(Z), 1)) / B;
                gpos = (1.0 / Z - 1.0) / (double)tau;
                Gneg = (std::move(E) / (broadcast_negs * Z)) / (double)tau;
            } else {
                auto sigp = 1.0 / (exp(-spos) + 1.0);           // sigma(s+)
                auto sigg = 1.0 / (exp(-Sneg) + 1.0);           // sigma(S-)
                if (want_loss)
                    loss = (-scalar(sum(log(sigp + 1e-12f), 1)) -
                            scalar(sum(sum(log(1.0 - sigg + 1e-12f), 0), 1))) /
                           B;
                gpos = sigp - 1.0;
                Gneg = std::move(sigg);
            }

            if (want_loss) {
                if (first_loss < 0) first_loss = loss;
                last_loss = loss;
                printf("step %6d   loss/pair = %8.4f   lr = %.4f\n", step,
                       loss, lr0 * (1.0f - (float)step / steps));
            }

            // backward (mean over the batch)
            const float lr = max(lr0 * (1.0f - (float)step / steps),
                                 lr0 * 0.01f);
            auto gb = broadcast_rows * gpos;          // (d, B) row broadcast
            auto dVc = hadmd(Uo, gb) + Un * Gneg;     // (d, B)
            auto dUo = hadmd(Vc, std::move(gb));      // (d, B)
            auto dUn = Vc * Gneg.T();                 // (d, K)

            Win -= (dVc * Xc.T()) * (double)(lr / B);
            Wout -= (dUo * Xo.T() + dUn * Xn.T()) * (double)(lr / B);
        }

        Win_h = as_host(Win);
        save_cache(C, Win_h);

        // smoke check for ctest: training must have moved the loss.
        // InfoNCE gets an absolute margin instead of a ratio — its loss is
        // bounded below by log(K+1) - I(center;context), so large relative
        // drops are impossible by construction.
        const bool ok = infonce ? last_loss < first_loss - 0.05f
                                : last_loss < 0.7f * first_loss;
        if (!ok) {
            cout << "loss did not decrease enough (" << first_loss << " -> "
                 << last_loss << ")" << endl;
            return 1;
        }
    }

    // ── evaluation ──────────────────────────────────────────────────────────
    // L2-normalize columns once; all queries are cosine similarities.
    Matrix<float> Wn = Win_h;
    for (size_t i = 0; i < Wn.num_col(); ++i) {
        float n = 0;
        for (size_t r = 0; r < Wn.num_row(); ++r)
            n += Wn.elem(r, i) * Wn.elem(r, i);
        n = sqrtf(n) + 1e-12f;
        for (size_t r = 0; r < Wn.num_row(); ++r) Wn.elem(r, i) /= n;
    }
    auto id_of = [&](const string& w) -> int {
        auto it = C.word2id.find(w);
        return it == C.word2id.end() ? -1 : it->second;
    };
    auto print_neighbors = [&](const string& w) {
        const int i = id_of(w);
        if (i < 0) { cout << "  '" << w << "' not in vocabulary" << endl; return; }
        auto q = column(Wn, i);
        cout << "  " << w << " ->";
        for (int j : nearest(q, Wn, {i}, 8)) cout << " " << C.words[j];
        cout << endl;
    };

    cout << endl << "nearest neighbours:" << endl;
    for (const char* w : {"cat", "king", "computer", "france", "three"})
        print_neighbors(w);

    // analogy check: a - b + c ~ answer
    struct Q { const char *a, *b, *c, *answer; };
    const vector<Q> quiz = {
        {"king", "man", "woman", "queen"},
        {"paris", "france", "italy", "rome"},
        {"bigger", "big", "small", "smaller"},
        {"brother", "he", "she", "sister"},
        {"london", "england", "germany", "berlin"},
    };
    cout << endl << "analogies (a - b + c = ?):" << endl;
    int hits = 0, asked = 0;
    for (auto& t : quiz) {
        const int a = id_of(t.a), b = id_of(t.b), c = id_of(t.c),
                  ans = id_of(t.answer);
        if (a < 0 || b < 0 || c < 0 || ans < 0) continue;
        ++asked;
        auto ids = nearest(analogy_query(Wn, a, b, c), Wn, {a, b, c}, 3);
        const bool hit = find(ids.begin(), ids.end(), ans) != ids.end();
        hits += hit;
        cout << "  " << t.a << " - " << t.b << " + " << t.c << " = ";
        for (int j : ids) cout << C.words[j] << " ";
        cout << (hit ? "[ok]" : "[expected: " + string(t.answer) + "]") << endl;
    }
    cout << "analogy top-3 hits: " << hits << "/" << asked << endl;

    // ── interactive loop ────────────────────────────────────────────────────
    const int inter = env_int("W2V_INTERACTIVE", STDIN_IS_TTY() ? 1 : 0);
    if (inter) {
        cout << endl
             << "interactive: `word` for neighbours, `a - b + c` for analogies,"
             << " empty line quits" << endl;
        string line;
        while (cout << "> " << flush, getline(cin, line)) {
            istringstream ss(line);
            vector<string> tok;
            for (string t; ss >> t;) tok.push_back(t);
            if (tok.empty()) break;
            if (tok.size() == 1) {
                print_neighbors(tok[0]);
            } else if (tok.size() == 5 && tok[1] == "-" && tok[3] == "+") {
                const int a = id_of(tok[0]), b = id_of(tok[2]), c = id_of(tok[4]);
                if (a < 0 || b < 0 || c < 0) { cout << "  word not in vocabulary" << endl; continue; }
                cout << "  =";
                for (int j : nearest(analogy_query(Wn, a, b, c), Wn, {a, b, c}, 5))
                    cout << " " << C.words[j];
                cout << endl;
            } else {
                cout << "  usage: `word`  or  `a - b + c`" << endl;
            }
        }
    }
    return 0;
}
