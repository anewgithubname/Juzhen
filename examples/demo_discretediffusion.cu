/**
 * @file demo_discretediffusion.cu
 * @brief Masked (absorbing-state) discrete diffusion language model on enwik8.
 *
 * This is the discrete-diffusion counterpart to demo_transformer.cu. Instead of
 * predicting the next character autoregressively, it learns to *denoise*: a
 * clean character window is corrupted by replacing a random fraction of tokens
 * with a special [MASK] symbol, and a bidirectional transformer is trained to
 * recover the originals. Generation then starts from an all-[MASK] sequence and
 * progressively unmasks it. This is the "absorbing state" D3PM / MDLM recipe
 * (Austin et al. 2021; Sahoo et al. 2024) — essentially BERT-style masking made
 * into a proper generative model via a noise schedule and iterative decoding.
 *
 *   - corpus: prefers examples/text8 (the standard 27-symbol char-LM benchmark
 *     the discrete-diffusion literature reports on), else examples/enwik8 (205
 *     byte-level symbols), else corpus.txt, else an embedded passage (Alice)
 *   - vocabulary = the corpus charset PLUS one extra [MASK] token
 *   - forward (noising) process: sample a mask level t~U(eps,1), then replace
 *     each token independently with [MASK] with probability t
 *   - denoiser: embed -> N x bidirectional TransformerLayer (causal=false) ->
 *     projection to the real vocabulary
 *   - loss: a CUSTOM masked cross-entropy. Only masked positions are supervised,
 *     and each is weighted by 1/t — the continuous-time MDLM ELBO weight for a
 *     linear schedule. Averaged over tokens this is an upper bound on the data
 *     NLL, so we also report it as bits-per-character (BPC).
 *   - 90/10 train/val split; held-out ELBO + masked-token accuracy are logged
 *   - generation: MaskGIT-style progressive decoding from all-[MASK], plus a
 *     text-infilling example that fills a masked span inside real context.
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

// ────────────────────────────────────────────────────────────────────────────
// Custom loss: masked, 1/t-weighted softmax cross-entropy.
//
// This mirrors the framework's LogisticLayer (stable softmax cross-entropy) but
// attaches a per-column weight w (shape 1xN). Columns are (position, sequence)
// pairs. For an unmasked token w=0, so it contributes neither loss nor gradient
// — the denoiser still *reads* it (bidirectional attention) but is not
// supervised there. For a masked token in a sequence noised at level t, we set
// w = (1/t)/N, which is the MDLM continuous-time ELBO weight (linear schedule)
// divided by the token count N so the reported value is a per-token bound.
//   loss   = sum_c  w_c * ( logsumexp(z_c) - <z_c, y_c> )
//   dloss/dz_c = w_c * ( softmax(z_c) - y_c )
// ────────────────────────────────────────────────────────────────────────────
// Column-wise max of a (V, N) matrix -> (1, N); used for a stable softmax.
// Free function: an extended __host__ __device__ lambda may not sit in a
// private/protected class member.
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
        auto mx = col_max(input);              // (1, N)
        auto E = exp(input - oneK1 * mx);      // (V, N)
        auto Z = oneK1 * sum(E, 0);            // (V, N) column sums broadcast
        auto softmax = E / std::move(Z);       // (V, N)
        // Per-column weight applied to (softmax - onehot).
        return hadmd(softmax - output, oneK1 * wrow);
    }

    void eval(const Matrix<D>& input) override {
        auto mx = col_max(input);                          // (1, N)
        auto lse = log(sum(exp(input - oneK1 * mx), 0)) + mx;   // (1, N)
        auto dot = sum(hadmd(input, output), 0);           // (1, N) <z, y>
        auto ce = lse - dot;                               // (1, N) per-token CE
        Layer<D>::val = sum(hadmd(wrow, ce), 1);           // (1, 1) weighted sum
    }
};

// ── enwik8 corpus loading (identical policy to demo_transformer.cu) ──────────
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
    const string base = string(PROJECT_DIR) + "/examples/";
    // Prefer text8 (the standard 27-symbol char-LM benchmark that the discrete-
    // diffusion literature reports, e.g. D3PM-absorbing ~1.45 BPC) when present,
    // so our BPC is directly comparable. Fetch it once into examples/:
    //     curl -L https://mattmahoney.net/dc/text8.zip -o examples/text8.zip
    //     unzip -o examples/text8.zip -d examples/       # -> examples/text8
    // Falls back to raw enwik8 (205 byte-level symbols, a harder alphabet).
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

// Render a byte for terminal-safe printing.
static char printable(unsigned char u) {
    if (u == '\n') return '/';
    if (u < 32 || u >= 127) return '.';
    return (char)u;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(42);   // initialize the curand generator used by randn()
#endif
    global_rand_gen.seed(42);
    uniform_real_distribution<float> u01(0.0f, 1.0f);

    string corpus_source;
    const string corpus = load_corpus(corpus_source);
    const int clen = (int)corpus.size();

    set<char> charset(corpus.begin(), corpus.end());
    vector<char> idx_to_char(charset.begin(), charset.end());
    const int V = (int)idx_to_char.size();     // real vocabulary size
    int c2i[256] = {};
    for (int i = 0; i < V; ++i) c2i[(unsigned char)idx_to_char[i]] = i;

    // The denoiser reads a (V+1)-way one-hot: the extra slot V is the [MASK]
    // token. It only ever predicts the V real characters (proj outputs V rows).
    const int MASK = V;
    const int V1 = V + 1;

    // Config: the best model that actually fits this hardware/time budget
    // (~15 min on one 24 GB GPU, ~2.1 BPC on text8 / ~2.2 on enwik8). The key
    // lever here was training throughput, not raw capacity: a high token count
    // per step (batch 128 x seq 128 = 16384) reduces the stratified-t gradient
    // noise that otherwise stalls training.
    //
    // NOTE on scaling: the discrete-diffusion literature reaches ~1.45 BPC on
    // text8 with a *much larger* 12-layer/d_model=768/seq_len=256 model, but
    // only after 100k-1M training steps at large batch. Simply adopting that
    // architecture here (see git history) trains WORSE, not better: at a
    // tractable ~20k-step budget the bigger model is badly undertrained
    // (~3.3 BPC, incoherent samples). Reproducing 1.45 is a compute-budget
    // problem (many GPU-hours), not a config tweak. For a quick ~5-min smoke
    // test, scale down: d_model=d_k=256, d_ff=1024, num_blocks=4,
    // seq_len=batch=64, steps=30000. Attention is bidirectional (causal=false).
    const int seq_len = 128;
    const int d_model = 512;
    const int d_k = 512;
    const int d_ff = 2048;
    const int num_heads = 8;       // d_k must be divisible by num_heads (d_h=64)
    const int num_blocks = 8;
    const int steps = 20000;       // hard cap; early stopping ends sooner
    const int log_every = 500;     // also the early-stopping evaluation interval
    const int patience = 12;       // stop after this many evals w/o val gain
    const int in_dim = V1 + seq_len;   // one-hot(char incl. MASK) + one-hot(pos)
    const float t_min = 1e-3f;     // floor on the mask level (bounds the 1/t weight)

    // LR schedule: linear warmup then cosine decay to 10% of peak.
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

    // Train/val split: last 10% of the corpus is held out. Windows are sampled
    // directly from start-offset ranges, so this scales to enwik8.
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

    cout << "=== Masked Discrete Diffusion Char-LM Demo ===\n";
    cout << "Corpus source: " << corpus_source << "\n";
    cout << "Corpus size: " << clen << " chars (train/val split at " << split << ", 90/10)\n";
    cout << "Windows: " << train_windows << " train, " << val_windows << " val\n";
    cout << "Vocabulary (" << V << " real symbols + 1 [MASK]): ";
    for (int i = 0; i < V; ++i) cout << printable((unsigned char)idx_to_char[i]);
    cout << "\n";
    cout << "Network: embed(" << in_dim << "->" << d_model << ") -> Transformer("
         << d_model << "," << d_k << "," << d_ff << "," << num_heads
         << "h,bidir) [x" << num_blocks << "] -> proj(" << d_model << "->" << V << ")\n";
    cout << "Noising: replace each token with [MASK] at prob t~U(" << t_min
         << ",1); loss = (1/t)-weighted CE on masked tokens\n";
    cout << "Training: mini-batch=" << batch << " x seq_len=" << seq_len
         << " (" << N << " tokens/step), steps=" << steps << "\n\n";

    using TF = TransformerLayer<FLOAT>;
    auto make_blocks = [&](int b) {
        vector<unique_ptr<TF>> v;
        for (int i = 0; i < num_blocks; ++i)
            v.push_back(make_unique<TF>(d_model, d_k, d_ff, seq_len, b, num_heads,
                                        /*causal=*/false));
        return v;
    };

    LinearLayer<FLOAT> embed(d_model, in_dim, N);
    auto blocks = make_blocks(batch);
    LinearLayer<FLOAT> proj(V, d_model, N);

    // Best-model snapshot for early stopping. It only ever stores weights (via
    // copy_*_weights below) and never runs a forward/backward pass, so we build
    // it with batch=1: the weight matrices are batch-independent, but this way
    // it does NOT allocate the large per-batch activation caches a Transformer
    // layer reserves in its constructor. Those caches scale with seq_len*batch
    // (and attention ~seq_len^2*batch), so this duplicate set would otherwise
    // ~double activation memory — the difference between fitting a large config
    // on a 24 GB GPU or not. (nb=1 likewise shrinks the linear layers' unused
    // output buffers.)
    LinearLayer<FLOAT> best_embed(d_model, in_dim, 1);
    auto best_blocks = make_blocks(1);
    LinearLayer<FLOAT> best_proj(V, d_model, 1);

    // Layer list: front = last layer (proj), back = first (embed); forward()
    // and backprop() run back()->front(), so data flows embed -> blocks -> proj.
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

    auto set_all_lr = [&](float lr) {
        embed.adamWstate().alpha = lr; embed.adambstate().alpha = lr;
        proj.adamWstate().alpha  = lr; proj.adambstate().alpha  = lr;
        for (int i = 0; i < num_blocks; ++i) blocks[i]->set_lr(lr);
    };

    // Fill one training/eval batch: given window offsets and a per-sequence mask
    // level t, produce the corrupted input X (in_dim,N), the one-hot target
    // Y (V,N), and the loss weights W (1,N). true_idx/masked record the clean
    // character and whether each column was masked (for accuracy).
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
                X.elem(m ? MASK : c, col) = 1.0f;   // one-hot char (MASK if noised)
                X.elem(V1 + s, col) = 1.0f;         // one-hot position
                Y.elem(c, col) = 1.0f;              // target = clean char
                W.elem(0, col) = m ? (1.0f / t) / (float)N : 0.0f;
                true_idx[col] = c;
                masked[col] = m ? 1 : 0;
            }
        }
    };

    // Antithetic / stratified sampling of the per-sequence mask level t. The
    // 1/t ELBO weight is high-variance under i.i.d. t (rare small-t sequences
    // dominate the gradient). Instead we place one sample in each of `batch`
    // equal strata of (t_min,1] using a single shared jitter u, so every batch
    // spans the full noise range. This is the standard low-discrepancy variance
    // reduction for diffusion training and both lowers loss and stabilizes it.
    auto stratified_t = [&](vector<float>& t) {
        const float u = u01(global_rand_gen);
        for (int b = 0; b < (int)t.size(); ++b)
            t[b] = t_min + (1.0f - t_min) * ((float)b + u) / (float)t.size();
        shuffle(t.begin(), t.end(), global_rand_gen);   // decorrelate t from window
    };

    // Evaluate held-out ELBO + masked-token accuracy on a batch, averaging over
    // `draws` independent noise realizations for a lower-variance estimate.
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
        const float loss = loss_sum / draws;                 // per-token NLL bound (nats)
        const float bpc = loss / (float)log(2.0);            // bits per character
        const float acc = total ? 100.0f * (float)correct / (float)total : 0.0f;
        return make_tuple(loss, bpc, acc);
    };

    uniform_int_distribution<int> pick(train_lo, train_hi - 1);

    float best_loss = numeric_limits<float>::infinity();
    int   best_step = 0, stale = 0;

    for (int step = 0; step < steps; ++step) {
        set_all_lr(lr_at(step));

        // Sample a fresh mini-batch: random windows, stratified per-sequence
        // mask levels, then corrupt.
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
                best_loss = va_l; best_step = step; save_best(); stale = 0;
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

    // ── Inference network (batch = 1) ───────────────────────────────────────
    LinearLayer<FLOAT> g_embed(d_model, in_dim, seq_len);
    auto g_blocks = make_blocks(1);
    LinearLayer<FLOAT> g_proj(V, d_model, seq_len);
    copy_linear_weights(g_embed, embed);
    copy_linear_weights(g_proj, proj);
    for (int i = 0; i < num_blocks; ++i) copy_tf_weights(*g_blocks[i], *blocks[i]);
    list<Layer<FLOAT>*> gennn = build_net(g_blocks, g_embed, g_proj);
    freeze(gennn);

    // Run the denoiser once on a length-seq_len token array (tok[s] in [0,V] with
    // V = [MASK]); returns host logits (V, seq_len).
    auto denoise = [&](const vector<int>& tok) {
        auto enc = Matrix<float>::zeros(in_dim, seq_len);
        for (int s = 0; s < seq_len; ++s) {
            enc.elem(tok[s], s) = 1.0f;      // char one-hot (index V == MASK)
            enc.elem(V1 + s, s) = 1.0f;      // position one-hot
        }
        forward(gennn, Matrix<FLOAT>(enc));
        return as_host(g_proj.value());
    };

    // Softmax-sample a column of the logits at temperature; also return the
    // chosen token's probability as a confidence score.
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

    // MaskGIT-style progressive decoding. Start from `tok` (with some [MASK]
    // positions); over T rounds, predict all masked positions, then permanently
    // commit a growing fraction per a cosine schedule until none remain. The
    // positions to commit are chosen by log-confidence plus ANNEALED Gumbel
    // noise: early rounds explore (so decoding order is not locked to the same
    // easy tokens), late rounds fall back to greedy confidence. This both
    // improves sample quality and lets more denoising steps actually help.
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

            // Target number still masked after this round (cosine schedule to 0).
            int target = (k >= T) ? 0
                : (int)floor(initial * cos(0.5 * PI * (double)k / (double)T));
            target = min(target, n_masked);
            int reveal = n_masked - target;      // commit this many best positions

            vector<int> order;
            for (int s = 0; s < seq_len; ++s) if (tok[s] == MASK) order.push_back(s);
            partial_sort(order.begin(), order.begin() + reveal, order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
            for (int j = 0; j < reveal; ++j) { tok[order[j]] = cand[order[j]]; n_masked--; }
        }
        string out;
        for (int s = 0; s < seq_len; ++s) out += (tok[s] == MASK) ? '?' : idx_to_char[tok[s]];
        return out;
    };

    const int T_steps = 64;   // ~2 tokens revealed per step over seq_len=128
    cout << "\n--- Unconditional generation (from all-[MASK], " << T_steps
         << " denoising steps) ---\n";
    for (int k = 0; k < 3; ++k) {
        vector<int> tok(seq_len, MASK);
        string g = decode(tok, T_steps, 1.0f, 2.0f);
        for (auto& ch : g) ch = printable((unsigned char)ch);
        printf("  [%d] \"%s\"\n", k, g.c_str());
    }

    // Infilling: keep the first and last quarter of a held-out window as context,
    // mask the middle half, and let the model fill it in.
    cout << "\n--- Infilling (context kept, middle masked) ---\n";
    for (int k = 0; k < 2; ++k) {
        const int off = val_lo + (int)((long long)k * val_windows / 2);
        const int lo = seq_len / 4, hi = 3 * seq_len / 4;   // masked span [lo,hi)
        vector<int> tok(seq_len);
        string truth;
        for (int s = 0; s < seq_len; ++s) {
            const int c = c2i[(unsigned char)corpus[off + s]];
            tok[s] = (s >= lo && s < hi) ? MASK : c;
            truth += idx_to_char[c];
        }
        string filled = decode(tok, T_steps, 0.7f, 1.0f);   // context constrains: low temp/noise
        for (auto& ch : truth)  ch = printable((unsigned char)ch);
        for (auto& ch : filled) ch = printable((unsigned char)ch);
        // Mark the region that was masked with a caret underline.
        string marker(seq_len, ' ');
        for (int s = lo; s < hi; ++s) marker[s] = '^';
        printf("  truth:  \"%s\"\n  filled: \"%s\"\n  masked:  %s\n\n",
               truth.c_str(), filled.c_str(), marker.c_str());
    }

    return 0;
}
