/**
 * @file demo_cnn_rectified.cu
 * @brief Train a rectified flow on CIFAR-10 using a convolutional UNet.
 *
 * Architecture mirrors the PyTorch TinyUNet (encoder→bottleneck→decoder with
 * skip connections).  Time conditioning is realised by concatenating sin/cos
 * embeddings as extra spatial channels, which avoids needing GroupNorm or
 * per-block time projection layers that the library does not provide.
 *
 * CIFAR-10 binary batches are loaded directly from the standard format
 * distributed at https://www.cs.toronto.edu/~kriz/cifar.html.
 *
 * Environment variables:
 *   CIFAR10_DIR          path to extracted cifar-10-batches-bin/ (default: data/CIFAR10/cifar-10-batches-bin)
 *   RF_EPOCHS            number of training epochs           (default: 10)
 *   RF_BATCH_SIZE        mini-batch size                     (default: 128)
 *   RF_LR                Adam learning rate                  (default: 2e-4)
 *   RF_EULER_STEPS       ODE sampling steps                  (default: 100)
 *   RF_SEED              random seed                         (default: 42)
 *   RF_FID_SAMPLES       images to dump per epoch for FID    (default: 1000)
 */

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace Juzhen;

// ── Backend-agnostic helpers ─────────────────────────────────────────────────
#ifdef CUDA
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n)  { return Matrix<CUDAfloat>::ones(m, n);  }
inline Matrix<CUDAfloat> zeros(int m, int n) { return Matrix<CUDAfloat>::zeros(m, n); }
inline Matrix<CUDAfloat> vs(std::vector<MatrixView<CUDAfloat>> matrices) { return vstack(matrices); }
#elif defined(ROCM_HIP)
#define FLOAT ROCMfloat
inline Matrix<ROCMfloat> randn(int m, int n) { return Matrix<ROCMfloat>::randn(m, n); }
inline Matrix<ROCMfloat> ones(int m, int n)  { return Matrix<ROCMfloat>::ones(m, n);  }
inline Matrix<ROCMfloat> zeros(int m, int n) { return Matrix<ROCMfloat>::zeros(m, n); }
inline Matrix<ROCMfloat> vs(std::vector<MatrixView<ROCMfloat>> matrices) { return vstack(matrices); }
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
inline Matrix<MPSfloat> randn(int m, int n) { return Matrix<MPSfloat>::randn(m, n); }
inline Matrix<MPSfloat> ones(int m, int n)  { return Matrix<MPSfloat>::ones(m, n);  }
inline Matrix<MPSfloat> zeros(int m, int n) { return Matrix<MPSfloat>::zeros(m, n); }
inline Matrix<MPSfloat> vs(std::vector<MatrixView<MPSfloat>> matrices) { return vstack(matrices); }
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n)  { return Matrix<float>::ones(m, n);  }
inline Matrix<float> zeros(int m, int n) { return Matrix<float>::zeros(m, n); }
inline Matrix<float> vs(std::vector<MatrixView<float>> matrices) { return vstack<float>(matrices); }
#endif

static Matrix<float> to_host(const Matrix<FLOAT>& M) {
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
    return M.to_host();
#else
    return M;
#endif
}

static int env_int(const char* key, int default_v) {
    const char* v = std::getenv(key);
    if (!v) return default_v;
    return std::max(1, std::atoi(v));
}
static float env_float(const char* key, float default_v) {
    const char* v = std::getenv(key);
    if (!v) return default_v;
    return std::atof(v);
}
static std::string env_str(const char* key, const std::string& def) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : def;
}

// ── CIFAR-10 binary loader ───────────────────────────────────────────────────
// Standard format: each file has 10000 records of (1 label byte + 3072 pixel bytes).
// Pixels are stored R-plane(1024) G-plane(1024) B-plane(1024).
// We load all 5 training batches, normalise to [-1, 1], and return as a
// Matrix<float> of shape (3*32*32, N) = (3072, 50000).
static Matrix<float> load_cifar10(const std::string& dir) {
    const int records_per_file = 10000;
    const int n_files = 5;
    const int C = 3, H = 32, W = 32;
    const int pixels = C * H * W;               // 3072
    const int record_size = 1 + pixels;          // 3073
    const int total = n_files * records_per_file; // 50000

    Matrix<float> images("cifar10", pixels, total);

    int idx = 0;
    for (int f = 1; f <= n_files; ++f) {
        std::string path = dir + "/data_batch_" + std::to_string(f) + ".bin";
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) {
            std::cerr << "Cannot open " << path << std::endl;
            std::cerr << "Please download CIFAR-10 binary version and extract to " << dir << std::endl;
            ERROR_OUT;
        }
        std::vector<unsigned char> buf(record_size);
        for (int r = 0; r < records_per_file; ++r) {
            if (fread(buf.data(), 1, record_size, fp) != (size_t)record_size) {
                fclose(fp);
                std::cerr << "Unexpected EOF in " << path << std::endl;
                ERROR_OUT;
            }
            // skip label byte (buf[0])
            for (int p = 0; p < pixels; ++p) {
                images.elem(p, idx) = (float)buf[1 + p] / 127.5f - 1.0f;  // -> [-1, 1]
            }
            ++idx;
        }
        fclose(fp);
    }
    return images;
}

// ── Save PPM grid ────────────────────────────────────────────────────────────
// samples: shape (3072, n).  Write an nrow × ncol grid of 32×32 RGB images.
static void save_ppm(const std::string& path, const Matrix<float>& samples,
                     int ncol, int nrow) {
    const int C = 3, H = 32, W = 32;
    int grid_w = ncol * W;
    int grid_h = nrow * H;
    std::vector<unsigned char> img(grid_h * grid_w * 3, 0);

    int n = (int)samples.num_col();
    for (int i = 0; i < std::min(n, nrow * ncol); ++i) {
        int gr = i / ncol, gc = i % ncol;
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float v = samples.elem(c * H * W + y * W + x, i);
                    v = (v + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
                    v = std::max(0.0f, std::min(1.0f, v));
                    int py = gr * H + y;
                    int px = gc * W + x;
                    img[(py * grid_w + px) * 3 + c] = (unsigned char)(v * 255.0f);
                }
            }
        }
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { std::cerr << "Cannot write " << path << std::endl; return; }
    fprintf(fp, "P6\n%d %d\n255\n", grid_w, grid_h);
    fwrite(img.data(), 1, img.size(), fp);
    fclose(fp);
}

// ── Save raw binary samples for FID computation ─────────────────────────────
// Writes N images as float32 in [-1,1], layout: N * 3072 floats (CHW per image).
// Header: int32 N, int32 C=3, int32 H=32, int32 W=32, then N*3072 float32.
static void save_raw_samples(const std::string& path,
                             const Matrix<float>& samples) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { std::cerr << "Cannot write " << path << std::endl; return; }
    int32_t header[4] = {(int32_t)samples.num_col(), 3, 32, 32};
    fwrite(header, sizeof(int32_t), 4, fp);
    // Write column-by-column (each column is one image, 3072 floats)
    for (int i = 0; i < (int)samples.num_col(); ++i) {
        for (int p = 0; p < (int)samples.num_row(); ++p) {
            float v = samples.elem(p, i);
            fwrite(&v, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

// ── UNet for rectified flow ──────────────────────────────────────────────────
// Input channels: 3 (RGB x_t) + 8 (4-freq sin/cos time embed) = 11
// Output channels: 3 (velocity prediction)
//
// Architecture (32×32 spatial, 3-level):
//   enc1  : 11 → 64,  k3 s1 p1  (32×32)
//   enc2  : 64 → 128, k3 s2 p1  (16×16)
//   enc3  : 128→ 256, k3 s2 p1  (8×8)
//   mid   : 256→ 256, k3 s1 p1  (8×8)
//   up2   : 256→ 128, k4 s2 p1  (16×16)
//   dec2  : 256→ 128, k3 s1 p1  (16×16)  (cat with enc2 skip)
//   up1   : 128→ 64,  k4 s2 p1  (32×32)
//   dec1  : 128→ 64,  k3 s1 p1  (32×32)  (cat with enc1 skip)
//   head  : 64 → 3,   k3 s1 p1  (32×32, no ReLU)
static constexpr int TIME_NFREQ = 4;       // sinusoidal frequencies
static constexpr int TIME_CH = TIME_NFREQ * 2; // 8
static constexpr float TIME_FREQS[TIME_NFREQ] = {
    2.0f * 3.14159265358979323846f,   // 2π
    4.0f * 3.14159265358979323846f,   // 4π
    8.0f * 3.14159265358979323846f,   // 8π
   16.0f * 3.14159265358979323846f    // 16π
};

class RectifiedUNet {
public:
    static constexpr int H = 32, W = 32;
    static constexpr int in_ch = 3 + TIME_CH;  // 11
    static constexpr int ch1 = 64;
    static constexpr int ch2 = 128;
    static constexpr int ch3 = 256;
    static constexpr int out_ch = 3;
    static constexpr int d32 = H * W;          // 1024
    static constexpr int d16 = (H/2) * (W/2);  // 256
    static constexpr int d8  = (H/4) * (W/4);  // 64

    ConvLayer enc1;      // in_ch → ch1, 32×32
    ConvLayer enc2;      // ch1   → ch2, 16×16
    ConvLayer enc3;      // ch2   → ch3, 8×8
    ConvLayer mid;       // ch3   → ch3, 8×8
    ConvTransLayer up2;  // ch3   → ch2, 16×16
    ConvLayer dec2;      // ch2*2 → ch2, 16×16 (skip from enc2)
    ConvTransLayer up1;  // ch2   → ch1, 32×32
    ConvLayer dec1;      // ch1*2 → ch1, 32×32 (skip from enc1)
    ConvLayer head;      // ch1   → out_ch, 32×32 (no relu)

    Matrix<FLOAT> e1, e2, e3, m_out, u2_out, cat2, d2_out, u1_out, cat1;

    explicit RectifiedUNet(int bs)
        : enc1(bs, in_ch,  H,   W,   ch1, 3, 3, 1, 1, true),   // 32→32
          enc2(bs, ch1,    H,   W,   ch2, 3, 3, 1, 2, true),   // 32→16
          enc3(bs, ch2,    H/2, W/2, ch3, 3, 3, 1, 2, true),   // 16→8
          mid (bs, ch3,    H/4, W/4, ch3, 3, 3, 1, 1, true),   // 8→8
          up2 (bs, ch3,    H/4, W/4, ch2, 4, 4, 1, 2, true),   // 8→16
          dec2(bs, ch2*2,  H/2, W/2, ch2, 3, 3, 1, 1, true),   // 16→16
          up1 (bs, ch2,    H/2, W/2, ch1, 4, 4, 1, 2, true),   // 16→32
          dec1(bs, ch1*2,  H,   W,   ch1, 3, 3, 1, 1, true),   // 32→32
          head(bs, ch1,    H,   W,   out_ch, 3, 3, 1, 1, false), // 32→32
          e1("e1", ch1 * d32, bs),
          e2("e2", ch2 * d16, bs),
          e3("e3", ch3 * d8, bs),
          m_out("m_out", ch3 * d8, bs),
          u2_out("u2", ch2 * d16, bs),
          cat2("cat2", ch2 * 2 * d16, bs),
          d2_out("d2", ch2 * d16, bs),
          u1_out("u1", ch1 * d32, bs),
          cat1("cat1", ch1 * 2 * d32, bs)
    {}

    void set_lr(float alpha) {
        for (auto* L : {(Layer<FLOAT>*)&enc1, (Layer<FLOAT>*)&enc2,
                        (Layer<FLOAT>*)&enc3, (Layer<FLOAT>*)&mid,
                        (Layer<FLOAT>*)&up2,  (Layer<FLOAT>*)&dec2,
                        (Layer<FLOAT>*)&up1,  (Layer<FLOAT>*)&dec1,
                        (Layer<FLOAT>*)&head}) {
            L->adamWstate().alpha = alpha;
            L->adambstate().alpha = alpha;
        }
    }

    const Matrix<FLOAT>& fwd(const Matrix<FLOAT>& inp) {
        enc1.eval(inp);
        e1 = enc1.value();

        enc2.eval(e1);
        e2 = enc2.value();

        enc3.eval(e2);
        e3 = enc3.value();

        mid.eval(e3);
        m_out = mid.value();

        up2.eval(m_out);
        cat2 = vs({MatrixView<FLOAT>(up2.value()), MatrixView<FLOAT>(e2)});
        dec2.eval(cat2);
        d2_out = dec2.value();

        up1.eval(d2_out);
        cat1 = vs({MatrixView<FLOAT>(up1.value()), MatrixView<FLOAT>(e1)});
        dec1.eval(cat1);

        head.eval(dec1.value());
        return head.value();
    }

    void bwd(const Matrix<FLOAT>& inp, Matrix<FLOAT>&& g) {
        auto g_head = head.backward(dec1.value(), std::move(g));
        auto g_cat1 = dec1.backward(cat1, std::move(g_head));

        const size_t skip1_sz = (size_t)ch1 * (size_t)d32;
        auto g_u1   = g_cat1.rows(0, skip1_sz);
        auto g_skip1 = g_cat1.rows(skip1_sz, g_cat1.num_row());

        auto g_d2 = up1.backward(d2_out, std::move(g_u1));
        auto g_cat2 = dec2.backward(cat2, std::move(g_d2));

        const size_t skip2_sz = (size_t)ch2 * (size_t)d16;
        auto g_u2   = g_cat2.rows(0, skip2_sz);
        auto g_skip2 = g_cat2.rows(skip2_sz, g_cat2.num_row());

        auto g_m  = up2.backward(m_out, std::move(g_u2));
        auto g_e3 = mid.backward(e3, std::move(g_m));
        auto g_e2 = enc3.backward(e2, std::move(g_e3));
        g_e2 = g_e2 + g_skip2;
        auto g_e1 = enc2.backward(e1, std::move(g_e2));
        g_e1 = g_e1 + g_skip1;
        enc1.backward(inp, std::move(g_e1));
    }
};

// ── Cosine LR schedule with warmup ───────────────────────────────────────────
static float cosine_lr(int step, int total_steps, float base_lr) {
    int warmup = total_steps / 20;  // 5% warmup
    float min_lr = base_lr * 0.01f;
    if (step < warmup) {
        return min_lr + (base_lr - min_lr) * (float)step / (float)warmup;
    }
    float progress = (float)(step - warmup) / (float)(total_steps - warmup);
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + std::cos(3.14159265358979323846f * progress));
}

// ── Build time-embedding channels ───────────────────────────────────────────
// Returns a matrix of shape (TIME_CH * H * W, batchsize).
// For each frequency f in TIME_FREQS, creates sin(f*t) and cos(f*t) planes.
static void build_time_channels(Matrix<float>& out,
                                const Matrix<float>& t_host,
                                int H, int W, int batchsize) {
    const int hw = H * W;
    for (int b = 0; b < batchsize; ++b) {
        float tv = t_host.elem(0, b);
        int row = 0;
        for (int f = 0; f < TIME_NFREQ; ++f) {
            float sv = std::sin(TIME_FREQS[f] * tv);
            float cv = std::cos(TIME_FREQS[f] * tv);
            for (int p = 0; p < hw; ++p) out.elem(row + p, b) = sv;
            row += hw;
            for (int p = 0; p < hw; ++p) out.elem(row + p, b) = cv;
            row += hw;
        }
    }
}

// ── Euler ODE sampling ───────────────────────────────────────────────────────
static Matrix<FLOAT> euler_sample(RectifiedUNet& net, int batchsize, int euler_steps) {
    const int C = 3, H = 32, W = 32;
    const int d = C * H * W;
    const int hw = H * W;

    auto Zs = randn(d, batchsize);
    float dt = 1.0f / euler_steps;

    std::list<Layer<FLOAT>*> all_layers{
        (Layer<FLOAT>*)&net.enc1, (Layer<FLOAT>*)&net.enc2,
        (Layer<FLOAT>*)&net.enc3, (Layer<FLOAT>*)&net.mid,
        (Layer<FLOAT>*)&net.up2,  (Layer<FLOAT>*)&net.dec2,
        (Layer<FLOAT>*)&net.up1,  (Layer<FLOAT>*)&net.dec1,
        (Layer<FLOAT>*)&net.head};
    freeze(all_layers);

    Matrix<float> t_host("t_s", 1, batchsize);
    Matrix<float> temb_host("temb_s", TIME_CH * hw, batchsize);

    for (int s = 0; s < euler_steps; ++s) {
        float tv = (float)s / euler_steps;
        for (int b = 0; b < batchsize; ++b)
            t_host.elem(0, b) = tv;
        build_time_channels(temb_host, t_host, H, W, batchsize);

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
        Matrix<FLOAT> temb(temb_host);
#else
        Matrix<FLOAT>& temb = temb_host;
#endif

        auto inp_s = vs({MatrixView<FLOAT>(Zs),
                         MatrixView<FLOAT>(temb)});
        const auto& vel = net.fwd(inp_s);
        Zs = Zs + vel * dt;
    }

    unfreeze(all_layers);
    return Zs;
}

// ── Entry point ──────────────────────────────────────────────────────────────
int compute() {
    const int seed        = env_int("RF_SEED", 42);
    const int n_epochs    = env_int("RF_EPOCHS", 10);
    const int batchsize   = env_int("RF_BATCH_SIZE", 128);
    const float lr        = env_float("RF_LR", 2e-4f);
    const int euler_steps = env_int("RF_EULER_STEPS", 100);
    const int fid_samples = env_int("RF_FID_SAMPLES", 1000);
    const int log_every   = 10;
    const std::string cifar_dir = env_str("CIFAR10_DIR",
        std::string(PROJECT_DIR) + "/data/CIFAR10/cifar-10-batches-bin");
    const std::string out_dir = std::string(PROJECT_DIR) + "/res/generated_cifar10";

#ifdef CUDA
    GPUSampler sampler(seed);
#endif
    global_rand_gen.seed(seed);

    // Create output directory
    {
#ifdef _WIN32
        std::string cmd = "mkdir \"" + out_dir + "\" 2>nul";
#else
        std::string cmd = "mkdir -p " + out_dir;
#endif
        if (system(cmd.c_str()) != 0)
            std::cerr << "Warning: could not create " << out_dir << std::endl;
    }

    const std::string backend_str =
#ifdef CUDA
        "cuda";
#elif defined(ROCM_HIP)
        "rocm";
#elif defined(APPLE_SILICON)
        "metal";
#else
        "cpu";
#endif

    cout << "=== Rectified Flow CIFAR-10 [" << backend_str << "] ===" << endl;
    cout << "  epochs=" << n_epochs << "  bs=" << batchsize
         << "  lr=" << lr << "  euler_steps=" << euler_steps
         << "  fid_samples=" << fid_samples
         << "  seed=" << seed
         << "  schedule=cosine"
         << "  time_freqs=" << TIME_NFREQ << endl;

    // ── Load CIFAR-10 ────────────────────────────────────────────────────
    cout << "Loading CIFAR-10 from: " << cifar_dir << endl;
    Matrix<float> cifar_host = load_cifar10(cifar_dir);
    const int total_images = (int)cifar_host.num_col();
    const int pixels = (int)cifar_host.num_row();
    cout << "Loaded " << total_images << " images (" << pixels << " dims)" << endl;

    const int steps_per_epoch = total_images / batchsize;
    const int total_steps = n_epochs * steps_per_epoch;

    // ── Create network ───────────────────────────────────────────────────
    const int C = 3, H = 32, W = 32;
    const int d = C * H * W;

    RectifiedUNet net(batchsize);
    net.set_lr(lr);

    auto t_start = Clock::now();
    float ema_loss = 0.0f;
    bool ema_init = false;

    std::vector<int> perm(total_images);
    std::iota(perm.begin(), perm.end(), 0);

    // ── Training loop ────────────────────────────────────────────────────
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        std::shuffle(perm.begin(), perm.end(), global_rand_gen);

        for (int step = 0; step < steps_per_epoch; ++step) {
            int global_step = epoch * steps_per_epoch + step;

            // Cosine LR schedule with warmup
            float cur_lr = cosine_lr(global_step, total_steps, lr);
            net.set_lr(cur_lr);

            int offset = step * batchsize;

            // Build batch on host
            Matrix<float> x1_host("x1", d, batchsize);
            for (int b = 0; b < batchsize; ++b) {
                int img_idx = perm[offset + b];
                for (int p = 0; p < d; ++p)
                    x1_host.elem(p, b) = cifar_host.elem(p, img_idx);
            }

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            Matrix<FLOAT> x1(x1_host);
#else
            Matrix<FLOAT>& x1 = x1_host;
#endif

            // Sample random time t ~ U[0,1]
            Matrix<float> t_host("t_host", 1, batchsize);
            for (int b = 0; b < batchsize; ++b)
                t_host.elem(0, b) = udist(global_rand_gen);

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            Matrix<FLOAT> t_val(t_host);
#else
            Matrix<FLOAT>& t_val = t_host;
#endif

            // x0 ~ N(0,1)
            auto x0 = randn(d, batchsize);

            // x_t = (1-t)*x0 + t*x1
            auto ones_d1 = ones(d, 1);
            auto one_minus_t = ones(1, batchsize) - t_val;
            auto x_t = hadmd(x0, ones_d1 * one_minus_t) + hadmd(x1, ones_d1 * t_val);

            // target velocity = x1 - x0
            auto target_v = x1 - x0;

            // Multi-frequency time encoding
            Matrix<float> temb_host("temb", TIME_CH * H * W, batchsize);
            build_time_channels(temb_host, t_host, H, W, batchsize);
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            Matrix<FLOAT> temb(temb_host);
#else
            Matrix<FLOAT>& temb = temb_host;
#endif

            // Build input: [x_t (3*1024); time_channels (8*1024)]
            auto inp = vs({MatrixView<FLOAT>(x_t),
                           MatrixView<FLOAT>(temb)});

            // Forward pass
            const auto& pred_v = net.fwd(inp);

            // MSE loss + backward
            auto diff = pred_v - target_v;
            float loss = item(sum(sum(square(Matrix<FLOAT>(diff)), 0), 1)) / (float)batchsize;
            auto grad = 2.0f * diff / (float)batchsize;
            net.bwd(inp, std::move(grad));

            if (!ema_init) { ema_loss = loss; ema_init = true; }
            else           { ema_loss = 0.95f * ema_loss + 0.05f * loss; }

            if (step % log_every == 0 || step == steps_per_epoch - 1) {
                float elapsed = time_in_ms(t_start, Clock::now()) / 1000.0f;
                int global_step = epoch * steps_per_epoch + step;
                cout << "\r  epoch " << (epoch + 1) << "/" << n_epochs
                     << "  step " << (step + 1) << "/" << steps_per_epoch
                     << "  loss=" << fixed << setprecision(5) << loss
                     << "  ema=" << ema_loss
                     << "  lr=" << scientific << setprecision(1) << cur_lr
                     << "  [" << global_step + 1 << "/" << total_steps << "]"
                     << "  " << fixed << setprecision(1) << elapsed << "s    " << flush;
            }
        }

        float elapsed = time_in_ms(t_start, Clock::now()) / 1000.0f;
        cout << endl;
        cout << "Epoch " << (epoch + 1) << " done.  ema_loss=" << fixed << setprecision(5) << ema_loss
             << "  elapsed=" << fixed << setprecision(1) << elapsed << "s" << endl;

        // ── End-of-epoch sampling (1000 images for FID) ────────────────
        cout << "  Generating " << fid_samples << " samples (" << euler_steps << " Euler steps)..." << flush;
        int n_batches = (fid_samples + batchsize - 1) / batchsize;
        Matrix<float> all_samples("all_samples", d, fid_samples);
        int filled = 0;
        for (int bi = 0; bi < n_batches && filled < fid_samples; ++bi) {
            auto Zs = euler_sample(net, batchsize, euler_steps);
            auto Zh = to_host(Zs);
            int take = std::min(batchsize, fid_samples - filled);
            for (int i = 0; i < take; ++i)
                for (int p = 0; p < d; ++p)
                    all_samples.elem(p, filled + i) = Zh.elem(p, i);
            filled += take;
        }

        // Save raw binary for FID computation
        std::string bin_path = out_dir + "/epoch_" + std::to_string(epoch + 1) + "_samples.bin";
        save_raw_samples(bin_path, all_samples);

        // Also save small PPM grid for visual inspection
        int sample_n = std::min(64, fid_samples);
        Matrix<float> grid_samples("grid", d, sample_n);
        for (int i = 0; i < sample_n; ++i)
            for (int p = 0; p < d; ++p)
                grid_samples.elem(p, i) = all_samples.elem(p, i);
        int ncol = (int)std::ceil(std::sqrt((double)sample_n));
        int nrow = (sample_n + ncol - 1) / ncol;
        std::string ppm_path = out_dir + "/epoch_" + std::to_string(epoch + 1) + ".ppm";
        save_ppm(ppm_path, grid_samples, ncol, nrow);
        cout << " saved to " << bin_path << endl;
    }

    // ── Final samples ────────────────────────────────────────────────────
    cout << "\nGenerating final samples (" << euler_steps << " Euler steps)..." << flush;
    auto Zs = euler_sample(net, batchsize, euler_steps);
    auto Zh = to_host(Zs);

    int sample_n = std::min(64, batchsize);
    Matrix<float> final_samples("final", d, sample_n);
    for (int i = 0; i < sample_n; ++i)
        for (int p = 0; p < d; ++p)
            final_samples.elem(p, i) = Zh.elem(p, i);

    int ncol = (int)std::ceil(std::sqrt((double)sample_n));
    int nrow = (sample_n + ncol - 1) / ncol;
    std::string final_path = out_dir + "/final_samples.ppm";
    save_ppm(final_path, final_samples, ncol, nrow);

    float total_time = time_in_ms(t_start, Clock::now()) / 1000.0f;
    cout << " done!" << endl;
    cout << "\n=== Training complete ===" << endl;
    cout << "  final_ema_loss=" << fixed << setprecision(6) << ema_loss << endl;
    cout << "  total_time=" << fixed << setprecision(1) << total_time << "s" << endl;
    cout << "  samples saved to: " << out_dir << "/" << endl;

    return 0;
}
