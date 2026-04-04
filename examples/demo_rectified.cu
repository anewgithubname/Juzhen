/**
 * @file demo_rectified.cu
 * @brief training rectified flow
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This is an implementation of the rectified flow
 * https://arxiv.org/abs/2209.03003
 *
    Copyright (C) 2024 Song Liu (song.liu@bristol.ac.uk)

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

// #define CPU_ONLY
#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace Juzhen;

#ifdef CUDA
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
inline Matrix<CUDAfloat> vs(std::vector<MatrixView<CUDAfloat>> matrices) { return vstack(matrices); }
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
inline Matrix<MPSfloat> randn(int m, int n) { return Matrix<MPSfloat>::randn(m, n); }
inline Matrix<MPSfloat> ones(int m, int n) { return Matrix<MPSfloat>::ones(m, n); }
inline Matrix<MPSfloat> vs(std::vector<MatrixView<MPSfloat>> matrices) { return vstack(matrices); }
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
inline Matrix<float> vs(std::vector<MatrixView<float>> matrices) { return vstack<float>(matrices); }
#endif

static Matrix<FLOAT> sample_X0(int n)
{
    return randn(2, n);
}

static Matrix<float> sample_ring_host(int n, int n_modes = 10, float radius = 2.0f, float sigma = 0.10f)
{
    Matrix<float> X("ring", 2, n);
    std::uniform_int_distribution<int> mode_dist(0, n_modes - 1);
    std::normal_distribution<float> noise(0.0f, sigma);

    for (int i = 0; i < n; ++i) {
        int k = mode_dist(global_rand_gen);
        float theta = 2.0f * 3.1415926535f * static_cast<float>(k) / static_cast<float>(n_modes);
        float cx = radius * std::cos(theta);
        float cy = radius * std::sin(theta);
        X.elem(0, i) = cx + noise(global_rand_gen);
        X.elem(1, i) = cy + noise(global_rand_gen);
    }
    return X;
}

static Matrix<FLOAT> sample_X1(int n)
{
    auto host = sample_ring_host(n);
#ifdef CUDA
    return Matrix<CUDAfloat>(host);
#elif defined(APPLE_SILICON)
    return Matrix<MPSfloat>(host);
#else
    return host;
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

static Matrix<float> to_host_float(const Matrix<FLOAT>& M) {
#if defined(CUDA) || defined(APPLE_SILICON)
    return M.to_host();
#else
    return M;
#endif
}

static float ring_radius_mse(const Matrix<float>& pts, float radius) {
    double acc = 0.0;
    for (size_t i = 0; i < pts.num_col(); ++i) {
        const float x = pts.elem(0, i);
        const float y = pts.elem(1, i);
        const float r = std::sqrt(x * x + y * y);
        const float d = r - radius;
        acc += (double)d * (double)d;
    }
    return (float)(acc / std::max<size_t>(pts.num_col(), 1));
}

int compute()
{
#ifdef CUDA
    GPUSampler sampler(42);
#else
    global_rand_gen.seed(42);
#endif

    const int batchsize      = 256;
    const int train_iters    = env_int("RECTIFIED_TRAIN_ITERS",  8000);
    const int euler_steps    = env_int("RECTIFIED_EULER_STEPS",   100);
    const int log_every      = env_int("RECTIFIED_LOG_EVERY",      50);  // loss graph resolution
    const int vis_every      = env_int("RECTIFIED_VIS_EVERY",     500);  // scatter refresh during training
    const int euler_sleep_ms = env_int("RECTIFIED_EULER_SLEEP",    10);  // ms between ODE steps
    const int vis_n          = 600;
    const float lr           = env_float("RECTIFIED_LR", 1e-3f);
    const int d              = 2;
    const float target_radius = 2.0f;

    // ------------------------------------------------------------------
    // Shared state between training thread and UI render thread
    // ------------------------------------------------------------------
    std::mutex mtx;
    int   s_iter      = 0;
    float s_loss      = 0.0f;
    float s_ema       = 0.0f;
    float s_elapsed   = 0.0f;
    bool  s_done      = false;
    bool  s_sampling  = false;   // true while doing step-by-step Euler viz
    int   s_euler_cur = 0;
    float s_gen_mse   = 0.0f;
    float s_ref_mse   = 0.0f;
    std::vector<float> s_loss_history;                        // EMA loss per log_every step
    std::vector<std::pair<float,float>> s_gen_pts;
    std::vector<std::pair<float,float>> s_ref_pts;
    std::atomic<bool> stop_early{false};

    // Pre-populate reference ring points (static throughout)
    {
        auto ref = sample_ring_host(vis_n);
        s_ref_pts.resize(vis_n);
        for (int i = 0; i < vis_n; ++i)
            s_ref_pts[i] = {ref.elem(0, i), ref.elem(1, i)};
    }
    // Initial gen_pts = Gaussian noise (before any training)
    {
        auto z0 = to_host_float(sample_X0(vis_n));
        s_gen_pts.resize(vis_n);
        for (int k = 0; k < vis_n; ++k)
            s_gen_pts[k] = {z0.elem(0, k), z0.elem(1, k)};
    }

    // ------------------------------------------------------------------
    // FTXUI interactive UI  (two vertical panes)
    // ------------------------------------------------------------------
    using namespace ftxui;
    auto screen = ScreenInteractive::Fullscreen();

    auto fmt_f = [](float v, int prec) -> std::string {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    };

    auto renderer = Renderer([&] {
        std::lock_guard<std::mutex> lock(mtx);

        float prog = s_done ? 1.0f
                   : static_cast<float>(s_iter) / static_cast<float>(std::max(1, train_iters - 1));

        const std::string backend_str =
#ifdef CUDA
            "cuda";
#elif defined(APPLE_SILICON)
            "metal";
#else
            "cpu";
#endif

        // ── Header + stats bar ────────────────────────────────────────
        auto header = text(" Rectified Flow  [" + backend_str + "]"
                           "  iters=" + std::to_string(train_iters) +
                           "  euler=" + std::to_string(euler_steps) +
                           "  lr="    + fmt_f(lr, 4)) | bold;

        auto stats = hbox({
            text(" iter: "),
            text(std::to_string(s_iter) + "/" + std::to_string(train_iters)) | color(Color::Cyan),
            text("   loss: "), text(fmt_f(s_loss, 4)) | color(Color::Yellow),
            text("   ema: "),  text(fmt_f(s_ema,  4)) | color(Color::Green),
            text("   t: "),    text(fmt_f(s_elapsed, 1) + "s") | color(Color::White),
        });

        auto progress_bar = hbox({
            text(" "), gauge(prog) | flex | color(Color::Blue), text(" "),
        });

        // ── Top pane: real-time training loss graph ───────────────────
        const int LW = 200, LH = 50;
        Canvas lc(LW, LH);
        if ((int)s_loss_history.size() >= 2) {
            float lmin = *std::min_element(s_loss_history.begin(), s_loss_history.end());
            float lmax = *std::max_element(s_loss_history.begin(), s_loss_history.end());
            float lrng = std::max(lmax - lmin, 1e-4f);
            int N = (int)s_loss_history.size();
            auto lx = [&](int i) { return (int)((float)i / (N - 1) * (LW - 1)); };
            auto ly = [&](float v) { return LH - 1 - (int)((v - lmin) / lrng * (LH - 1)); };
            for (int i = 1; i < N; ++i)
                lc.DrawPointLine(lx(i-1), ly(s_loss_history[i-1]),
                                 lx(i),   ly(s_loss_history[i]),
                                 Color::Yellow);
            lc.DrawText(0,      0,      fmt_f(lmax, 2));
            lc.DrawText(0,      LH - 8, fmt_f(lmin, 2));
            lc.DrawText(LW / 2 - 12, LH - 8, "iter=" + std::to_string(s_iter));
        } else {
            lc.DrawText(LW / 2 - 20, LH / 2 - 4, "waiting for loss data...");
        }

        auto loss_pane = vbox({
            text(" Training Loss (EMA)") | bold | color(Color::Yellow),
            canvas(std::move(lc)) | border,
        });

        // ── Bottom pane: scatter (reference=blue, generated=red) ──────
        const int CW = 200, CH = 100;
        const float range_v = 4.5f;
        auto to_px = [&](float w, int sz) -> int {
            return std::clamp(static_cast<int>((w + range_v) / (2.f * range_v) * (sz - 1)), 0, sz - 1);
        };
        Canvas sc(CW, CH);
        for (auto& [wx, wy] : s_ref_pts)
            sc.DrawPoint(to_px(wx, CW), CH - 1 - to_px(wy, CH), true, Color::BlueLight);
        for (auto& [wx, wy] : s_gen_pts)
            sc.DrawPoint(to_px(wx, CW), CH - 1 - to_px(wy, CH), true, Color::RedLight);

        std::string scatter_title = s_sampling
            ? " ODE Sampling  step " + std::to_string(s_euler_cur) +
              "/" + std::to_string(euler_steps) +
              "   [red = generated   blue = reference]"
            : s_done
              ? " ODE Samples   [red]  vs  Reference [blue]"
                "   gen_MSE=" + fmt_f(s_gen_mse, 4) +
                "   ref_MSE=" + fmt_f(s_ref_mse, 4)
              : " ODE Samples (training preview, refreshes every " +
                std::to_string(vis_every) + " iters)"
                "   [red = generated   blue = reference]";

        auto scatter_pane = vbox({
            text(scatter_title) | bold | color(Color::Cyan),
            canvas(std::move(sc)) | border,
        });

        // ── Footer ────────────────────────────────────────────────────
        auto footer = (s_done)
            ? text(" Done!  Press 'q' to quit.") | color(Color::GreenLight) | bold
            : text(" Training...  Press 'q' to stop early.") | color(Color::Yellow);

        return vbox({
            header,
            stats | border,
            progress_bar,
            loss_pane,
            scatter_pane,
            footer,
        });
    });

    auto component = CatchEvent(renderer, [&](Event event) -> bool {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            stop_early = true;
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    // ------------------------------------------------------------------
    // Training + sampling thread
    // ------------------------------------------------------------------
    std::thread train_thread([&] {
        ReluLayer<FLOAT> L0(128, d + 1, batchsize),
                         L1(128, 128, batchsize),
                         L2(128, 128, batchsize);
        LinearLayer<FLOAT> L3(d, 128, batchsize);
        list<Layer<FLOAT>*> trainnn({&L3, &L2, &L1, &L0});
        for (auto* l : trainnn) {
            l->adamWstate().alpha = lr;
            l->adambstate().alpha = lr;
        }

        auto t_start = Clock::now();
        float ema_loss = 0.0f;
        bool  ema_init = false;

        // ── Training loop ─────────────────────────────────────────────
        for (int i = 0; i < train_iters && !stop_early; ++i) {
            auto X0_i  = sample_X0(batchsize);
            auto X1_i  = sample_X1(batchsize);
            auto t     = Matrix<FLOAT>::rand(1, batchsize);
            auto Xt_i  = hadmd(X0_i, ones(d, 1) * (1 - t)) + hadmd(X1_i, ones(d, 1) * t);
            auto inp_i = vs({Xt_i, t});
            LossLayer<FLOAT> L11(batchsize, X1_i - X0_i);
            trainnn.push_front(&L11);
            float loss = item(forward(trainnn, inp_i));
            backprop(trainnn, inp_i);
            trainnn.pop_front();

            if (!ema_init) { ema_loss = loss; ema_init = true; }
            else           { ema_loss = 0.95f * ema_loss + 0.05f * loss; }

            float elapsed = time_in_ms(t_start, Clock::now()) / 1000.0f;

            if (i % log_every == 0 || i == train_iters - 1) {
                bool do_scatter = (vis_every > 0) && (i % vis_every == 0 || i == train_iters - 1);

                // Quick Euler pass for scatter preview (runs in-place, no sleeps)
                std::vector<std::pair<float,float>> pts;
                if (do_scatter) {
                    auto Zt = sample_X0(vis_n);
                    const float dt = 1.0f / euler_steps;
                    for (int s = 0; s < euler_steps; ++s) {
                        float tv = (float)s / euler_steps;
                        auto t_row = Matrix<FLOAT>::ones(1, vis_n) * tv;
                        Zt = Zt + forward(trainnn, vs({Zt, t_row})) * dt;
                    }
                    auto Z_host = to_host_float(Zt);
                    pts.resize(vis_n);
                    for (int k = 0; k < vis_n; ++k)
                        pts[k] = {Z_host.elem(0, k), Z_host.elem(1, k)};
                }

                std::lock_guard<std::mutex> lock(mtx);
                s_iter = i; s_loss = loss; s_ema = ema_loss; s_elapsed = elapsed;
                s_loss_history.push_back(ema_loss);
                if (do_scatter) s_gen_pts = std::move(pts);
                screen.PostEvent(Event::Custom);
            }
        }

        if (stop_early) return;

        // ── Progressive ODE sampling (step by step with sleep) ────────
        {
            std::lock_guard<std::mutex> lock(mtx);
            s_sampling = true;
            screen.PostEvent(Event::Custom);
        }

        auto Zt = sample_X0(vis_n);
        // Show initial Gaussian noise before the flow starts
        {
            auto Z_host = to_host_float(Zt);
            std::lock_guard<std::mutex> lock(mtx);
            s_gen_pts.resize(vis_n);
            for (int k = 0; k < vis_n; ++k)
                s_gen_pts[k] = {Z_host.elem(0, k), Z_host.elem(1, k)};
            s_euler_cur = 0;
            screen.PostEvent(Event::Custom);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(euler_sleep_ms));

        const float dt = 1.0f / euler_steps;
        for (int s = 0; s < euler_steps && !stop_early; ++s) {
            float tv  = (float)s / euler_steps;
            auto t_row = Matrix<FLOAT>::ones(1, vis_n) * tv;
            Zt = Zt + forward(trainnn, vs({Zt, t_row})) * dt;

            auto Z_host = to_host_float(Zt);
            std::vector<std::pair<float,float>> pts(vis_n);
            for (int k = 0; k < vis_n; ++k)
                pts[k] = {Z_host.elem(0, k), Z_host.elem(1, k)};
            {
                std::lock_guard<std::mutex> lock(mtx);
                s_gen_pts   = std::move(pts);
                s_euler_cur = s + 1;
                screen.PostEvent(Event::Custom);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(euler_sleep_ms));
        }

        if (!stop_early) {
            auto Z_final = to_host_float(Zt);
            auto X1_ref  = to_host_float(sample_X1(vis_n));
            float gen_mse = ring_radius_mse(Z_final, target_radius);
            float ref_mse = ring_radius_mse(X1_ref,  target_radius);
            std::lock_guard<std::mutex> lock(mtx);
            s_sampling = false;
            s_done     = true;
            s_gen_mse  = gen_mse;
            s_ref_mse  = ref_mse;
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(component);
    train_thread.join();

    {
        std::lock_guard<std::mutex> lock(mtx);
        if (s_done) {
            cout << "[done] gen_radius_mse=" << fixed << setprecision(6) << s_gen_mse
                 << " ref_radius_mse=" << s_ref_mse << endl;
        } else {
            cout << "[stopped early at iter=" << s_iter
                 << " ema_loss=" << fixed << setprecision(4) << s_ema << "]" << endl;
        }
    }
    return 0;
}