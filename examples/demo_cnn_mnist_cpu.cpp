/**
 * @file demo_cnn_mnist_cpu.cpp
 * @brief MNIST classification using a CNN (CPU implementation) with live FTXUI display
 */

#include "../ml/layer.hpp"
#include "../ml/dataloader.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

using namespace std;
using namespace Juzhen;

static int env_int(const char* key, int default_v) {
    const char* v = std::getenv(key);
    if (!v) return default_v;
    return std::atoi(v);
}

static std::string env_str(const char* key, const std::string& default_v) {
    const char* v = std::getenv(key);
    if (!v) return default_v;
    return std::string(v);
}

static Matrix<float> make_host_random_matrix(size_t rows, size_t cols, std::mt19937& rng, float stddev) {
    Matrix<float> m("det_init", rows, cols);
    std::normal_distribution<float> nd(0.0f, stddev);
    for (size_t j = 0; j < cols; ++j)
        for (size_t i = 0; i < rows; ++i)
            m.elem(i, j) = nd(rng);
    return m;
}

struct SampleViz {
    std::vector<float> pixels; // C*H*W floats in [0,1]
    int true_label = -1;
    int pred_label = -1;
};

int compute() {
    using namespace ftxui;

    auto t_start = Clock::now();

    const int seed      = env_int("CNN_MNIST_SEED", 43);
    const int n_epochs  = env_int("CNN_MNIST_EPOCHS", 10);
    const int log_every = 10;   // loss graph resolution
    const int vis_every = 100;  // sample prediction refresh rate
    const std::string loss_path = env_str("CNN_MNIST_LOSS_PATH",
        PROJECT_DIR + std::string("/res/cnn_mnist_cpu_loss.csv"));

    global_rand_gen.seed(seed);

    const int C = 1, H = 28, W = 28;
    const int kH = 3, kW = 3, pad = 1;
    const int C1 = 16, S1 = 1;
    const int H1 = (H + 2*pad - kH)/S1 + 1, W1 = (W + 2*pad - kW)/S1 + 1;
    const int C2 = 32, S2 = 2;
    const int H2 = (H1 + 2*pad - kH)/S2 + 1, W2 = (W1 + 2*pad - kW)/S2 + 1;
    const int C3 = 64, S3 = 2;
    const int H3 = (H2 + 2*pad - kH)/S3 + 1, W3 = (W2 + 2*pad - kW)/S3 + 1;
    const int conv_out_dim = C3 * H3 * W3;
    const int fc_hidden = 256, k = 10, batchsize = 64;
    const int train_samples = 60000;
    const int steps_per_epoch = train_samples / batchsize;
    const int total_steps = n_epochs * steps_per_epoch;
    const int n_vis = 8;

    // ── Shared UI state ────────────────────────────────────────────────
    std::mutex mtx;
    int   s_global_step = 0;
    int   s_epoch   = 0;
    float s_loss    = 0.0f;
    float s_acc     = -1.0f; // test accuracy (1 - misclassification rate)
    float s_elapsed = 0.0f;
    bool  s_done    = false;
    std::vector<float>     s_loss_history;
    std::vector<SampleViz> s_samples(n_vis);
    std::atomic<bool>      stop_early{false};

    // ── FTXUI setup ────────────────────────────────────────────────────
    auto screen = ScreenInteractive::Fullscreen();

    auto fmt_f = [](float v, int p) -> std::string {
        std::ostringstream ss; ss << std::fixed << std::setprecision(p) << v; return ss.str();
    };

    auto renderer = Renderer([&] {
        std::lock_guard<std::mutex> lk(mtx);

        float prog = s_done ? 1.0f
               : static_cast<float>(s_global_step) / static_cast<float>(std::max(1, total_steps - 1));

        // Header + stats
        auto header = text(" CNN MNIST  [cpu]   epochs=" + std::to_string(n_epochs)
                           + "   seed=" + std::to_string(seed)) | bold;
        std::string acc_str = s_acc < 0.0f ? "n/a" : fmt_f(s_acc * 100.f, 1) + "%";
        auto stats = hbox({
            text(" epoch: "),   text(std::to_string(s_epoch) + "/" + std::to_string(n_epochs)) | color(Color::Cyan),
            text("   step: "),  text(std::to_string(s_global_step) + "/" + std::to_string(total_steps)) | color(Color::Cyan),
            text("   loss: "),  text(fmt_f(s_loss, 4))  | color(Color::Yellow),
            text("   acc: "),   text(acc_str)            | color(Color::Green),
            text("   t: "),     text(fmt_f(s_elapsed, 1) + "s") | color(Color::White),
        });
        auto pbar = hbox({ text(" "), gauge(prog) | flex | color(Color::Blue), text(" ") });

        // Loss graph
        const int LW = 200, LH = 40;
        Canvas lc(LW, LH);
        if ((int)s_loss_history.size() >= 2) {
            float lmin = *std::min_element(s_loss_history.begin(), s_loss_history.end());
            float lmax = *std::max_element(s_loss_history.begin(), s_loss_history.end());
            float lrng = std::max(lmax - lmin, 1e-4f);
            int N = (int)s_loss_history.size();
            auto lx = [&](int i) { return (int)((float)i / (N-1) * (LW-1)); };
            auto ly = [&](float v) { return LH - 1 - (int)((v - lmin) / lrng * (LH-1)); };
            for (int i = 1; i < N; ++i)
                lc.DrawPointLine(lx(i-1), ly(s_loss_history[i-1]),
                                 lx(i),   ly(s_loss_history[i]), Color::Yellow);
            lc.DrawText(0,       0,      fmt_f(lmax, 2));
            lc.DrawText(0,       LH - 8, fmt_f(lmin, 2));
            lc.DrawText(LW/2 - 16, LH - 8, "step=" + std::to_string(s_global_step));
        } else {
            lc.DrawText(LW/2 - 20, LH/2 - 4, "waiting for loss data...");
        }
        auto loss_pane = vbox({
            text(" Training Loss") | bold | color(Color::Yellow),
            canvas(std::move(lc)) | border,
        });

        // Sample grid: 8 test images with true / predicted labels
        std::vector<Element> cells;
        for (auto& s : s_samples) {
            Canvas dc(56, 56);
            if (!s.pixels.empty()) {
                for (int py = 0; py < 28; ++py)
                    for (int px = 0; px < 28; ++px) {
                        float v = s.pixels[py * 28 + px];
                        if (v > 0.3f) {
                            dc.DrawPoint(px*2,   py*2,   true, Color::White);
                            dc.DrawPoint(px*2+1, py*2,   true, Color::White);
                            dc.DrawPoint(px*2,   py*2+1, true, Color::White);
                            dc.DrawPoint(px*2+1, py*2+1, true, Color::White);
                        }
                    }
            }
            std::string lbl = "T:" + std::to_string(s.true_label) + " P:"
                            + (s.pred_label < 0 ? "?" : std::to_string(s.pred_label));
            bool correct = (s.pred_label >= 0 && s.pred_label == s.true_label);
            Color lbl_color = (s.pred_label < 0) ? Color::White
                            : correct ? Color::GreenLight : Color::RedLight;
            cells.push_back(vbox({
                canvas(std::move(dc)) | border,
                text(lbl) | center | color(lbl_color),
            }));
        }
        auto sample_pane = vbox({
            text(" Sample Predictions  (first 8 test images)") | bold | color(Color::Cyan),
            hbox(cells),
        });

        auto footer = s_done
            ? text(" Done!  Press 'q' to quit.") | color(Color::GreenLight) | bold
            : text(" Training...  Press 'q' to stop early.") | color(Color::Yellow);

        return vbox({ header, stats | border, pbar, loss_pane, sample_pane, footer });
    });

    auto component = CatchEvent(renderer, [&](Event event) -> bool {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            stop_early = true;
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    // ── Training thread ────────────────────────────────────────────────
    std::thread train_thread([&] {
        const std::string mnist_dir = PROJECT_DIR + std::string("/datasets/MNIST");

        // Load a fixed test batch for sample visualization
        DataLoader<float, int> vis_loader(mnist_dir, "test", batchsize);
        auto [vis_X_cpu, vis_y_mat] = vis_loader.next_batch();
        while (vis_X_cpu.num_col() != static_cast<size_t>(batchsize))
            std::tie(vis_X_cpu, vis_y_mat) = vis_loader.next_batch();

        std::vector<SampleViz> init_samples(n_vis);
        for (int i = 0; i < n_vis; ++i) {
            init_samples[i].true_label = static_cast<int>(vis_y_mat.elem(0, i));
            init_samples[i].pixels.resize(C * H * W);
            for (int p = 0; p < C * H * W; ++p)
                init_samples[i].pixels[p] = vis_X_cpu.elem(p, i) / 255.0f;
        }
        auto vis_input = Matrix<float>(vis_X_cpu / 255.0f); // normalized vis batch
        {
            std::lock_guard<std::mutex> lk(mtx);
            s_samples = init_samples;
            screen.PostEvent(Event::Custom);
        }

        // Build layers
        ConvLayer conv1(batchsize, C,  H,  W,  C1, kH, kW, pad, S1);
        ConvLayer conv2(batchsize, C1, H1, W1, C2, kH, kW, pad, S2);
        ConvLayer conv3(batchsize, C2, H2, W2, C3, kH, kW, pad, S3);
        ReluLayer<float>   L1(fc_hidden, conv_out_dim, batchsize);
        LinearLayer<float> L2(k,         fc_hidden,    batchsize);
        list<Layer<float>*> trainnn = { &L2, &L1, &conv3, &conv2, &conv1 };

        std::mt19937 init_rng(seed);
        for (auto* layer : trainnn) {
            layer->W() = make_host_random_matrix(layer->W().num_row(), layer->W().num_col(), init_rng, 0.01f);
            layer->b() = make_host_random_matrix(layer->b().num_row(), layer->b().num_col(), init_rng, 0.01f);
        }

        DataLoader<float, int> train_loader(mnist_dir, "train", batchsize);
        const size_t n_test_batches = 10000 / batchsize;
        std::vector<float> loss_history;
        loss_history.reserve(total_steps);

        for (int epoch = 0; epoch < n_epochs && !stop_early; ++epoch) {
            for (int step = 0; step < steps_per_epoch && !stop_early; ++step) {
                const int global_step = epoch * steps_per_epoch + step;

                auto [X_i_cpu, y_i_idx] = train_loader.next_batch();
                while (X_i_cpu.num_col() != static_cast<size_t>(batchsize))
                    std::tie(X_i_cpu, y_i_idx) = train_loader.next_batch();
                auto X_i = Matrix<float>(X_i_cpu / 255.0f);
                auto Y_i = Matrix<float>(one_hot(y_i_idx, k));

                forward(trainnn, X_i);
                LogisticLayer<float> loss_layer(batchsize, std::move(Y_i));
                loss_layer.eval(L2.value());
                float cur_loss = item(loss_layer.value());
                loss_history.push_back(cur_loss);
                trainnn.push_front(&loss_layer);
                backprop(trainnn, X_i);
                trainnn.pop_front();

                float elapsed = (float)time_in_ms(t_start, Clock::now()) / 1000.0f;
                bool do_log = (global_step % log_every == 0) || (global_step == total_steps - 1);
                bool do_vis = (global_step % vis_every == 0) || (global_step == total_steps - 1);

            // Run inference on fixed vis batch to update sample predictions
                std::vector<int> preds;
                if (do_vis) {
                    list<Layer<float>*> infnn = { &L2, &L1, &conv3, &conv2, &conv1 };
                    auto out = forward(infnn, vis_input); // shape: (k, batchsize)
                    preds.resize(n_vis);
                    for (int i = 0; i < n_vis; ++i) {
                        int best = 0; float bval = out.elem(0, i);
                        for (int c = 1; c < k; ++c)
                            if (out.elem(c, i) > bval) { bval = out.elem(c, i); best = c; }
                        preds[i] = best;
                    }
                }

                if (do_log || do_vis) {
                    std::lock_guard<std::mutex> lk(mtx);
                    s_global_step = global_step;
                    s_epoch   = epoch + 1;
                    s_loss    = cur_loss;
                    s_elapsed = elapsed;
                    if (do_log) s_loss_history.push_back(cur_loss);
                    if (do_vis)
                        for (int i = 0; i < n_vis; ++i)
                            s_samples[i].pred_label = preds[i];
                    screen.PostEvent(Event::Custom);
                }
            }

            // Test evaluation at the end of each epoch.
            float total_err = 0.0f;
            DataLoader<float, int> test_loader(mnist_dir, "test", batchsize);
            for (size_t b = 0; b < n_test_batches; ++b) {
                auto [XT_b_cpu, yT_b_idx] = test_loader.next_batch();
                if (XT_b_cpu.num_col() != static_cast<size_t>(batchsize)) continue;
                auto XT_b = Matrix<float>(XT_b_cpu / 255.0f);
                auto YT_b = Matrix<float>(one_hot(yT_b_idx, k));
                ZeroOneLayer<float> eval_loss(batchsize, std::move(YT_b));
                list<Layer<float>*> evalnn = { &eval_loss, &L2, &L1, &conv3, &conv2, &conv1 };
                total_err += forward(evalnn, XT_b).elem(0, 0);
            }
            std::lock_guard<std::mutex> lk(mtx);
            s_acc = 1.0f - total_err / (float)n_test_batches;
            screen.PostEvent(Event::Custom);
        }

        // Save CSV
        std::ofstream out_csv(loss_path);
        out_csv << "iter,loss\n";
        for (size_t i = 0; i < loss_history.size(); ++i)
            out_csv << i << "," << std::setprecision(9) << loss_history[i] << "\n";
        out_csv.close();

        {
            std::lock_guard<std::mutex> lk(mtx);
            s_done = true;
            screen.PostEvent(Event::Custom);
        }
        screen.ExitLoopClosure()();
    });

    screen.Loop(component);
    train_thread.join();

    {
        std::lock_guard<std::mutex> lk(mtx);
        if (s_done)
            cout << "Saved loss trajectory to: " << loss_path << "\n"
                 << "Total time: " << time_in_ms(t_start, Clock::now()) / 1000.0 << " s\n";
        else
            cout << "[stopped early at epoch=" << s_epoch << " step=" << s_global_step << " loss=" << s_loss << "]\n";
    }
    return 0;
}
