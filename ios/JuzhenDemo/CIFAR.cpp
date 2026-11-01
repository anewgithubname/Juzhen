// The CIFAR-10 rectified flow demo (examples/demo_cnn_rectified.cu) on this
// device: sampling images from a trained checkpoint, and a speed test of the
// demo's UNet, training step and Euler sampler fed random images instead of
// CIFAR-10 (the speed does not depend on the pixels).
//
// Speed test overrides: CIFAR_SPEED_BATCHES (e.g. "64,128"),
// CIFAR_SPEED_STEPS (timed training steps), CIFAR_SPEED_EULER (sampling steps).

#define compute juzhen_cnn_rectified
#include "../../examples/demo_cnn_rectified.cu"
#undef compute

#include <TargetConditionals.h>
#include <mach/mach.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

namespace {

// Memory iOS counts against the app (what it is killed for), in MB.
double footprint_mb() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.phys_footprint / 1048576.0;
}

// Memory the app may still allocate before iOS kills it, in MB (0: unknown).
double available_mb() {
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
    return os_proc_available_memory() / 1048576.0;
#else
    return 0;
#endif
}

double seconds_since(Clock::time_point t0) { return time_in_ms(t0, Clock::now()) / 1000.0; }

// Forward FLOPs of RectifiedUNet per image (convolutions as im2col GEMMs);
// a training step is counted as 3x forward.
constexpr double kForwardFlops = 1359e6;

// Times `steps` training steps (after 2 warm-up steps) and one Euler sampling
// run at batch size bs. Returns the memory the run added, in MB.
double run_batch(int bs, int steps, int euler_steps) {
    const int C = 3, H = 32, W = 32;
    const int d = C * H * W;
    const double base_mb = footprint_mb();

    RectifiedUNet net(bs);
    AdamWOptions options;
    options.lr = 2e-4f;
    options.weight_decay = 0;
    AdamW<FLOAT> optimizer(net.parameters(), options);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);

    Clock::time_point t0;
    for (int step = 0; step < 2 + steps; ++step) {
        if (step == 2) {
            mpsSynchronize();
            t0 = Clock::now();
        }

        // The body of the training loop in compute(), with random pixels in
        // [-1, 1] in place of a CIFAR-10 batch.
        Matrix<float> x1_host("x1", d, bs);
        for (int b = 0; b < bs; ++b)
            for (int p = 0; p < d; ++p)
                x1_host.elem(p, b) = 2.0f * udist(global_rand_gen) - 1.0f;
        Matrix<FLOAT> x1(x1_host);

        Matrix<float> t_host("t_host", 1, bs);
        for (int b = 0; b < bs; ++b)
            t_host.elem(0, b) = udist(global_rand_gen);
        Matrix<FLOAT> t_val(t_host);

        auto x0 = randn(d, bs);
        auto ones_d1 = ones(d, 1);
        auto one_minus_t = ones(1, bs) - t_val;
        auto x_t = hadmd(x0, ones_d1 * one_minus_t) + hadmd(x1, ones_d1 * t_val);
        auto target_v = x1 - x0;

        optimizer.zero_grad();
        const auto& pred_v = net.eval(x_t, t_host);
        auto diff = pred_v - target_v;
        float loss = item(sum(sum(square(Matrix<FLOAT>(diff)), 0), 1)) / (float)bs;
        (void)loss;
        auto grad = 2.0f * diff / (float)bs;
        net.backward(std::move(grad));
        optimizer.step();
    }
    mpsSynchronize();
    const double train_s = seconds_since(t0);
    const double step_s = train_s / steps;
    const double images_per_s = bs / step_s;
    const int steps_per_epoch = 50000 / bs;

    cout << "batch " << bs << ":" << endl;
    cout << fixed << setprecision(0)
         << "  training: " << step_s * 1000 << " ms/step, " << images_per_s << " images/s"
         << setprecision(2) << " (~" << images_per_s * 3 * kForwardFlops / 1e12 << " TFLOPS)" << endl;
    cout << setprecision(1)
         << "  1 epoch (" << steps_per_epoch << " steps): ~" << step_s * steps_per_epoch / 60
         << " min, 10 epochs: ~" << step_s * steps_per_epoch * 10 / 60 << " min" << endl;

    auto s0 = Clock::now();
    auto samples = ode_sample(net, bs, euler_steps);
    samples.to_host();
    cout << "  sampling " << bs << " images, " << euler_steps << " Euler steps: "
         << seconds_since(s0) << " s" << endl;

    const double used_mb = footprint_mb() - base_mb;
    cout << setprecision(0) << "  memory: +" << used_mb << " MB (app total " << footprint_mb() << " MB";
    if (available_mb() > 0) cout << ", " << available_mb() << " MB still available";
    cout << ")" << endl;
    return used_mb;
}

std::vector<int> batches_to_run() {
    std::vector<int> batches;
    std::stringstream ss(env_str("CIFAR_SPEED_BATCHES", "64,128"));
    for (std::string item; std::getline(ss, item, ',');)
        if (int bs = std::atoi(item.c_str()); bs > 0) batches.push_back(bs);
    return batches;
}

}  // namespace

int juzhen_cifar_speed() {
    const int steps = env_int("CIFAR_SPEED_STEPS", 20);
    const int euler_steps = env_int("CIFAR_SPEED_EULER", 100);
    global_rand_gen.seed(42);

    cout << "CIFAR-10 rectified flow (examples/demo_cnn_rectified.cu), random data" << endl;
    cout << "UNet 11->64->128->256, ~" << setprecision(2) << fixed << 3 * kForwardFlops / 1e9
         << " GFLOP per training image; " << steps << " timed steps after 2 warm-up" << endl << endl;

    double mb_per_image = 0;
    for (int bs : batches_to_run()) {
        // Skip a batch that the previous one says would not fit: iOS kills
        // the app instead of failing an allocation.
        const double need = mb_per_image * bs * 1.3;
        if (need > 0 && available_mb() > 0 && need > available_mb()) {
            cout << setprecision(0) << "batch " << bs << ": skipped, needs ~" << need << " MB, "
                 << available_mb() << " MB available" << endl;
            continue;
        }
        {
            // Release this batch size's buffers before the next one.
            Memory<int> mdi;
            Memory<float> md32;
            Memory<double> md64;
            Memory<MPSfloat> mpsmd;
            mb_per_image = run_batch(bs, steps, euler_steps) / bs;
            mpsSynchronize();
        }
        cout << endl;
    }
    return 0;
}

namespace {

// RGBA bytes of the samples laid out ncol images wide, with the pixel mapping
// of save_ppm() in the demo.
std::vector<uint8_t> image_grid(const Matrix<float>& samples, int ncol, int nrow) {
    const int C = 3, H = 32, W = 32;
    const int grid_w = ncol * W;
    std::vector<uint8_t> rgba((size_t)grid_w * nrow * H * 4, 255);
    const int n = std::min((int)samples.num_col(), ncol * nrow);
    for (int i = 0; i < n; ++i) {
        const int gr = i / ncol, gc = i % ncol;
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float v = samples.elem(c * H * W + y * W + x, i);
                    v = std::max(0.0f, std::min(1.0f, (v + 1.0f) * 0.5f));  // [-1,1] -> [0,1]
                    rgba[((size_t)(gr * H + y) * grid_w + gc * W + x) * 4 + c] = (uint8_t)(v * 255.0f);
                }
    }
    return rgba;
}

}  // namespace

bool juzhen_cifar_generate(const std::string& checkpoint, int count, int steps, bool heun, int every,
                           int* epochs,
                           const std::function<void(int, const std::vector<uint8_t>&, int, int)>& frame,
                           std::string* error) {
    RectifiedUNet net(count);
    TrainingProgress progress;
    if (!load_checkpoint(net.layers(), checkpoint, progress, error))
        return false;
    if (epochs) *epochs = (int)progress.epoch;

    const int ncol = (int)std::ceil(std::sqrt((double)count));
    const int nrow = (count + ncol - 1) / ncol;
    ode_sample(net, count, steps, heun, [&](int step, const Matrix<FLOAT>& Zs) {
        if (step % every == 0 || step == steps)
            frame(step, image_grid(to_host(Zs), ncol, nrow), ncol * 32, nrow * 32);
    });
    return true;
}
