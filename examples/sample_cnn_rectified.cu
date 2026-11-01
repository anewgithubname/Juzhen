/**
 * @file sample_cnn_rectified.cu
 * @brief Draw CIFAR-10 samples from a demo_cnn_rectified checkpoint for FID.
 *
 * Writes the samples in the format examples/compute_fid.py reads:
 *   CKPT=res/generated_cifar10/unet.ckpt OUT=samples.bin ./sample_cnn_rectified
 *   python examples/compute_fid.py samples.bin
 *
 * Environment variables:
 *   CKPT      checkpoint saved by demo_cnn_rectified (required)
 *   OUT       samples file to write                  (required)
 *   N         number of samples                      (default: 10000)
 *   BS        batch size                             (default: 500)
 *   STEPS     ODE steps                              (default: 100)
 *   SAMPLER   euler|heun                             (default: euler)
 *   SEED      random seed                            (default: 7)
 */

#define compute juzhen_cnn_rectified
#include "demo_cnn_rectified.cu"
#undef compute

int compute() {
    const std::string ckpt = env_str("CKPT", ""), out = env_str("OUT", "");
    if (ckpt.empty() || out.empty()) {
        cerr << "set CKPT (checkpoint to load) and OUT (samples file to write)" << endl;
        return 1;
    }
    const int n = env_int("N", 10000), bs = env_int("BS", 500), steps = env_int("STEPS", 100);
    const bool heun = env_str("SAMPLER", "euler") == "heun";
    const int seed = env_int("SEED", 7);
#ifdef CUDA
    GPUSampler sampler(seed);
#endif
    global_rand_gen.seed(seed);

    RectifiedUNet net(bs);
    TrainingProgress progress;
    std::string error;
    if (!load_checkpoint(net.layers(), ckpt, progress, &error)) {
        cerr << "cannot load " << ckpt << ": " << error << endl;
        return 1;
    }

    const int d = 3 * 32 * 32;
    Matrix<float> all("all", d, n);
    auto t0 = Clock::now();
    for (int filled = 0; filled < n; filled += bs) {
        auto Zh = to_host(ode_sample(net, bs, steps, heun));
        for (int i = 0; i < std::min(bs, n - filled); ++i)
            for (int p = 0; p < d; ++p) all.elem(p, filled + i) = Zh.elem(p, i);
    }
    save_raw_samples(out, all);
    cout << "epoch " << progress.epoch << ": " << n << " samples, " << steps << (heun ? " Heun" : " Euler")
         << " steps in " << time_in_ms(t0, Clock::now()) / 1000.0 << " s -> " << out << endl;
    return 0;
}
