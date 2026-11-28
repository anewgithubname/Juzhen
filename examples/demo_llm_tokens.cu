/** Small token language model: embedding, positions, one block, tied output.
 * The synthetic cyclic token stream checks learning, not language quality.
 * JUZHEN_LLM_STEPS sets the number of updates (default 200);
 * JUZHEN_LLM_CHECKPOINT names a checkpoint file to resume from and save to.
 */
#include "../ml/embedding.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include "../ml/checkpoint.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
using namespace Juzhen;
#ifdef CUDA
using Backend = CUDAfloat;
#elif defined(ROCM_HIP)
using Backend = ROCMfloat;
#elif defined(APPLE_SILICON)
using Backend = MPSfloat;
#else
using Backend = float;
#endif

int compute() {
    try {
        global_rand_gen.seed(42);
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
        GPUSampler sampler(42);
#endif
        constexpr int vocab = 32, dim = 16, seq = 8, batch = 2, accumulation = 2;
        int steps = 200;
        if (const char* raw = std::getenv("JUZHEN_LLM_STEPS"))
            steps = std::atoi(raw);

        EmbeddingLayer<Backend> token(vocab, dim);
        PositionEmbeddingLayer<Backend> position(seq, dim);
        TransformerLayer<Backend> block(dim, dim, 4 * dim, seq, batch, 4, true);
        TiedLinearLayer<Backend> output(token);
        Sequential<Backend> network{&output, &block, &position, &token};

        AdamWOptions config;
        config.lr = .003f;
        config.weight_decay = .01f;
        AdamW<Backend> optimizer(network.parameters(), config);

        TrainingProgress progress;
        std::string error, path;
        if (const char* selected = std::getenv("JUZHEN_LLM_CHECKPOINT")) {
            path = selected;
            if (std::filesystem::exists(path) && !load_checkpoint(network.layers(), path, progress, &error))
                throw std::runtime_error("checkpoint load: " + error);
        }

        Matrix<int> x("x", seq, batch), y("y", seq, batch);  // one sequence per column
        float first = -1, last = -1;
        while (progress.step < uint64_t(steps)) {
            optimizer.zero_grad();
            double total = 0;
            for (int micro = 0; micro < accumulation; ++micro) {
                for (int b = 0; b < batch; ++b)
                    for (int t = 0; t < seq; ++t) {
                        const int id = int((progress.step * 11 + micro * 5 + b * 3 + t) % vocab);
                        x.elem(t, b) = id;
                        y.elem(t, b) = (id + 1) % vocab;
                    }
                auto [loss, dlogits] = softmax_cross_entropy(network.eval(x), y);
                network.backward(std::move(dlogits) / accumulation);
                total += item(loss) / accumulation;
            }
            optimizer.clip_grad_norm(1.0);
            optimizer.step();
            ++progress.step;
            last = float(total);
            if (first < 0) first = last;
            if (progress.step == 1 || progress.step % 25 == 0 || progress.step == uint64_t(steps))
                std::cout << "step=" << progress.step << " token_loss=" << last << '\n';
        }
        if (!path.empty() && !save_checkpoint(network.layers(), path, progress, &error))
            throw std::runtime_error("checkpoint save: " + error);
        std::cout << "RESULT first_loss=" << first << " final_loss=" << last << " completed_steps=" << progress.step
                  << '\n';
        if (steps >= 100 && first > 0 && !(last < first * .7f))
            throw std::runtime_error("synthetic token LM did not learn");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
