/**
 * @file demo_mnist.cu
 * @brief MNIST multiclass logistic classifier
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
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

// #define CPU_ONLY

#include "../ml/layer.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include <thread>

using namespace std;
using namespace Juzhen;

std::string getCPUInfo();
std::string getGPUInfo();
std::string getRAMInfo();

#if defined(CUDA)
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
#elif defined(ROCM_HIP)
#define FLOAT ROCMfloat
inline Matrix<ROCMfloat> randn(int m, int n) { return Matrix<ROCMfloat>::randn(m, n); }
inline Matrix<ROCMfloat> ones(int m, int n) { return Matrix<ROCMfloat>::ones(m, n); }
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
#endif
#define MatrixI Matrix<int> 

int compute() {
    auto t1 = std::chrono::high_resolution_clock::now();
    // spdlog::set_level(spdlog::level::debug);
#ifdef CUDA
    GPUSampler sampler(1);
#endif
#ifdef ROCM_HIP
    global_rand_gen.seed(1);
#endif
    const int d = 28*28, k = 10, batchsize = 32;
    auto vecXY = mnist_dataset();
    auto X = vecXY[0]; 
    auto Y = vecXY[1];
    
    const size_t numbatches = X.num_col() / batchsize;

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
    auto XT = Matrix<FLOAT>(vecXY[2]);
#else
    auto &XT = vecXY[2];
#endif

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
    auto YT = Matrix<FLOAT>(vecXY[3]);
#else
    auto &YT = vecXY[3];
#endif

    // define layers
    Layer<FLOAT> L0(1024, d, batchsize), L1(128, 1024, batchsize);
    LinearLayer<FLOAT> L2(k, 128, batchsize);

    // nns are linked lists containing layers, output layer first
    list<Layer<FLOAT>*> trainnn({ &L2, &L1, &L0 });
    Sequential<FLOAT> network(trainnn);
    AdamWOptions options;
    options.weight_decay = 0; // Preserve the original Adam update.
    AdamW<FLOAT> optimizer(network.parameters(), options);

    // //if file exists, load weights
    // FILE *fp = fopen((std::string(PROJECT_DIR) + "/mnist.weights").c_str(), "r");
    // if (fp) {
    //     fclose(fp);
    //     loadweights(trainnn, std::string(PROJECT_DIR) + "/mnist.weights");
    // }

    // sgd
    int iter = 0;
    while (iter < 10000) {
        size_t batch_id = (iter % numbatches);

        // obtaining batches
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
        auto X_i = Matrix<FLOAT>(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
        auto Y_i = Matrix<FLOAT>(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
        auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
        auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif

        // Compute gradients, then update all parameters together.
        optimizer.zero_grad();
        const auto& prediction = network.eval(X_i);
        network.backward(softmax_cross_entropy(prediction, Y_i).grad);
        optimizer.step();
        if (iter % 1000 == 0) {
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            cout << "Misclassification Rate: " << error_rate(forward(trainnn, XT), YT).to_host().elem(0, 0) << endl;
#else
            cout << "Misclassification Rate: " << error_rate(forward(trainnn, XT), YT).elem(0, 0) << endl;
#endif
        }

        iter++;
    }

    // dumpweights(trainnn, std::string(PROJECT_DIR) + "/mnist.weights");

    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = time_in_ms(t1, t2);

    return 0;
}