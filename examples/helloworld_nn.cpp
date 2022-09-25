/**
 * @file helloworld_nn.cpp
 * @brief MNIST implementation
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

#include "../cpp/layer.hpp"
#include "../cpp/juzhen.hpp"
#include <math.h>
#include <ctime>
#include <thread>

using namespace std;

#ifndef CPU_ONLY
typedef cuMatrix MatrixF;
inline cuMatrix randn(int m, int n) { return cuMatrix::randn(m, n); }
inline cuMatrix ones(int m, int n) { return cuMatrix::ones(m, n); }
#else
typedef Matrix<float> MatrixF;
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
#endif
#define MatrixI Matrix<int> 

void compute() {
    // problem set up
    const int n = 5000, d = 10, batchsize = 50, numbatches = n / batchsize;

    // regression dataset generation
    auto X = randn(10, n);
    auto beta = randn(10, 1);
    auto Y = beta.T() * X + randn(1, n);

    auto XT = randn(10, n);
    auto YT = beta.T() * XT + randn(1, n);

    // define layers
    Juzhen::Layer<MatrixF> L0(16, 10, batchsize), L1(4, 16, batchsize);
    Juzhen::LinearLayer<MatrixF> L2(1, 4, batchsize);
    // least sqaure loss
    Juzhen::LossLayer<MatrixF> L3t(n, YT);

    // nns are linked lists containing layers
    std::list<Juzhen::Layer<MatrixF>*> trainnn({ &L2, &L1, &L0 }), testnn({ &L3t, &L2, &L1, &L0});

    // sgd
    static int iter = 0;
    while (iter < 10000) {
        int batch_id = (iter % numbatches);

        // obtaining batches
#ifndef CPU_ONLY
        auto X_i = cuMatrix(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
        auto Y_i = cuMatrix(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
        auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
        auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif
        // forward-backward pass
        Juzhen::forward(trainnn, X_i);
        Juzhen::LossLayer<MatrixF> L3(batchsize, Y_i);
        trainnn.push_front(&L3);
        Juzhen::backprop(trainnn, X_i);
        trainnn.pop_front();
        if (iter % 1000 == 0) {
#ifndef CPU_ONLY
            cout << "testing loss: " << Juzhen::forward(testnn, XT).to_host().elem(0, 0) << endl;
#else
            cout << "testing loss: " << Juzhen::forward(testnn, XT).elem(0, 0) << endl;
#endif
        }

        iter++;
    }
}

int main() {
    MemoryDeleter<int> md2;
    MemoryDeleter<float> md;

#ifndef LOGGING_OFF
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("%^[%l]%$, %v");
#endif

#ifndef CPU_ONLY
    GPUMemoryDeleter gpumd;
    GPUSampler sampler(1);
    cublasCreate(&cuMatrix::global_handle);
    LOG_INFO("CuBLAS INITIALIZED!");
#endif
  
    compute();

#ifndef CPU_ONLY
    cublasDestroy(cuMatrix::global_handle);
    LOG_INFO("CuBLAS FREED!");
#endif
    return 0;
}