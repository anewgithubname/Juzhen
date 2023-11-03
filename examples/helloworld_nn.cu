/**
 * @file helloworld_nn.cpp
 * @brief basic NN regression example
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

#include "../ml/layer.hpp"
#include "../cpp/juzhen.hpp"

using namespace std;
using namespace Juzhen;

#ifndef CPU_ONLY
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
#endif

int compute() {
#ifndef CPU_ONLY
    GPUSampler sampler(2345);
#endif

    // problem set up
    const int n = 5000, d = 10, batchsize = 50, numbatches = n / batchsize;

    // regression dataset generation
    auto X = randn(d, n), beta = randn(d, 1), Y = beta.T() * X + .5*randn(1, n);

    auto XT = randn(d, n), YT = beta.T() * XT + .5*randn(1, n);

    // define layers
    Layer<FLOAT> L0(16, d, batchsize), L1(4, 16, batchsize);
    LinearLayer<FLOAT> L2(1, 4, batchsize);
    // least sqaure loss
    LossLayer<FLOAT> L3t(n, YT);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L2, &L1, &L0 }), testnn({ &L3t, &L2, &L1, &L0});

    // sgd
    int iter = 0;
    while (iter < 10000) {
        int batch_id = (iter % numbatches);

        // obtaining batches
#ifndef CPU_ONLY
        auto X_i = Matrix<FLOAT>(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
        auto Y_i = Matrix<FLOAT>(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
        auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
        auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif
        // forward-backward pass
        forward(trainnn, X_i);
        LossLayer<FLOAT> L3(batchsize, Y_i);
        trainnn.push_front(&L3);
        backprop(trainnn, X_i);
        trainnn.pop_front();
        if (iter % 1000 == 0) {
#ifndef CPU_ONLY
            cout << "testing loss: " << forward(testnn, XT).to_host().elem(0, 0) << endl;
#else
            cout << "testing loss: " << forward(testnn, XT).elem(0, 0) << endl;
#endif
        }

        iter++;
    }
    return 0;
}