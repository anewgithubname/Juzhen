/**
 * @file demo_classification.cu
 * @brief basic logisitc classification on synthetic data
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
#include "../cpp/juzhen.hpp"
#include <math.h>
#include <ctime>
#include <thread>

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
#define MatrixI Matrix<int> 

vector<Matrix<float>> dataset(int n, int d){

    // regression dataset generation
    auto X = hstack<float>({Matrix<float>::randn(d, n/2)-2, Matrix<float>::randn(d, n/2) + 2});

    Matrix<float> Y("One_hot", 2, X.num_col());
    Y.zeros();

    for (int i = 0; i < Y.num_col(); i++) {
        if (i < n/2)
            Y.elem(0, i) = 1;
        else
            Y.elem(1, i) = 1;
    }

    cout << "X: " << X.num_row() << ", " <<  X.num_col() << endl;
    cout << "Y: " << Y.num_row() << ", " <<  Y.num_col() << endl;

    idxlist idx = shuffle(n);
    X = X.columns(idx); 
    Y = Y.columns(idx);

    return {X, Y};
}

int compute() {
    //spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(2345);
#endif
    const int n = 50000, d = 2, k = 2, batchsize = 500, numbatches = n / batchsize;
    auto vecXY = dataset(n, d);
    auto X = vecXY[0]; 
    auto Y = vecXY[1];

    auto vecXtYt = dataset(n, d);
#ifndef CPU_ONLY
    auto XT = Matrix<CUDAfloat>(vecXtYt[0]);
#else
    auto &XT = vecXtYt[0];
#endif

#ifndef CPU_ONLY
    auto YT = Matrix<CUDAfloat>(vecXtYt[1]);
#else
    auto &YT = vecXtYt[1];
#endif

    // define layers
    ReluLayer<FLOAT> L0(4096, d, batchsize), L1(128, 4096, batchsize);
    LinearLayer<FLOAT> L2(k, 128, batchsize);
    // least sqaure loss
    ZeroOneLayer<FLOAT> L3t(XT.num_col(), YT);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L2, &L1, &L0 }), testnn({ &L3t, &L2, &L1, &L0 });

    // sgd
    int iter = 0;
    while (iter < 2000+1) {
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
        LogisticLayer<FLOAT> L3(batchsize, Y_i);
        trainnn.push_front(&L3);

        backprop(trainnn, X_i);
        trainnn.pop_front();
        if (iter % 1000 == 0) {
#ifndef CPU_ONLY
            cout << "Misclassification Rate: " << forward(testnn, XT).to_host().elem(0, 0) << endl;
#else
            cout << "Misclassification Rate: " << forward(testnn, XT).elem(0, 0) << endl;
#endif
        }

        iter++;
    }

    dumpweights(trainnn, std::string(PROJECT_DIR) + "/classify.weights");
    return 0;
}