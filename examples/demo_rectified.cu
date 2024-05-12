/**
 * @file demo_denoising.cu
 * @brief denoising MNIST imaages
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This is an implementation of the rectified flow
 * https://arxiv.org/abs/2209.03003
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

#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"
#include "../ml/dataloader.hpp"
#include <fstream>

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

int compute() {
    // spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif

    using namespace Juzhen;

    std::string base = PROJECT_DIR;
    int batchsize = 100;
    int d = 20;
    
    auto X0 = randn(d, 2000);
    auto X1 = randn(d, 2000) * .25 - 5.3;

    const size_t numbatches = X0.num_col() / batchsize;

    //create a neural network
    // define layers
    Layer<FLOAT> L0(32, d+1, batchsize), L1(32, 32, batchsize), L2(32, 32, batchsize);
    LinearLayer<FLOAT> L6(d, 32, batchsize);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L6, &L2, &L1, &L0 });
       
    // loop over all images
    for (int i = 0; i < numbatches * 100; i++) {
        size_t batch_id = i % numbatches;
        
        // obtain batch
        auto X0_i = X0.columns(batchsize * batch_id, batchsize * (batch_id + 1)); 
        auto X1_i = X1.columns(batchsize * batch_id, batchsize * (batch_id + 1));

        // sample time uniformly from [0, 1]
        float t = (float)rand() / RAND_MAX;
        
        //compute the interpolation between X0i and X1i
        auto Xt_i = X0_i * (1 - t) + X1_i * t;
        auto inp_i = vstack({Xt_i, ones(1, batchsize)*t});
        auto Yt_i = X1_i - X0_i;

        // forward-backward pass
        LossLayer<FLOAT> L7(batchsize, Yt_i);
        trainnn.push_front(&L7);
        
        if (i % (25 * numbatches) == 0) {
            std::cout << "epoch: " << i / numbatches << " time: " << t << std::endl;
            float loss = forward(trainnn, inp_i).to_host().elem(0, 0);
            std::cout << "training loss: " << loss << std::endl;
        }else{
            forward(trainnn, inp_i);
        }
        
        backprop(trainnn, inp_i);
        trainnn.pop_front();

    }
    std::cout << std::endl;

    // start euler integration
    auto Zt = randn(d, 20000);

    int steps = 1000;
    float dt = 1.0 / steps;
    
    for (int i = 0; i < steps; i++){
        float t = (float)i/steps;    
        auto inpt = vstack({Zt, ones(1, 20000)*t});
        
        auto u = forward(trainnn, inpt);
        Zt += u * dt;
    }

    std::cout << "mean of Zt: " << (sum(Zt, 1) / Zt.num_col()).to_host() << std::endl;

    return 0;
}