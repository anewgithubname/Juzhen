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
#include <sstream>
#include <fstream>

using namespace std;
using namespace Juzhen;

#ifndef CPU_ONLY
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
inline Matrix<CUDAfloat> vs(std::vector<MatrixView<CUDAfloat>> matrices) {return vstack(matrices);}
inline Matrix<CUDAfloat> hs(std::vector<MatrixView<CUDAfloat>> matrices) {return hstack(matrices);}
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
inline Matrix<float> vs(std::vector<MatrixView<float>> matrices) {return vstack<float>(matrices);}
inline Matrix<float> hs(std::vector<MatrixView<float>> matrices) {return hstack<float>(matrices);}
#endif
#define MatrixI Matrix<int> 

void PrintSeparationLine();

auto sample_X0(int n, int d) {
    auto vecMNIST = mnist_dataset();
    auto X = hstack<float>({vecMNIST[0], vecMNIST[2]});
    auto Y = hstack<float>({vecMNIST[1], vecMNIST[3]});
    // selecting only digit 6s
    
    int digit = 6;

    int num1 = sum(Y, 1).elem(digit, 0);
    std::cout << "num1: " << num1 << std::endl;
    auto X0 = Matrix<float>::zeros(d, num1);

    int j = 0;
    for (int i = 0; i < Y.num_col(); i++){
        if (Y.elem(digit, i) == 1){
            X0.columns(j, j+1, X.columns(i, i+1));
            j++;
        }
    }
    
    return Matrix<CUDAfloat>(X0.columns(0, n));
}

auto sample_X1(int n, int d){
    auto vecMNIST = mnist_dataset();
    auto X = hstack<float>({vecMNIST[0], vecMNIST[2]});
    auto Y = hstack<float>({vecMNIST[1], vecMNIST[3]});
    
    // selecting only digit 7s
    int digit1 = 7;
    int num7 = sum(Y, 1).elem(digit1, 0);

    auto X7 = Matrix<float>::zeros(d, num7);
    std::cout << "num7: " << num7 << std::endl;

    int j = 0;
    for (int i = 0; i < Y.num_col(); i++){
        if (Y.elem(digit1, i) == 1){
            X7.columns(j, j+1, X.columns(i, i+1));
            j++;
        }
    }
    return Matrix<CUDAfloat>(X7.columns(0, n));
}

int compute() {
    // spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif
    
    Profiler p("comp");
    using namespace Juzhen;
    std::string base = PROJECT_DIR;

    int batchsize = 100;
    int d = 28*28;
    int n = 6500;

    auto X0 = sample_X0(n, d); // reference data
    writetocsv<float>(base + "/X0.csv", X0.to_host());
    auto X1 = sample_X1(n, d); // target data
    writetocsv<float>(base + "/X1.csv", X1.to_host());

    const size_t numbatches = X0.num_col() / batchsize;

    // create a neural network
    // define layers
    ReluLayer<FLOAT> L0(2048, d+1, batchsize), L1(2048, 2048, batchsize), 
                     L2(2048, 2048, batchsize), L3(2048, 2048, batchsize), 
                     L4(2048, 2048, batchsize), L5(2048, 2048, batchsize);
    LinearLayer<FLOAT> L10(d, 2048, batchsize);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L10, &L5, &L4, &L3, &L2, &L1, &L0 });
    
    for (int r = 0; r < 1; r++){
        // start the training loop
        for (int i = 0; i < numbatches * 50000; i++) {
            size_t batch_id = i % numbatches;
            
            // obtain batch
            auto X0_i = X0.columns(batchsize * batch_id, batchsize * (batch_id + 1)); 
            auto X1_i = X1.columns(batchsize * batch_id, batchsize * (batch_id + 1));

            // sample time uniformly from [0, 1]
            float t = (float)rand() / RAND_MAX;
            
            //compute the interpolation between X0i and X1i
            auto Xt_i = X0_i * (1 - t) + X1_i * t;
            // add time to the input
            auto inp_i = vs({Xt_i, ones(1, batchsize)*t});
            auto Yt_i = X1_i - X0_i;

            // forward-backward pass
            LossLayer<FLOAT> L11(batchsize, Yt_i);
            trainnn.push_front(&L11);
            
            if (i % (25 * numbatches) == 0) {
                std::cout << "epoch: " << i / numbatches << " time: " << t << std::endl;
                float loss = item(forward(trainnn, inp_i));
                std::cout << "training loss: " << loss << std::endl;
                dumpweights(trainnn, base+"/net.weights");
            }else{
                forward(trainnn, inp_i);
            }
            
            backprop(trainnn, inp_i);
            trainnn.pop_front();

        }
        std::cout << std::endl;

        X0 = sample_X0(n, d); // reference data
        auto Zt = euler_integration(X0, trainnn, 51);

        X1 = Zt;
    }

    PrintSeparationLine();
    
    X0 = sample_X0(n, d); // reference data
    auto Zt = euler_integration(X0, trainnn, 51);

    PrintSeparationLine();
    auto mu_Zt =  (sum(Zt, 1) / Zt.num_col());
    std::cout << "mean of Zt: " << mu_Zt << std::endl;

    PrintSeparationLine();
    
    writetocsv<float>(base + "/X0.csv", X0.to_host());
    writetocsv<float>(base + "/X1.csv", X1.to_host());
    writetocsv<float>(base + "/Zt.csv", Zt.to_host());

    dumpweights(trainnn, base+"/net.weights");
    
    return 0;
}