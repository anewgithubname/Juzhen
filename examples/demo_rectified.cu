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

void writetocsv(const std::string& filename, const MatrixView<float>& Zt_host) {
    
    ofstream myfile;

    myfile.open(filename);
    for (int i = 0; i < Zt_host.num_col(); i++){
        for (int j = 0; j < Zt_host.num_row(); j++){
            myfile << Zt_host.elem(j, i) << ",";
        }
        myfile << std::endl;
    }
    myfile.close();

}

Matrix<FLOAT> euler_integration(const Matrix<FLOAT>& Z0, list<Layer<FLOAT>*>& trainnn, int steps) {
    // start euler integration
    std::cout << "start euler integration: " << std::endl;

    Profiler p("int");
    
    auto Zt = Z0;
    int n = Z0.num_col();

    float dt = 1.0f / steps;
    
    for (int i = 0; i < steps; i++){
        if (steps > 10 && i % (steps/10) == 0)
            std::cout << ".";
        
        float t = (float)i/steps;    
        auto inpt = vs({Zt, ones(1, n)*t});
        
        Zt += forward(trainnn, inpt) * dt;
    }

    std::cout << "done!" << std::endl;
    return Zt;
}

void PrintSeparationLine();


auto sample_X0(int n, int d) {
    return hs({randn(d, n/2)*.5 -2 , randn(d, n/2)*.5 + 2});
}

auto sample_X1(int n, int d){
    auto mu1 = vs({ones(1, n/2)*-2, ones(1, n/2)*2});
    auto mu2 = vs({ones(1, n/2)*2, ones(1, n/2)*-2});
    return hs({randn(d, n/2)*.5 + mu1 , randn(d, n/2)*.5 + mu2});
}

int compute() {
    // spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif
    
    Profiler p("comp");
    using namespace Juzhen;
    std::string base = PROJECT_DIR;

    int batchsize = 200;
    int d = 2;
    int n = 2000;

    auto X0 = sample_X0(n, d); // reference data
    auto X1 = sample_X1(n, d); // target data

    const size_t numbatches = X0.num_col() / batchsize;

    // create a neural network
    // define layers
    ReluLayer<FLOAT> L0(1024, d+1, batchsize), L1(1024, 1024, batchsize), 
                     L2(1024, 1024, batchsize), L3(1024, 1024, batchsize), 
                     L4(1024, 1024, batchsize), L5(1024, 1024, batchsize);
    LinearLayer<FLOAT> L6(d, 1024, batchsize);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L6, &L5, &L4, &L3, &L2, &L1, &L0 });
    
    for (int r = 0; r < 5; r++){
        // start the training loop
        for (int i = 0; i < numbatches * 500; i++) {
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
            LossLayer<FLOAT> L7(batchsize, Yt_i);
            trainnn.push_front(&L7);
            
            if (i % (25 * numbatches) == 0) {
                std::cout << "epoch: " << i / numbatches << " time: " << t << std::endl;
                float loss = item(forward(trainnn, inp_i));
                std::cout << "training loss: " << loss << std::endl;
            }else{
                forward(trainnn, inp_i);
            }
            
            backprop(trainnn, inp_i);
            trainnn.pop_front();

        }
        std::cout << std::endl;

        // dumpweights(trainnn, base+"/net.weights");
        // loadweights(trainnn, base+"/net.weights");

        X0 = sample_X0(n, d); // reference data
        auto Zt = euler_integration(X0, trainnn, 1000);

        X1 = Zt;
    }

    // loadweights(trainnn, base+"/net.weights");

    PrintSeparationLine();
    
    X0 = sample_X0(n, d); // reference data
    auto Zt = euler_integration(X0, trainnn, 1);

    PrintSeparationLine();
    auto mu_Zt =  (sum(Zt, 1) / Zt.num_col());
    std::cout << "mean of Zt: " << mu_Zt << std::endl;

    PrintSeparationLine();
    
    writetocsv(base + "/X0.csv", X0.to_host());
    writetocsv(base + "/X1.csv", X1.to_host());
    writetocsv(base + "/Zt.csv", Zt.to_host());

    return 0;
}