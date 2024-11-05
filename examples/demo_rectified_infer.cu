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

auto sample_X0(int n, int d)
{
    return randn(d, n);
}

int compute() {
    // spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif
    
    Profiler p("comp");
    using namespace Juzhen;
    std::string base = PROJECT_DIR;

    int d = 1;
    int n = 6500;

    auto X0 = sample_X0(n, d); // reference data
    plot_histogram(X0.data(), X0.num_row() * X0.num_col(), 23);

    const size_t batchsize = 1;
    
    // create a neural network
    // define layers
    ReluLayer<FLOAT> L0(333, d + 1, batchsize), L1(333, 333, batchsize),
        L2(333, 333, batchsize), L3(333, 333, batchsize);
    LinearLayer<FLOAT> L10(d, 333, batchsize);

    // nns are linked lists containing layers
    list<Layer<FLOAT> *> trainnn({&L10, &L3, &L2, &L1, &L0});
    
    loadweights(trainnn, base+"/net.weights");

    PrintSeparationLine();
    
    X0 = sample_X0(n, d); // reference data
    auto Zt = euler_integration(X0, trainnn, 100).back();

    plot_histogram(Zt.data(), Zt.num_row() * Zt.num_col(), 23);    
    
    return 0;
}