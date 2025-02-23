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
#include <thread>

using namespace std;
using namespace Juzhen;

std::string getCPUInfo();
std::string getGPUInfo();
std::string getRAMInfo();

void send_computing_time(double elapsed){

    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    std::string CPU = getCPUInfo(), GPU = getGPUInfo(), MEM = getRAMInfo();
    std::string jsonData = R"({\"cpu\":\")" + CPU + R"(\", \"gpu\":\")" +
                GPU + R"(\", \"ram\":\")" + MEM + R"(\", \"time\":\")" +
                std::to_string(elapsed) + "\\\"}";

    std::string server = BENCHMARKSERVER;
    // Prepare the curl command. Notice the escape of double quotes for Windows command line.
    std::string command = R"(curl -X POST -H "Content-Type: application/json" -d ")"
                            + jsonData + "\" http://" + server +":5000/update";

    std::cout << command << std::endl;
    std::cout << "See your ranking at: http://" + server + ":8080" << std::endl;
    // Execute the curl command using system()
    int result = system(command.c_str());

    // Check the result of the system call
    if (result != 0) {
        // Handle the error case
        ERROR_OUT;
    }
}

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
    auto t1 = std::chrono::high_resolution_clock::now();
    // spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif
    const int d = 28*28, k = 10, batchsize = 32;
    auto vecXY = mnist_dataset();
    auto X = vecXY[0]; 
    auto Y = vecXY[1];
    
    const size_t numbatches = X.num_col() / batchsize;

#ifndef CPU_ONLY
    auto XT = Matrix<CUDAfloat>(vecXY[2]);
#else
    auto &XT = vecXY[2];
#endif

#ifndef CPU_ONLY
    auto YT = Matrix<CUDAfloat>(vecXY[3]);
#else
    auto &YT = vecXY[3];
#endif

    // define layers
    ReluLayer<FLOAT> L0(1024, d, batchsize), L1(128, 1024, batchsize);
    LinearLayer<FLOAT> L2(k, 128, batchsize);
    // logistic loss
    ZeroOneLayer<FLOAT> L3t(XT.num_col(), YT);

    // nns are linked lists containing layers
    list<Layer<FLOAT>*> trainnn({ &L2, &L1, &L0 }), testnn({ &L3t, &L2, &L1, &L0 });

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
#ifndef CPU_ONLY
        auto X_i = Matrix<FLOAT>(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
        auto Y_i = Matrix<FLOAT>(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
        auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
        auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif

        // forward-backward pass
        forward(trainnn, X_i);
        LogisticLayer<FLOAT> L3(batchsize, std::move(Y_i));
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

    // dumpweights(trainnn, std::string(PROJECT_DIR) + "/mnist.weights");

    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = time_in_ms(t1, t2);

    send_computing_time(elapsed);

    return 0;
}