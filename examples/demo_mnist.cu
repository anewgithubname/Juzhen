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
#include "../cpp/juzhen.hpp"
#include <math.h>
#include <ctime>
#include <thread>

using namespace std;
using namespace Juzhen;

std::string getCPUInfo();
std::string getGPUInfo();
std::string getRAMInfo();

void send_computing_time(double elapsed){

    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    std::string CPU = getCPUInfo(), GPU = getGPUInfo(), MEM = getRAMInfo();
    std::string jsonData = "{\\\"cpu\\\":\\\"" + CPU + "\\\", \\\"gpu\\\":\\\"" + 
                GPU + "\\\", \\\"ram\\\":\\\"" + MEM + "\\\", \\\"time\\\":\\\"" + 
                std::to_string(elapsed) + "\\\"}";

    std::string server = BENCHMARKSERVER;
    // Prepare the curl command. Notice the escape of double quotes for Windows command line.
    std::string command = "curl -X POST -H \"Content-Type: application/json\" -d \"" 
                            + jsonData + "\" http://" + server +":5000/update";

    std::cout << command << std::endl;
    std::cout << "See your ranking at: http://" + server + ".com:8080" << std::endl;
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

// convert label Y matrix (1 X n) to one-hot encoding. 
Matrix<float> one_hot(const MatrixI& Y, int k) {
    Matrix<float> Y_one_hot("One_hot", k, Y.num_col());
    Y_one_hot.zeros();

    for (int i = 0; i < Y.num_col(); i++) {
        Y_one_hot.elem(Y.elem(0, i), i) = 1.0;
    }

    return Y_one_hot;
}

vector<Matrix<float>> mnist_dataset(){
    const int k = 10;
    std::string base = PROJECT_DIR + std::string("/datasets/MNIST");

    // check if *.matrix files exist
    FILE *fp = fopen((base + "/X.matrix").c_str(), "r");
    if (!fp) {
        // unzip dataset.zip to the folder 
        std::string command = "unzip " + base + "/dataset.zip -d " + base;
        int result = system(command.c_str());
        if (result != 0) {
            ERROR_OUT;
        }
    }


    auto X = read<float>(base + "/X.matrix"); 
    std::cout << "size of X: " << X.num_row() << " " << X.num_col() << std::endl;

    auto labels = read<int>(base +"/Y.matrix"); 
    std::cout << "size of labels: " << labels.num_row() << " " << labels.num_col() << std::endl;

    auto Y = one_hot(labels, k);
    std::cout << "size of Y: " << Y.num_row() << " " << Y.num_col() << std::endl;

    auto Xt = read<float>(base + "/T.matrix");
    std::cout << "size of Xt: " << Xt.num_row() << " " << Xt.num_col() << std::endl;

    auto labels_t = read<int>(base + "/YT.matrix"); 
    std::cout << "size of labels_t: " << labels_t.num_row() << " " << labels_t.num_col() << std::endl;

    auto Yt = one_hot(labels_t, k);
    std::cout << "size of Yt: " << Yt.num_row() << " " << Yt.num_col() << std::endl;

    return {X, Y, Xt, Yt};
}

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
    
    const int numbatches = X.num_col() / batchsize;

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

    // dumpweights(trainnn, std::string(PROJECT_DIR) + "/mnist.weights");

    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = time_in_ms(t1, t2);

    send_computing_time(elapsed);

    return 0;
}