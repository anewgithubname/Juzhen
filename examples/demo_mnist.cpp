/**
 * @file main_cpp.cpp
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
#define CPU_ONLY
#define LOGGING_ON

#include "../cpp/layer.hpp"
#include "../cpp/juzhen.hpp"
#include <math.h>
#include <ctime>
#include <thread>

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

// convert label Y matrix (1 X n) to one-hot encoding. 
Matrix<float> one_hot(const MatrixI& Y, int k) {
    Matrix<float> Y_one_hot("One_hot", k, Y.num_col());
    Y_one_hot.zeros();

    for (int i = 0; i < Y.num_col(); i++) {
        Y_one_hot.elem(Y.elem(0, i), i) = 1.0;
}

    return Y_one_hot;
}

class ComputeCode
{
    const int d = 28*28, n = 60000, batchsize = 30, numbatches = n / batchsize, nt = 10000;
    const int max_iter = numbatches * 5;

    const int k = 10;
    Matrix <float> X, Y; 
    MatrixF *pXt, *pYt;
    std::list<Juzhen::Layer<MatrixF>*> trainnn, testnn;
    Juzhen::LinearLayer<MatrixF> L2;
    Juzhen::Layer<MatrixF> L0, L1;
    Juzhen::LogisticLayer<MatrixF> *L3t;

    bool compute_done = false;
    std::thread* thread = NULL;
public:
    ComputeCode() :
        L2(k, 128, batchsize),
        L1(128, 1024, batchsize),
        //L02(1024, d * 4, batchsize),
        //L01(d * 4, d * 10, batchsize),
        L0(1024, d, batchsize),
        //preparing dataset
        X("X", d, n),
        Y("Y", k, n)    
    {
        std::string base = "./";
        X.read(base + "X.matrix"); 
        std::cout << "size of X: " << X.num_row() << " " << X.num_col() << std::endl;

        MatrixI labels("labels", 1, n);
        labels.read(base +"Y.matrix"); 
        std::cout << "size of labels: " << labels.num_row() << " " << labels.num_col() << std::endl;

        Y = one_hot(labels, k);
        std::cout << "size of Y: " << Y.num_row() << " " << Y.num_col() << std::endl;

        Matrix<float> Xt("Xt", d, nt);  Xt.read(base + "T.matrix");
        std::cout << "size of Xt: " << Xt.num_row() << " " << Xt.num_col() << std::endl;

        MatrixI labels_t("labels_t", 1, nt);
        labels_t.read(base + "YT.matrix"); 
        std::cout << "size of labels_t: " << labels_t.num_row() << " " << labels_t.num_col() << std::endl;

        Matrix<float> Yt("Yt", k, nt);  Yt = one_hot(labels_t, k);
        std::cout << "size of Y: " << Yt.num_row() << " " << Yt.num_col() << std::endl;
        
        pXt = new MatrixF(Xt);
        pYt = new MatrixF(Yt);
        L3t = new Juzhen::LogisticLayer<MatrixF>(nt, *pYt);
    }

    virtual void OnAttach() {
        trainnn.push_back(&L2);
        trainnn.push_back(&L1);
        //trainnn.push_back(&L02);
        //trainnn.push_back(&L01);
        trainnn.push_back(&L0);

        testnn.push_back(L3t);
        testnn.push_back(&L2);
        testnn.push_back(&L1);
        //testnn.push_back(&L02);
        //testnn.push_back(&L01);
        testnn.push_back(&L0);
        
        thread = new std::thread(&ComputeCode::compute, this);
    }
    virtual void OnUIRender()
    {
    }

    virtual void OnDetach() {
        //free cublas
        compute_done = true;
         thread->join();
         delete thread;
    }

//protected:
    void compute() {
        TIC;
        using namespace std;

        static int iter = 0;
        Clock::time_point t = Clock::now();
        while (!compute_done && iter <= max_iter) {

            int batch_id = (iter % numbatches);

#ifndef CPU_ONLY
            auto X_i = cuMatrix(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
            auto Y_i = cuMatrix(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
            auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
            auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif
            Juzhen::forward(trainnn, X_i);
            Juzhen::LogisticLayer<MatrixF> L3(batchsize, Y_i);
            trainnn.push_front(&L3);
            Juzhen::backprop(trainnn, X_i);
            trainnn.pop_front();

            if (iter % numbatches == 0)
            {
                cout << "iteration: " << iter << ", testing error: "
#ifndef CPU_ONLY
                    //<< Juzhen::forward(testnn, *pXt).to_host().elem(0, 0) << endl;
                    ;
#else
                    //<< Juzhen::forward(testnn, *pXt).elem(0, 0) << endl;
                    ;
#endif
                cout << "duration for one epoch " << time_in_ms(t, Clock::now()) << "ms. " << endl;
                t = Clock::now();
            }

            iter++;
        }

        cout << Juzhen::forward(testnn, *pXt).elem(0, 0) << endl;
        LOG_INFO("Computing done. Close the plot window to quit.");
        TOC;
    }
};

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

    ComputeCode c;
    c.OnAttach();
    //c.compute();
     std::cout << "Enter to quit." << std::endl;
     getchar();
    c.OnUIRender();
    c.OnDetach();

#ifndef CPU_ONLY
    cublasDestroy(cuMatrix::global_handle);
    LOG_INFO("CuBLAS FREED!");
#endif
    return 0;
}