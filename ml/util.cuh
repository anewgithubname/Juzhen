/**
 * @file cutil.cuh
 * @brief Miscellaneous utility functions for Machine Learning
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2023 Song Liu (song.liu@bristol.ac.uk)

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

#pragma once

#include "../cpp/juzhen.hpp"

#ifdef CUDA
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

#ifdef CUDA
#define dvector thrust::device_vector
#else
#define dvector std::vector
#endif

template <class T>
Matrix<T> comp_dist(const Matrix<T> &a, const Matrix<T> &b) {
    return sum(square(a), 1) * Matrix<T>::ones(1, b.num_row()) +
           Matrix<T>::ones(a.num_row(), 1) * sum(square(b), 1).T() -
           2 * a * b.T();
}

template <class T>
float comp_med(const Matrix<T> &a) {
    STATIC_TIC;
    size_t n = a.num_row() * a.num_row();
#ifdef CUDA
    const float *s = (float *)comp_dist(a, a).data();
    thrust::device_vector<float> vec(s, s + n);
    thrust::sort(vec.begin(), vec.end());
    STATIC_TOC;
    return sqrt(.5*vec[n / 2]);
#else
    float *s = (float *)comp_dist(a, a).data();
    std::sort(s, s + n);
    STATIC_TOC;
    return sqrt(.5*s[n / 2]);
#endif
}

template <class T>
Matrix<T> kernel_gau(Matrix<T> &&b, float sigma) {
    return exp(-b / (2 * sigma * sigma));
}

template <class T>
Matrix<T> relu(Matrix<T> &&M) {
    return elemwise([=] __GPU_CPU__(float x) { return x > 0.0 ? x : 0.0; }, M);
}

template <class T>
Matrix<T> d_relu(Matrix<T> &&M) {
    return elemwise([=] __GPU_CPU__(float x) { return x > 0.0 ? 1.0 : 0.0; }, M);
}

template <class T>
inline int argmin(std::vector<T> a) {
    // replace all nan with inf
    std::replace_if(a.begin(), a.end(), [](T x) { return std::isnan(x); },
                    std::numeric_limits<T>::infinity());
    return std::min_element(a.begin(), a.end()) - a.begin();
}

template <class T>
inline int argmax(std::vector<T> a) {
    // replace all nan with -inf
    std::replace_if(a.begin(), a.end(), [](T x) { return std::isnan(x); },
                    -std::numeric_limits<T>::infinity());
    return std::max_element(a.begin(), a.end()) - a.begin();
}

template <class T>
inline float item(const Matrix<T> &M){
    // assert(M.num_row() == 1 && M.num_col() == 1);
    #if !defined(CUDA) || !defined(APPLE_SILIICON)
        return M.elem(0, 0);
    #else
        return M.to_host().elem(0, 0);
    #endif
}

#define sqrtM(b) elemwise([=] __GPU_CPU__(float x) { return sqrt(x); }, b)

template <class T>
struct adam_state{
    int iteration;
    float alpha, beta1, beta2, eps;
    Matrix<T> m, v;
    adam_state(const Matrix<T> &theta)
        :iteration(1), alpha(0.01), beta1(0.9), beta2(0.999), eps(1e-8){
        m = zeros_like(theta);
        v = zeros_like(theta);
    }
    adam_state(double alpha, size_t nrow, size_t ncol)
        :iteration(1), alpha(alpha), beta1(0.9), beta2(0.999), eps(1e-8){
        m = Matrix<T>::zeros(nrow, ncol);
        v = Matrix<T>::zeros(nrow, ncol);
    }

    void print_stats(){
        std::cout << "iteration: " << iteration << " alpha: " << alpha << " beta1: " << beta1 << " beta2: " << beta2 << " eps: " << eps << std::endl << "m: " << m.norm() << std::endl << "v: " << v.norm() << std::endl;
	}
};

template <class T>
Matrix<T> adam_update(Matrix<T> &&g, adam_state<T> &state){
    int &iteration = state.iteration;
    float &alpha = state.alpha, &beta1 = state.beta1, &beta2 = state.beta2, &eps = state.eps;
    Matrix<T> &m = state.m, &v = state.v;

    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * square(g);

    Matrix<T> m_hat = m / (1 - pow(beta1, iteration));
    Matrix<T> v_hat = v / (1 - pow(beta2, iteration));

    g = alpha * m_hat / (sqrtM(v_hat) + eps);
    iteration++;
    return std::move(g);
}

template <class T>
Matrix<T> predict_one_hot(const Matrix<T>& input){
    //finding the maximum of input for each column
    auto colmax = reduce(

        [] __GPU_CPU__(float *v, float *vdes, int lenv, int lendes){
            double max = - DBL_MAX;
            for(int i = 0; i < lenv; i++){
                if(v[i] > max){
                    max = v[i];
                }
            }
            vdes[0] = max;
        },
        
        input, 0, 1);

    //substraction of the maximum from each column
    auto sub = input - Matrix<T>::ones(input.num_row(), 1) * colmax;
    //converting it to zero one matrix
    return elemwise([] __GPU_CPU__ (float x){return x > -1e-5 ? 1 : 0;}, std::move(sub));
}

// convert label Y matrix (1 X n) to one-hot encoding. 
Matrix<float> one_hot(const Matrix<int>& Y, int k) {
    Matrix<float> Y_one_hot("One_hot", k, Y.num_col());
    Y_one_hot.zeros();

    for (int i = 0; i < Y.num_col(); i++) {
        Y_one_hot.elem(Y.elem(0, i), i) = 1.0;
    }

    return Y_one_hot;
}

std::vector<Matrix<float>> mnist_dataset(){
    const int k = 10;
    std::string base = PROJECT_DIR + std::string("/datasets/MNIST");

    // check if *.matrix files exist
    FILE *fp = fopen((base + "/train_x.matrix").c_str(), "r");
    if (!fp) {
        // unzip dataset.zip to the folder 
        std::string command = "unzip " + base + "/dataset.zip -d " + base;
        int result = system(command.c_str());
        if (result != 0) {
            ERROR_OUT;
        }
    }


    auto X = read<float>(base + "/train_x.matrix"); 
    // std::cout << "size of X: " << X.num_row() << " " << X.num_col() << std::endl;

    auto labels = read<int>(base +"/train_y.matrix"); 
    // std::cout << "size of labels: " << labels.num_row() << " " << labels.num_col() << std::endl;

    auto Y = one_hot(labels, k);
    // std::cout << "size of Y: " << Y.num_row() << " " << Y.num_col() << std::endl;

    auto Xt = read<float>(base + "/test_x.matrix");
    // std::cout << "size of Xt: " << Xt.num_row() << " " << Xt.num_col() << std::endl;

    auto labels_t = read<int>(base + "/test_y.matrix"); 
    // std::cout << "size of labels_t: " << labels_t.num_row() << " " << labels_t.num_col() << std::endl;

    auto Yt = one_hot(labels_t, k);
    // std::cout << "size of Yt: " << Yt.num_row() << " " << Yt.num_col() << std::endl;

    return {X/255.0, Y, Xt/255.0, Yt};
}
