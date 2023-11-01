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

#ifndef CPU_ONLY
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define __GPU_CPU__ __device__ __host__
#else
#define __GPU_CPU__
#endif

#ifndef CPU_ONLY
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
#ifndef CPU_ONLY
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
    #ifdef CPU_ONLY
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
        m = Matrix<T>::zeros(theta.num_row(), theta.num_col());
        v = Matrix<T>::zeros(theta.num_row(), theta.num_col());
    }
    adam_state(double alpha, size_t nrow, size_t ncol)
        :iteration(1), alpha(alpha), beta1(0.9), beta2(0.999), eps(1e-8){
        m = Matrix<T>::zeros(nrow, ncol);
        v = Matrix<T>::zeros(nrow, ncol);
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