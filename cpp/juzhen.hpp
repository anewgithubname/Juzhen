#ifndef JUZHEN_HPP
#define JUZHEN_HPP

int compute();

class Codeimp;

#include "core.hpp"
#include "matrix.hpp"
#include "operators.hpp"

class Code {
    std::unique_ptr<Codeimp> pimpl;

   public:
    Code();
    ~Code();
    int run();
    int render();
};

typedef Matrix<float> M;

#ifdef CUDA
#include "cumatrix.cuh"
typedef Matrix<CUDAfloat> CM;
#endif

// some basic statistics functions
#ifdef CUDA
#define __GPU_CPU__ __device__ __host__
#else
#define __GPU_CPU__
#endif

#include "mpsmatrix.hpp"

template <class D>
Matrix<D> randn_like(const Matrix<D> &M) {
    return Matrix<D>::randn(M.num_row(), M.num_col());
}
template <class D>
Matrix<D> rand_like(const Matrix<D> &M) {
    return Matrix<D>::rand(M.num_row(), M.num_col());
}
template <class D>
Matrix<D> ones_like(const Matrix<D> &M) {
    return Matrix<D>::ones(M.num_row(), M.num_col());
}
template <class D>
Matrix<D> zeros_like(const Matrix<D> &M) {
    return Matrix<D>::zeros(M.num_row(), M.num_col());
}

template <class T>
Matrix<T> mean(const Matrix<T> &a, int axis) {
    if (axis == 0) {
        return sum(a, 0) / (T)a.num_row();
    } else {
        return sum(a, 1) / (T)a.num_col();
    }
}

template <class T>
Matrix<T> sqrt(const Matrix<T> &M) {
    return elemwise([=] __GPU_CPU__(float x) { return sqrt(x); }, M);
}

template <class T>
Matrix<T> sqrt(Matrix<T> &&M) {
    return elemwise([=] __GPU_CPU__(float x) { return sqrt(x); }, M);
}

template <class T>
Matrix<T> stddev(const Matrix<T> &a, int axis) {
    Matrix<T> m = mean(a, axis);
    if (axis == 0) {
        m = Matrix<T>::ones(a.num_row(), 1) * m;
        return sqrt(sum(square(a - m), 0) / (T)a.num_row());
    } else {
        m = m * Matrix<T>::ones(1, a.num_col());
        return sqrt(sum(square(a - m), 1) / (T)a.num_col());
    }
}

template <class T>
Matrix<T> cov(const Matrix<T> &a, int axis) {
    Matrix<T> m = mean(a, axis);
    if (axis == 0) {
        m = Matrix<T>::ones(a.num_row(), 1) * m;
        return (a - m).T() * (a - m) / (T)a.num_row();
    } else {
        m = m * Matrix<T>::ones(1, a.num_col());
        return (a - m) * (a - m).T() / (T)a.num_col();
    }
}

#endif