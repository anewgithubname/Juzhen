/**
 * @file cudam.hpp
 * @brief header or the cuda-powered matrix class.
 * @author Song Liu (song.liu@bristol.ac.uk)
 * 
    Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#ifndef CUDAM_HPP
#define CUDAM_HPP

#include "core.hpp"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda.h"
#include <curand.h>

typedef cublasHandle_t GPU_handle;
typedef cublasStatus_t GPU_status;

struct cuArrayDeleter
{
    void operator()(float p[]) const
    {
        // cout << "Cuda object freed." << endl;
        cudaFree(p);
    }
};


class cuMatrix : public Matrix<float> {
    GPU_handle handle;

    cuMatrix(GPU_handle handle, const char *name, int numrow, int numcol, int trans, shared_ptr<float[]> elements)
    :Matrix<float>(name, numrow, numcol, trans, elements){this->handle = handle;}
    cuMatrix(GPU_handle handle, const char* name, int numrow, int numcol, int trans);

public:
    //constructors and copiers.
    cuMatrix() : Matrix<float>() {handle = NULL;}
    cuMatrix(GPU_handle handle, const Matrix<float>& M);
    cuMatrix(GPU_handle handle, const char* name, int numrow, int numcol):cuMatrix(handle, name, numrow, numcol, 0) {};

    //matrix filler
    void ones();
    void zeros();
    void randn(){ randn(0.0, 1.0);}
    void randn(double mean, double std);

    //basic matrix ops
    cuMatrix dot(const cuMatrix &B) const;
    cuMatrix add(const cuMatrix &B, float s1, float s2) const;
    cuMatrix add(const float a, const float s1) const;
    float norm() const;
    const cuMatrix T() const;

    //do not support single element access
    float elem(int i, int j) const = delete;
    float& elem(int i, int j) = delete;

    //upload to host mem
    Matrix<float> to_host() const;

    //slicing matrix
    Matrix<float> slice(int rstart, int rend, int cstart, int cend) const { return to_host().slice(rstart, rend, cstart, cend); }
    Matrix<float> slice(const idxlist &rowidx, const idxlist &colidx) const { return to_host().slice(rowidx, colidx); }
    Matrix<float> rows(int rstart, int rend) const { return to_host().rows(rstart, rend); }
    Matrix<float> rows(idxlist rlist) const { return to_host().rows(rlist); }
    Matrix<float> columns(int cstart, int cend) const { return to_host().columns(cstart, cend); }
    Matrix<float> columns(idxlist clist) const { return to_host().columns(clist); }

    //our friends
    friend cuMatrix operator*(const float& l, cuMatrix&& rM);
    friend cuMatrix operator+=(cuMatrix &lM, const cuMatrix &rM);
    friend cuMatrix operator-=(cuMatrix &lM, const cuMatrix &rM);
    friend cuMatrix operator/(const float &l, const cuMatrix &rM);
    friend cuMatrix operator/(const cuMatrix &lM, const cuMatrix &rM);
    friend cuMatrix operator/(cuMatrix &&lM, const float &r);
    friend cuMatrix sum(const cuMatrix &M, int dim);
    friend cuMatrix exp(const cuMatrix &M);
    friend cuMatrix exp(cuMatrix &&M);
    friend cuMatrix log(const cuMatrix &M);
    friend cuMatrix tanh(const cuMatrix &M);
    friend cuMatrix tanh(cuMatrix &&M);
    friend cuMatrix d_tanh(const cuMatrix &M);
    friend cuMatrix d_tanh(cuMatrix &&M);
    friend void copy(cuMatrix &dest, const cuMatrix &src);
    friend cuMatrix fill(cuMatrix &M, float a);
    friend cuMatrix hstack(vector<cuMatrix> matrices);
    friend const cuMatrix vstack(vector<cuMatrix> matrices);
    friend cuMatrix hadmd(const cuMatrix& M1, const cuMatrix& M2);
    friend cuMatrix hadmd(const cuMatrix &M1, cuMatrix &&M2);
    friend cuMatrix hadmd(cuMatrix &&M1, const cuMatrix &M2);
};

cuMatrix sum(const cuMatrix &M, int dim);
ostream & operator <<(ostream &os, const cuMatrix &M);
cuMatrix operator*(const cuMatrix &lM, const cuMatrix &rM);
cuMatrix operator*(const cuMatrix &lM, const float &r);
cuMatrix operator*(const float &l, const cuMatrix &rM);
cuMatrix operator/(const cuMatrix &lM, const cuMatrix &rM);
cuMatrix operator/(const cuMatrix &lM, const float &r);
cuMatrix operator/(const float &l, const cuMatrix &rM);
cuMatrix operator+(const cuMatrix &lM, const cuMatrix &rM);
cuMatrix operator+(const cuMatrix &lM, const float r);
cuMatrix operator+(const float &l, const cuMatrix &rM);
cuMatrix operator-(const cuMatrix &lM, const cuMatrix &rM);
cuMatrix operator-(cuMatrix&lM, const float &r);
cuMatrix operator-(const float &l, const cuMatrix &rM);
cuMatrix operator-(const cuMatrix &rM);
cuMatrix operator+=(cuMatrix &lM, const cuMatrix &rM);
cuMatrix operator-=(cuMatrix &lM, const cuMatrix &rM);
cuMatrix exp(const cuMatrix &M);
cuMatrix exp(cuMatrix &&M);
cuMatrix log(const cuMatrix &M);
cuMatrix tanh(const cuMatrix &M);    
cuMatrix tanh(cuMatrix &&M);
cuMatrix d_tanh(cuMatrix &&M);
cuMatrix d_tanh(const cuMatrix &M);
cuMatrix hadmd(const cuMatrix &M1, const cuMatrix &M2);
cuMatrix fill(cuMatrix &M, float a);
void copy(cuMatrix &dest, const cuMatrix &src);
cuMatrix hstack(vector<cuMatrix> matrices);
const cuMatrix vstack(vector<cuMatrix> matrices);

#endif