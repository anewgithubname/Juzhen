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
#include "matrix.hpp"
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

typedef cublasHandle_t GPU_handle;
typedef cublasStatus_t GPU_status;
struct GPUMemoryDeleter;

class cuMatrix : public Matrix<float> {
    //NOTE: each matrix does have its own handle, however, for now, 
    //they are all assigned to the global handle. 
    GPU_handle handle;

    static std::list<mem_space<float>> alive_gpu_mems;
    static std::list<mem_space<float>> dead_gpu_mems;
    static float* allocate(int size) {
        // static Profiler p("allocator_GPU"); p.start();
        // search for space in the freed space.
        for (auto it = dead_gpu_mems.begin(); it != dead_gpu_mems.end(); it++) {
            if (it->size == size) {
                LOG_DEBUG("Found space in dead GPU memory: {}, address: {}", size, fmt::ptr(it->ptr));
                float* ptr = it->ptr;
                mem_space<float> mem; mem.ptr = it->ptr;  mem.size = it->size;
                alive_gpu_mems.push_back(mem);
                dead_gpu_mems.erase(it);
                return ptr;
            }
        }
        // no space available, allocate new space
        float* ptr = NULL; 
        cudaError_t stat = cudaMalloc(&ptr, size * sizeof(float));
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory allocation failed, {}.", cudaGetErrorString(stat));
            exit(1);
        }
        LOG_DEBUG("No GPU space available, allocate new space: {}, address: {}", size, fmt::ptr(ptr));
        mem_space<float> space = { ptr, size };
        alive_gpu_mems.push_back(space);
        return ptr;
        // p.end();
    }

    static void free(float* ptr) {
        //static Profiler p("deleter_GPU"); p.start();
        // find the ptr in the alive mem list
        int size = -1;
        for (auto it = alive_gpu_mems.begin(); it != alive_gpu_mems.end(); it++) {
            if (it->ptr == ptr) {
                size = it->size;
            }
        }
        //copy it to the dead mem list
        //cout << "freeing GPU " << ptr << " size: " << size << endl; 
        mem_space<float> mem = { ptr, size };
        dead_gpu_mems.push_back(mem);

        // delete the entry from the alive list
        for (auto it = alive_gpu_mems.begin(); it != alive_gpu_mems.end(); it++) {
            if (it->ptr == ptr) {
                alive_gpu_mems.erase(it);
                break;
            }
        }
         //p.end();
    }
    cuMatrix(const char *name, int numrow, int numcol, int trans, 
             std::shared_ptr<float[]> elements)
    :Matrix<float>(name, numrow, numcol, trans, elements){this->handle = global_handle;}
    cuMatrix(const char* name, int numrow, int numcol, int trans);

public:
    static GPU_handle global_handle;
    //constructors and copiers.
    cuMatrix() : Matrix<float>() {handle = global_handle;}
    cuMatrix(const Matrix<float>& M);
    cuMatrix(const char* name, int numrow, int numcol):cuMatrix(name, numrow, numcol, 0) {};

    cuMatrix(const cuMatrix &M);
    cuMatrix(cuMatrix &&M) noexcept;
    cuMatrix &operator=(const cuMatrix &M);
    cuMatrix &operator=(cuMatrix &&M) noexcept;

    //matrix filler
    void ones();
    void zeros();
    void randn(){ randn(0.0, 1.0);}
    void randn(double mean, double std) override;

    static cuMatrix randn(int m, int n);
    static cuMatrix ones(int m, int n);
    static cuMatrix zeros(int m, int n);

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
    friend cuMatrix operator*(cuMatrix&& lM, const float &r);
    friend cuMatrix& operator+=(cuMatrix &lM, const cuMatrix &rM);
    friend cuMatrix& operator-=(cuMatrix &lM, const cuMatrix &rM);
    friend cuMatrix operator/(const float &l, const cuMatrix &rM);
    friend cuMatrix operator/(const float &l, cuMatrix &&rM);
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
    friend cuMatrix square(cuMatrix&& M);
    friend void copy(cuMatrix &dest, const cuMatrix &src);
    friend cuMatrix& fill(cuMatrix &M, float a);
    friend cuMatrix hstack(std::vector<cuMatrix> &matrices);
    friend cuMatrix hstack(std::vector<cuMatrix> &&matrices);
    friend const cuMatrix vstack(std::vector<cuMatrix> &matrices);
    friend const cuMatrix vstack(std::vector<cuMatrix> &&matrices);
    friend cuMatrix hadmd(const cuMatrix& M1, const cuMatrix& M2);
    // friend cuMatrix hadmd(const cuMatrix &M1, cuMatrix &&M2);
    // friend cuMatrix hadmd(cuMatrix &&M1, const cuMatrix &M2);
    friend GPUMemoryDeleter;
};

struct GPUMemoryDeleter{
    ~GPUMemoryDeleter();
};

struct GPUSampler {
 
    static curandStatus_t stats;
    static curandGenerator_t gen;
    
    //random sampler
    GPUSampler(int seed) {
        // random generator
        curandStatus_t stats = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand create generator failed.");
            ERROR_OUT;
        }
        // set seed
        stats = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand set seed failed");
            ERROR_OUT;
        }
        LOG_INFO("GPU sampler is initialized with seeed {}.", seed);
    }
    //deconstruct the sampler
    ~GPUSampler() {
        // destroy generator 
        stats = curandDestroyGenerator(gen);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand destroy generator failed!");
            ERROR_OUT;
        }
        LOG_INFO("GPU sampler is destroyed!");
    }
};

//operators, math functions
cuMatrix sum(const cuMatrix &M, int dim);
std::ostream & operator <<(std::ostream &os, const cuMatrix &M);
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
cuMatrix& operator+=(cuMatrix &lM, const cuMatrix &rM);
cuMatrix& operator-=(cuMatrix &lM, const cuMatrix &rM);
cuMatrix exp(const cuMatrix &M);
cuMatrix exp(cuMatrix &&M);
cuMatrix log(const cuMatrix &M);
cuMatrix tanh(const cuMatrix &M);    
cuMatrix tanh(cuMatrix &&M);
cuMatrix d_tanh(cuMatrix &&M);
cuMatrix d_tanh(const cuMatrix &M);
cuMatrix square(cuMatrix&& M);
cuMatrix hadmd(const cuMatrix &M1, const cuMatrix &M2);
cuMatrix& fill(cuMatrix &M, float a);
void copy(cuMatrix &dest, const cuMatrix &src);
cuMatrix hstack(std::vector<cuMatrix> &matrices);
cuMatrix hstack(std::vector<cuMatrix> &&matrices);
const cuMatrix vstack(std::vector<cuMatrix> &matrices);
const cuMatrix vstack(std::vector<cuMatrix> &&matrices);

#endif