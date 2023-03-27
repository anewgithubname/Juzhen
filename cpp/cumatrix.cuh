/**
 * @file cumatrix.hpp
 * @brief header or the cuda-powered matrix class.
 * @author Song Liu (song.liu@bristol.ac.uk)
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

#ifndef CUMATRIX_HPP
#define CUMATRIX_HPP

#include "core.hpp"
#include "matrix.hpp"

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#define CudaErrorCheck(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("CUDA ERROR: {} {}:{}", cudaGetErrorString(code), file, line);
        ERROR_OUT;
    }
}

#define CuBLASErrorCheck(ans) { CuBLASAssert((ans), __FUNCTION__,  __FILE__, __LINE__); }
inline void CuBLASAssert(cublasStatus_t code, const char* func, const char *file, int line, bool abort=true)
{
   if (code != cublasStatus_t::CUBLAS_STATUS_SUCCESS) 
   {
        LOG_ERROR("CUBLAS ERROR: {}, {}, {}:{}", cublasGetStatusString(code), func, file, line);
        ERROR_OUT;
   }
   else{
      LOG_DEBUG("CUBLAS OK, {}, {}:{}", func, file, line);
   }
}

#define threadsPerBlock 1024
#define cudaConfig(numElem) ((unsigned int) numElem + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock

typedef cublasHandle_t GPU_handle;
typedef cublasStatus_t GPU_status;

//memory functions
template<>
inline CUDAfloat* mem_alloc(size_t size){
    CUDAfloat *ptr = NULL; 
    CudaErrorCheck(cudaMalloc(&ptr, size * sizeof(CUDAfloat)));
    if(ptr == NULL)
        throw std::bad_alloc();
    return ptr;
}

template<>
inline void mem_free(CUDAfloat* ptr){
    cudaFree(ptr);
}

__global__ void copyKernel(float* d_out, float* d_in, size_t numElements, size_t numrow, size_t rowstart, size_t rowend, size_t colstart, size_t colend, bool direction);
__global__ void addKernel(float *d_out, float s1, float a, size_t numElements);

template <>
class Matrix<CUDAfloat> {
    size_t numcol;
    size_t numrow;
    bool transpose;
    std::string name;

    std::shared_ptr<CUDAfloat[]> elements;

    //NOTE: each matrix does have its own handle, however, for now, 
    //they are all assigned to the global handle. 
    GPU_handle handle;
    
    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol, int trans, std::shared_ptr<CUDAfloat[]> elements) {
        this->name = name;
        this->numrow = numrow;
        this->numcol = numcol;
        this->transpose = trans;
        this->elements = elements;
        this->handle = global_handle;
    }
    
    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol, int trans);

public: 
    static GPU_handle global_handle;

    //constructors
    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol, std::shared_ptr<CUDAfloat[]> elements): Matrix<CUDAfloat>(name, numrow, numcol, 0, elements) {};
    explicit Matrix<CUDAfloat>(const Matrix<float>& M);
    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol) :Matrix<CUDAfloat>(name, numrow, numcol, 0) {};

    //copy and move constructors, assignment operators
    Matrix<CUDAfloat>(const Matrix<CUDAfloat>& M);
    Matrix<CUDAfloat>(Matrix<CUDAfloat>&& M) noexcept;
    Matrix<CUDAfloat>& operator=(const Matrix<CUDAfloat>& M);
    Matrix<CUDAfloat>& operator=(Matrix<CUDAfloat>&& M) noexcept;


    inline size_t idx(size_t i, size_t j) const{
        return transpose ? i *numrow + j : j * numrow + i;
    }
    // access matrix info
    inline CUDAfloat elem(size_t i, size_t j) const { return elements[idx(i, j)]; }
    inline CUDAfloat &elem(size_t i, size_t j) { return elements[idx(i, j)]; }
    inline CUDAfloat operator()(size_t i, size_t j) const { return elements[idx(i, j)]; }
    inline CUDAfloat &operator()(size_t i, size_t j) { return elements[idx(i, j)]; }
    
    inline size_t num_col() const { return transpose ? numrow : numcol; }
    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline size_t get_transpose() const { return transpose; }
    std::string get_name() const { return name; }
    const CUDAfloat * data() const { return elements.get(); }

    //matrix filler
    void ones();
    void zeros();

    static Matrix<CUDAfloat> randn(size_t m, size_t n);
    static Matrix<CUDAfloat> rand(size_t m, size_t n);
    static Matrix<CUDAfloat> ones(size_t m, size_t n);
    static Matrix<CUDAfloat> zeros(size_t m, size_t n);

    //basic matrix ops
    Matrix<CUDAfloat> dot(const Matrix<CUDAfloat>& B) const;
    
    Matrix<CUDAfloat> add(const Matrix<CUDAfloat>& B, float s1, float s2) const;
    void add(const Matrix<CUDAfloat>& B, float s1, float s2);

    Matrix<CUDAfloat> add(float a, float s1) const;
    void add(float a, float s1);

    Matrix<CUDAfloat> scale(float s1) const { return add(0, s1); }
    void scale(float s1);
    
    void reciprocal(double l);
    Matrix<CUDAfloat> reciprocal(double l) const;

    float norm() const;
    const Matrix<CUDAfloat> T() const;

    //upload to host mem
    Matrix<float> to_host() const;

    ////slicing matrix
    //Matrix<float> slice(int rstart, int rend, int cstart, int cend) const { return to_host().slice(rstart, rend, cstart, cend); }
    //Matrix<float> slice(const idxlist& rowidx, const idxlist& colidx) const { return to_host().slice(rowidx, colidx); }
    Matrix<CUDAfloat> slice(size_t rstart, size_t rend, size_t cstart, size_t cend) const {
        size_t numElem = (rend - rstart) * (cend - cstart);
        if (transpose) {
            size_t tmp = rstart;
            rstart = cstart;
            cstart = tmp;
            tmp = rend;
            rend = cend;
            cend = tmp;
        }
        Matrix<CUDAfloat> M("submatrix", rend - rstart, cend - cstart, transpose);
        copyKernel <<< cudaConfig(numElem) >>> ((float *)M.elements.get(), (float*) elements.get(), numElem, numrow, rstart, rend, cstart, cend, true);
        return M; 
    }

    void slice(size_t rstart, size_t rend, size_t cstart, size_t cend, const Matrix<CUDAfloat>& M) {
        size_t numElem = (rend - rstart) * (cend - cstart);
        if (transpose) {
            size_t tmp = rstart;
            rstart = cstart;
            cstart = tmp;
            tmp = rend;
            rend = cend;
            cend = tmp;
        }
        copyKernel <<< cudaConfig(numElem) >>> ((float *)M.elements.get(), (float*) elements.get(), numElem, numrow, rstart, rend, cstart, cend, false);
    }

    Matrix<CUDAfloat> rows(size_t rstart, size_t rend) const { return slice(rstart, rend, 0, num_col()); }
    void rows(size_t rstart, size_t rend, const Matrix<CUDAfloat>& M) { return slice(rstart, rend, 0, num_col(), M); }
    //Matrix<float> rows(idxlist rlist) const { return to_host().rows(rlist); }
    Matrix<CUDAfloat> columns(size_t cstart, size_t cend) const { return slice(0, num_row(), cstart, cend); }
    void columns(size_t cstart, size_t cend, const Matrix<CUDAfloat>& M) { return slice(0, num_row(), cstart, cend, M); }
    //Matrix<float> columns(idxlist clist) const { return to_host().columns(clist); }

    //our friends
    friend Matrix<CUDAfloat> sum(const Matrix<CUDAfloat>& M, int dim);
    friend Matrix<CUDAfloat> exp(const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> exp(Matrix<CUDAfloat>&& M);
    friend Matrix<CUDAfloat> log(const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> tanh(const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> tanh(Matrix<CUDAfloat>&& M);
    friend Matrix<CUDAfloat> d_tanh(const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> d_tanh(Matrix<CUDAfloat>&& M);
    friend Matrix<CUDAfloat> square(const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> square(Matrix<CUDAfloat>&& M);
    friend Matrix<CUDAfloat> elemwise(float (*func)(float), const Matrix<CUDAfloat>& M);
    friend Matrix<CUDAfloat> elemwise(float (*func)(float), Matrix<CUDAfloat>&& M);
    template<class Function>
    friend Matrix<CUDAfloat> reduce(Function func, const Matrix<CUDAfloat>& M, int dim, int k); //columns operations
    template<class Function>
    friend Matrix<CUDAfloat> elemwise(Function func, const Matrix<CUDAfloat>& M); //lvalue elementwise op
    template<class Function> 
    friend Matrix<CUDAfloat> elemwise(Function func, Matrix<CUDAfloat>&& M); //rvalue elementwise op

    friend void copy(Matrix<CUDAfloat>& dest, const Matrix<CUDAfloat>& src);
    friend Matrix<CUDAfloat>& fill(Matrix<CUDAfloat>& M, double a);
    friend Matrix<CUDAfloat> hstack(std::vector<Matrix<CUDAfloat>>& matrices);
    friend Matrix<CUDAfloat> hstack(std::vector<Matrix<CUDAfloat>>&& matrices);
    friend const Matrix<CUDAfloat> vstack(std::vector<Matrix<CUDAfloat>>& matrices);
    friend const Matrix<CUDAfloat> vstack(std::vector<Matrix<CUDAfloat>>&& matrices);
    friend Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1, const Matrix<CUDAfloat>& M2);
    friend Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat> &M1, Matrix<CUDAfloat> &&M2);
    friend Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat> &&M1, const Matrix<CUDAfloat> &M2);

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
        LOG_INFO("GPU sampler is initialized with seed {}.", seed);
    }

    void setseed(int seed) {
        // set seed
        stats = curandSetGeneratorOffset(gen, seed);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand set seed failed");
            ERROR_OUT;
        }
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
Matrix<CUDAfloat> sum(const Matrix<CUDAfloat>& M, int dim);
std::ostream& operator <<(std::ostream& os, const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> exp(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> exp(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> log(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> tanh(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> tanh(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> d_tanh(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> d_tanh(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> square(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> square(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> elemwise(float (*func)(float), const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> elemwise(float (*func)(float), Matrix<CUDAfloat>&& M);

template<class Function>
__global__ void reduce_kernel(Function func, float* vecdes, float* vec, size_t lenvec, size_t lenvecdes, size_t numvecs)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numvecs)
    {
        func(&vec[i*lenvec], &vecdes[i*lenvecdes], lenvec, lenvecdes);
    }
}

template<class Function>
Matrix<CUDAfloat> reduce(Function func, const Matrix<CUDAfloat>& M, int dim, int k)
{   
    // STATIC_TIC;
    bool trans = false;
    if (dim == 0 && !M.transpose) {
        trans = false;
    }else if(dim == 1 && !M.transpose){
        trans = true;
    }else if(dim == 0 && M.transpose){
        trans = true; 
    }else{
        trans = false;
    }

    if (!trans) {
        Matrix<CUDAfloat> result("resM", k, M.numcol);
        reduce_kernel << <cudaConfig(M.numcol) >> > (func, (float*)result.elements.get(), (float*)M.elements.get(), M.numrow, k, M.numcol);
        if(M.transpose){
            return result.T();
        }else{
            return result;
        }
    }
    else {
        //transpose the matrix
        Matrix<CUDAfloat> t("tzeros", M.numcol, M.numrow);
        t += M.transpose? M : M.T();

        Matrix<CUDAfloat> result("resM", k, t.numcol);
        reduce_kernel << <cudaConfig(t.numcol) >> > (func, (float*)result.elements.get(), (float*)t.elements.get(), t.numrow, k, t.numcol);
        if(M.transpose){
            return result; 
        }else{
            return result.T();
        }
    }
    
}

template<class Function>
__global__ void elemwise_kernel(Function func, float* vecdes, float* vec, size_t numElements)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = func(vec[i]);
    }
}

template<class Function>
__global__ void Inplace_elemwise_kernel(Function func, float* vecdes, size_t numElements)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = func(vecdes[i]);
    }
}

template<class Function>
Matrix<CUDAfloat> elemwise(Function func, const Matrix<CUDAfloat>& M)
{   
    // STATIC_TIC;

    Matrix<CUDAfloat> result("resM", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col(); 
    elemwise_kernel <<<cudaConfig(numElem)>>> (func, (float*)result.elements.get(), (float*)M.elements.get(), numElem);
    //TODO: check cuda kernel error

    // STATIC_TOC;
    return result;
}

template<class Function>
Matrix<CUDAfloat> elemwise(Function func, Matrix<CUDAfloat>&& M)
{   
    // STATIC_TIC;

    size_t numElem = M.num_row() * M.num_col(); 
    Inplace_elemwise_kernel <<<cudaConfig(numElem)>>> (func, (float*)M.elements.get(), numElem);
    //TODO: check cuda kernel error
    // STATIC_TOC;
    return std::move(M);
}

Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1, const Matrix<CUDAfloat>& M2);
Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1, Matrix<CUDAfloat>&& M2);
Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat>&& M1, const Matrix<CUDAfloat>& M2);
Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat>&& M1, Matrix<CUDAfloat>&& M2);
Matrix<CUDAfloat>& fill(Matrix<CUDAfloat>& M, double a);
void copy(Matrix<CUDAfloat>& dest, const Matrix<CUDAfloat>& src);
Matrix<CUDAfloat> hstack(std::vector<Matrix<CUDAfloat>>& matrices);
Matrix<CUDAfloat> hstack(std::vector<Matrix<CUDAfloat>>&& matrices);
const Matrix<CUDAfloat> vstack(std::vector<Matrix<CUDAfloat>>& matrices);
const Matrix<CUDAfloat> vstack(std::vector<Matrix<CUDAfloat>>&& matrices);

#endif