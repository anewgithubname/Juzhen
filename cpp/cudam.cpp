/**
 * @file cudam.cpp
 * @brief implementation of the cuda-powered matrix class.
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

#include "cudam.h"

using namespace std;
list<mem_space<float>> cuMatrix::alive_gpu_mems;
list<mem_space<float>> cuMatrix::dead_gpu_mems;
GPU_handle cuMatrix::global_handle = NULL;

GPUMemoryDeleter::~GPUMemoryDeleter(){
    long size = 0; 
    for(auto it = cuMatrix::alive_gpu_mems.begin(); it != cuMatrix::alive_gpu_mems.end(); it++){
        cudaFree(it->ptr);
        size += it->size;
    }
    for(auto it = cuMatrix::dead_gpu_mems.begin(); it != cuMatrix::dead_gpu_mems.end(); it++){
        cudaFree(it->ptr);
        size += it->size;
    }
    LOG_INFO("Total GPU memory released: {:.2f} MB.", size*sizeof(float)/1024.0/1024.0);
}

cuMatrix::cuMatrix(const Matrix<float>& M)
{
    static Profiler profiler("gpu copy");
    this->handle = global_handle;
    this->name = "cu_"+M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    float * p = cuMatrix::allocate(numcol * numrow);
    elements = shared_ptr<float[]>(p, [](auto p) {
        cuMatrix::free(p);
    });

    profiler.start();
    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyHostToDevice);
    profiler.end();

    if (stat != cudaSuccess) {
        LOG_ERROR("host to device memory copy failed, {}.", stat);
    }
}

cuMatrix::cuMatrix(const char *name, int numrow, int numcol, int trans)
{
    this->handle = global_handle;
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    float *p = cuMatrix::allocate(numcol * numrow);
    
    elements.reset(p, [](auto p) {
        cuMatrix::free(p);
        });

    zeros();
}

cuMatrix::cuMatrix(const cuMatrix& M){
    LOG_DEBUG("cuda copy constructor called");
    this->handle = M.handle;
    this->name = "copy of" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<float[]>(
            cuMatrix::allocate(numcol * numrow), [](auto p) {
                cuMatrix::free(p);
            });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", stat);
    }

}

cuMatrix::cuMatrix(cuMatrix &&M) noexcept{
    LOG_DEBUG("cuda move constructor called");
    this->handle = M.handle;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
}

cuMatrix & cuMatrix::operator=(const cuMatrix &M){
    if(this == &M) return *this;
    LOG_DEBUG("cuda copy assignment called");
    this->handle = M.handle;
    this->name = "copy of " + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<float[]>(
            cuMatrix::allocate(numcol * numrow), [](auto p) {
                cuMatrix::free(p);
            });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", stat);
        ERROR_OUT;
    }

    return *this;
}

cuMatrix &cuMatrix::operator=(cuMatrix &&M) noexcept{
    if(this == &M) return *this;
    LOG_DEBUG("cuda move assignment called");
    this->handle = M.handle;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = M.elements;
    M.elements = nullptr;
    return *this;
}

void cuMatrix::ones()
{
    fill(*this, 1.0f);
}

void cuMatrix::zeros()
{
    cudaError_t stat = cudaMemset(elements.get(), 0, numcol * numrow* sizeof(float));
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory set zero failed, {}.", stat);
    }
}


void cuMatrix::randn(double mean, double std){
    curandGenerator_t gen;
    // random generator
    curandStatus_t stat =  curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT);
    if (stat != CURAND_STATUS_SUCCESS) {
        LOG_ERROR("curand create generator failed, {}.", stat);
        ERROR_OUT;
    }
    // set seed
    stat = curandSetPseudoRandomGeneratorSeed(gen, Clock::now().time_since_epoch().count());
    if (stat != CURAND_STATUS_SUCCESS) {
        LOG_ERROR("curand set seed failed, {}.", stat);
        ERROR_OUT;
    }

    if (numcol * numrow % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (numcol * numrow + 1)*sizeof(float));
        stat = curandGenerateNormal(gen, p, numcol * numrow + 1, (float) mean, (float) std);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", stat);
            ERROR_OUT;
        }
        cudaMemcpy(elements.get(), p, numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(p);
    }else{
        // generate normal 
        stat = curandGenerateNormal(gen, elements.get(), numcol * numrow, (float) mean, (float) std);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", stat);
            ERROR_OUT;
        }
    }
    // destroy generator 
    stat = curandDestroyGenerator(gen);
    if (stat != CURAND_STATUS_SUCCESS) {
        LOG_ERROR("curand destroy generator failed, {}.", stat);
        ERROR_OUT;
    }
}


Matrix<float> cuMatrix::to_host() const
{
    Matrix<float> ret((string(name) + "->host").c_str(), numrow, numcol, transpose);
    GPU_status stat = cublasGetMatrix(numrow, numcol, sizeof(float), elements.get(), 
                    numrow, ret.elements.get(), numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("device memory upload to host failed, {}.", stat);
        ERROR_OUT;
    }
    return ret;
}


float cuMatrix::norm() const
{
    float res; 
    cublasSnrm2(handle, numrow * numcol, elements.get(), 1, &res);
    return res;
}

cuMatrix cuMatrix::dot(const cuMatrix &B) const
{
    static Profiler p("GPU dot"); p.start();
    cuMatrix C("dot", num_row(), B.num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f;
    GPU_status stat = cublasSgemm(handle, transA, transB, 
                    num_row(), B.num_col(), num_col(), &one, elements.get(), numrow,
                    B.elements.get(), B.numrow, &one, C.elements.get(), C.numrow);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("dot product failed");
    }
    p.end();
    return C;
}
//s1*M + a
cuMatrix cuMatrix::add(const float a, const float s1) const
{
    cuMatrix C("add", numrow, numcol, transpose);
    fill(C,a);

    GPU_status stat = cublasSaxpy(handle, numrow * numcol, &s1, elements.get(), 1, C.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("axpy failed");
    }
    return C;
}
//s1*M + s2*B
cuMatrix cuMatrix::add(const cuMatrix &B, float s1, float s2) const
{
    cuMatrix C("add", num_row(), num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    GPU_status stat = cublasSgeam(handle, transA, transB, num_row(), num_col(), &s1, elements.get(), numrow,
                &s2, B.elements.get(), B.numrow, C.elements.get(), C.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("add failed");
    }

    return C;
}

const cuMatrix cuMatrix::T() const
{
    string newname = name+"_T";
    cuMatrix MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}
//similar to MATLAB's sum function, sum the matrix along the specified dimension
cuMatrix sum(const cuMatrix &M, int dim)
{
    int transM = M.transpose;
    if (dim == 0)
    {
        transM = !transM; 
    }

    cuMatrix sumM("sumM", transM?M.numcol:M.numrow , 1, 0);
    cuMatrix ones("ones", transM?M.numrow:M.numcol, 1, 0);
    ones.ones();

    cublasOperation_t cudaTransM = transM ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f, zero = 0.0f;
    GPU_status stat = cublasSgemv(M.handle, cudaTransM, M.numrow, M.numcol, 
                &one, M.elements.get(), M.numrow, ones.elements.get(), 
                1, &zero, sumM.elements.get(), 1);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("sum failed, {}.", stat);
    }

    if (dim == 0)
    {
        sumM.transpose = 1;
    }
    return sumM;
}
ostream & operator <<(ostream &os, const cuMatrix &M){
    using namespace std;
    Matrix<float> hostM = M.to_host();
    // write obj to stream
    os << hostM.get_name()<< " " << hostM.num_row() << " by " << hostM.num_col() << endl;
    for (int i = 0; i < hostM.num_row(); i++)
    {
        os << endl;
        for (int j = 0; j < hostM.num_col(); j++)
        {
            os << hostM.elem(i, j) << " ";
        }
    }
    return os;
}
cuMatrix operator*(const cuMatrix &lM, const cuMatrix &rM)
{
    // cout << "* " << endl;
    return lM.dot(rM);
}
cuMatrix operator*(const cuMatrix &lM, const float &r)
{
    return lM.add(0, r);
}
//rvalue (in place) scaling
cuMatrix operator*(cuMatrix&& lM, const float& r)
{
    LOG_DEBUG("rval * called!");
    return operator*(r, std::move(lM));
}
cuMatrix operator*(const float &l, const cuMatrix &rM)
{
    // cout << "l*" << endl;
    return rM.add(0,l);
}
//rvalue (in place) scaling
cuMatrix operator*(const float& l, cuMatrix&& rM)
{
    int numelems = rM.numrow * rM.numcol;
    //original content rM is replaced to store the scaled result.
    GPU_status stat = cublasSscal(rM.handle, numelems, &l, rM.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("scal failed, {}.", stat);
    }
    return std::move(rM);
}
cuMatrix operator/(const cuMatrix &lM, const float &r)
{
    return lM.add(0, 1.0/r);
}

// rvalue(in place) division
cuMatrix operator/(cuMatrix &&lM, const float &r)
{
    const float s =  1.0/r;
    //original content lM is replaced to store the scaled result.
    int numelems = lM.numrow * lM.numcol;
    GPU_status stat = cublasSscal(lM.handle, numelems, &s, lM.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("scal failed, {}.", stat);
    }
    return std::move(lM);
}

cuMatrix operator+(const cuMatrix &lM, const cuMatrix &rM)
{
    return lM.add(rM, 1.0, 1.0);
}
cuMatrix operator+(const cuMatrix &lM, const float r)
{
    return lM.add(r, 1.0);
}
cuMatrix operator+(const float &l, const cuMatrix &rM)
{
    return rM.add(l, 1.0);
}
cuMatrix operator-(const cuMatrix &lM, const cuMatrix &rM)
{
    // cout<< "-" << endl;
    return lM.add(rM, 1.0, -1.0);
}
cuMatrix operator-(cuMatrix&lM, const float &r)
{
    return lM.add(-r, 1.0);
}
cuMatrix operator-(const float &l, const cuMatrix &rM)
{
    return rM.add(l, -1.0);
}
cuMatrix operator-(const cuMatrix &rM)
{
    return rM.add(0, -1.0);
}
cuMatrix& operator+=(cuMatrix &lM, const cuMatrix &rM)
{
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = rM.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    float s1 = 1.0, s2 = 1.0;
    GPU_status stat = cublasSgeam(lM.handle, transA, transB, lM.num_row(), lM.num_col(), 
                                  &s1, lM.elements.get(), lM.numrow, &s2, rM.elements.get(), 
                                  rM.numrow, lM.elements.get(), lM.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("+= failed, {}.", stat);
    }

    return lM;
}

cuMatrix& operator-=(cuMatrix &lM, const cuMatrix &rM)
{
    static Profiler p("GPU -="); p.start();
    // cout << "-=" << endl;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = rM.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    float s1 = 1.0, s2 = -1.0;
    GPU_status stat = cublasSgeam(lM.handle, transA, transB, lM.num_row(), lM.num_col(), 
                                  &s1, lM.elements.get(), lM.numrow, &s2, rM.elements.get(), 
                                  rM.numrow, lM.elements.get(), lM.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("-= failed, {}.", stat);
    }

    p.end();
    return lM;
}

curandStatus_t GPUSampler::stats = CURAND_STATUS_NOT_INITIALIZED;
curandGenerator_t GPUSampler::gen = NULL;


cuMatrix cuMatrix::randn(int m, int n){
    static Profiler p("rand gen"); p.start();
    auto gen = GPUSampler::gen;
    cuMatrix M("randn", m, n);

    if (m * n % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (m * n + 1)*sizeof(float));
        curandStatus_t curand_stat = curandGenerateNormal(gen, p, m * n + 1, 0.0f, 1.0f);
        if (curand_stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("sample random nums failed, {}.", curand_stat);
            ERROR_OUT;
        }
        cudaError_t stat = cudaMemcpy(M.elements.get(), p, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory copy failed, {}.", stat);
            ERROR_OUT;
        }
        cudaFree(p);
    }
    else {
        // generate normal 
        auto stat = curandGenerateNormal(gen, M.elements.get(), M.numcol * M.numrow, 0.0f, 1.0f);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", stat);
            ERROR_OUT;
        }
    }
    p.end();
    return M;
}

cuMatrix cuMatrix::ones(int m, int n)
{
    static Profiler p("ones"); p.start();
    cuMatrix M("ones", m, n); fill(M, 1.0f);
    p.end();
    return M;
}

cuMatrix cuMatrix::zeros(int m, int n)
{
    cuMatrix M("zeros", m, n);
    cudaError_t stat = cudaMemset(M.elements.get(), 0, M.numcol * M.numrow * sizeof(float));
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory set zero failed, {}.", stat);
    }
    return M;
}
