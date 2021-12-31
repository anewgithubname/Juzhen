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
// #define PROFILING

using namespace std;
list<mem_space<float>> cuMatrix::alive_gpu_mems;
list<mem_space<float>> cuMatrix::dead_gpu_mems;

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
    std::cout << "Total GPU memory released: " << size*sizeof(float)/1024.0/1024.0 << " MB." << std::endl;
}

cuMatrix::cuMatrix(GPU_handle handle, const Matrix<float>& M)
{
    this->handle = handle;
    this->name = "cu_"+M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    float * p = cuMatrix::allocate(numcol * numrow);
    elements = shared_ptr<float[]>(p, [](auto p) {
        cuMatrix::free(p);
    });

#ifdef PROFILING
    static Profiler profiler("Copying from Host to GPU"); 
    if(M.num_col() == 128 && M.num_row() == 28*28)
        profiler.start();
#endif
    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyHostToDevice);
    
#ifdef PROFILING
    if(M.num_col() == 128 && M.num_row() == 28*28)
        profiler.end();
#endif

    if (stat != cudaSuccess) {
        printf ("host to device memory copy failed\n");
    }
}

cuMatrix::cuMatrix(GPU_handle handle, const char *name, int numrow, int numcol, int trans)
{
    this->handle = handle;
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    float *p = cuMatrix::allocate(numcol * numrow);
    //cudaError_t custat = cudaMalloc(&p, numcol * numrow* sizeof(float));
    //if (custat != cudaSuccess) {
    //    printf ("device memory allocation failed");
    //}
    //
    elements.reset(p, [](auto p) {
        cuMatrix::free(p);
        });

    zeros();
}

cuMatrix::cuMatrix(const cuMatrix& M){
    std::cout << "cuda copy constructor called" << std::endl;
    this->handle = M.handle;
    this->name = "copy of" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<float[]>(
            cuMatrix::allocate(numcol * numrow), [](auto p) {
                cuMatrix::free(p);
            });

    cudaError_t custat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (custat != cudaSuccess) {
        printf ("device memory copy failed");
    }

}

cuMatrix::cuMatrix(cuMatrix &&M) noexcept{
    // cout << "cuda move constructor called" << endl;
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
    std::cout << "cuda copy assignment called" << std::endl;
    this->handle = M.handle;
    this->name = "copy of " + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<float[]>(
            cuMatrix::allocate(numcol * numrow), [](auto p) {
                cuMatrix::free(p);
            });

    cudaError_t custat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (custat != cudaSuccess) {
        printf ("device memory copy failed");
    }

    return *this;
}

cuMatrix &cuMatrix::operator=(cuMatrix &&M) noexcept{
    if(this == &M) return *this;
    // cout << "cuda move assignment called" << endl;
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
    float *vec = new float[numcol * numrow];
    for (int i = 0; i < numcol * numrow; i++) 
        vec[i] = 1.0;
    
    GPU_status stat = cublasSetVector(numcol * numrow, 
                        sizeof(float), vec, 1, elements.get(), 1);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory set one failed");
    }

    delete[] vec;
    
}

void cuMatrix::zeros()
{
    cudaError_t stat = cudaMemset(elements.get(), 0, numcol * numrow* sizeof(float));
    if (stat != cudaSuccess) {
        printf ("device memory set zero failed");
    }
}


void cuMatrix::randn(double mean, double std){
    curandGenerator_t gen;
    curandStatus_t stats =  curandCreateGenerator(&gen, 
            CURAND_RNG_PSEUDO_DEFAULT);
    if (stats != CURAND_STATUS_SUCCESS) {
        printf ("curand create generator failed");
    }
    stats = curandSetPseudoRandomGeneratorSeed(gen, Clock::now().time_since_epoch().count());
    if (stats != CURAND_STATUS_SUCCESS) {
        printf ("curand set seed failed");
    }
    stats = curandGenerateNormal(gen, elements.get(), numcol * numrow, (float) mean, (float) std);
    if (stats != CURAND_STATUS_SUCCESS) {
        printf ("curand generate normal failed");
    }

    stats = curandDestroyGenerator(gen);
    if (stats != CURAND_STATUS_SUCCESS) {
        printf ("curand destroy generator failed");
    }
}


Matrix<float> cuMatrix::to_host() const
{
    Matrix<float> ret("->host", numrow, numcol, transpose);
    GPU_status stat = cublasGetMatrix(numrow, numcol, sizeof(float), elements.get(), 
                    numrow, ret.elements.get(), numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
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
    // cout << "dot " << endl;
    cuMatrix C(handle, "->dot", num_row(), B.num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f;
    GPU_status stat = cublasSgemm(handle, transA, transB, 
                    num_row(), B.num_col(), num_col(), &one, elements.get(), numrow,
                    B.elements.get(), B.numrow, &one, C.elements.get(), C.numrow);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("dot product failed");
    }
    return C;
}
//s1*M + a
cuMatrix cuMatrix::add(const float a, const float s1) const
{
    cuMatrix C(handle, "add_C", numrow, numcol, transpose);
    fill(C,a);

    GPU_status stat = cublasSaxpy(handle, numrow * numcol, &s1, elements.get(), 1, C.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("axpy failed");
    }
    return C;
}
//s1*M + s2*B
cuMatrix cuMatrix::add(const cuMatrix &B, float s1, float s2) const
{
    // cout << "add s1s2 " << s1 << " " << s2 << endl;
    cuMatrix C(handle, "add_C", num_row(), num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    GPU_status stat = cublasSgeam(handle, transA, transB, num_row(), num_col(), &s1, elements.get(), numrow,
                &s2, B.elements.get(), B.numrow, C.elements.get(), C.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("sum failed");
    }

    return C;
}

const cuMatrix cuMatrix::T() const
{
    string newname = name+"_T";
    cuMatrix MT(handle, newname.c_str(), numrow, numcol, !transpose, elements);
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

    cuMatrix sumM(M.handle, "sumM", transM?M.numcol:M.numrow , 1, 0);
    cuMatrix ones(M.handle, "ones", transM?M.numrow:M.numcol, 1, 0);
    ones.ones();

    cublasOperation_t cudaTransM = transM ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f, zero = 0.0f;
    GPU_status stat = cublasSgemv(M.handle, cudaTransM, M.numrow, M.numcol, 
                &one, M.elements.get(), M.numrow, ones.elements.get(), 
                1, &zero, sumM.elements.get(), 1);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("sum failed");
    }

    if (dim == 0)
    {
        sumM.transpose = 1;
    }
    return sumM;
}
ostream & operator <<(ostream &os, const cuMatrix &M){
    using namespace std;
    // write obj to stream
    os << M.get_name()<< " " << M.num_row() << " by " << M.num_col();
    Matrix<float> hostM = M.to_host();
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
cuMatrix operator*(const float &l, const cuMatrix &rM)
{
    // cout << "l*" << endl;
    return rM.add(0,l);
}
cuMatrix operator*(const float& l, cuMatrix&& rM)
{
    // cout << "rval *\n";
    int numelems = rM.numrow * rM.numcol;
    GPU_status stat = cublasSscal(rM.handle, numelems, &l, rM.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("axpy failed");
    }
    return std::move(rM);
}
cuMatrix operator/(const cuMatrix &lM, const float &r)
{
    return lM.add(0, 1.0/r);
}
//rvalue division
cuMatrix operator/(cuMatrix &&lM, const float &r)
{
    const float s =  1.0/r;
    // cout << "rvalue division " << s << endl;
    int numelems = lM.numrow * lM.numcol;
    GPU_status stat = cublasSscal(lM.handle, numelems, &s, lM.elements.get(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("axpy failed");
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
        printf ("+= failed");
    }

    return lM;
}

cuMatrix& operator-=(cuMatrix &lM, const cuMatrix &rM)
{
    // cout << "-=" << endl;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = rM.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    float s1 = 1.0, s2 = -1.0;
    GPU_status stat = cublasSgeam(lM.handle, transA, transB, lM.num_row(), lM.num_col(), 
                                  &s1, lM.elements.get(), lM.numrow, &s2, rM.elements.get(), 
                                  rM.numrow, lM.elements.get(), lM.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("-= failed");
    }

    return lM;
}