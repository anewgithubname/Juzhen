/**
 * @file cumatrix.cu
 * @brief some non-kernel implementation of CUDA matrix class
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

#include "cumatrix.cuh"

using namespace std;

GPU_handle Matrix<CUDAfloat>::global_handle = NULL;

Matrix<CUDAfloat>::Matrix(const Matrix<float>& M)
{
    static Profiler profiler("upload to CUDA memory");
    this->handle = global_handle;
    this->name = "cu_" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    CUDAfloat* p = Memory<CUDAfloat>::allocate( numcol * numrow);
    elements = shared_ptr<CUDAfloat[]>(p, [](CUDAfloat* p) {
        Memory<CUDAfloat>::free(p);
        });

    profiler.start();
    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyHostToDevice);
    profiler.end();

    if (stat != cudaSuccess) {
        LOG_ERROR("host to device memory copy failed, {}.", stat);
    }
}

Matrix<CUDAfloat>::Matrix(const char* name, size_t numrow, size_t numcol, int trans)
{
    this->handle = global_handle;
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    CUDAfloat* p = Memory<CUDAfloat>::allocate( numcol * numrow);
    elements.reset(p, [](CUDAfloat* p) {
        Memory<CUDAfloat>::free(p);
        });

    zeros();
}

Matrix<CUDAfloat>::Matrix(const Matrix<CUDAfloat>& M) {
    LOG_DEBUG("cuda copy constructor called");
    this->handle = M.handle;
    this->name = "copy of" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<CUDAfloat[]>(
        Memory<CUDAfloat>::allocate(  numcol * numrow), [](CUDAfloat* p) {
            Memory<CUDAfloat>::free(p);
        });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", stat);
    }

}

Matrix<CUDAfloat>::Matrix(Matrix<CUDAfloat>&& M) noexcept {
    LOG_DEBUG("cuda move constructor called");
    this->handle = M.handle;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
}

Matrix<CUDAfloat>& Matrix<CUDAfloat>::operator=(const Matrix<CUDAfloat>& M) {
    if (this == &M) return *this;
    LOG_DEBUG("cuda copy assignment called");
    this->handle = M.handle;
    this->name = "copy of " + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<CUDAfloat[]>(
        Memory<CUDAfloat>::allocate(  numcol * numrow), [](CUDAfloat* p) {
            Memory<CUDAfloat>::free(p);
        });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", stat);
        ERROR_OUT;
    }

    return *this;
}

Matrix<CUDAfloat>& Matrix<CUDAfloat>::operator=(Matrix<CUDAfloat>&& M) noexcept {
    if (this == &M) return *this;
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


void Matrix<CUDAfloat>::ones()
{
    fill(*this, 1.0f);
}

void Matrix<CUDAfloat>::zeros()
{
    LOG_DEBUG("cuda zeros called, {}, size, {}. ", (void *)elements.get(),  numcol * numrow * sizeof(CUDAfloat));
    // cudaError_t stat = cudaMemset( elements.get(), 0, numcol * numrow * sizeof(GPUfloat));
    // if (stat != cudaSuccess) {
    //     LOG_ERROR("device memory set zero failed, {}, size {}.", stat, numcol * numrow * sizeof(GPUfloat));
    // }
    fill(*this, 0.0f);
}

Matrix<float> Matrix<CUDAfloat>::to_host() const
{
    STATIC_TIC;
    Matrix<float> ret((string(name) + "->host").c_str(), numrow, numcol, transpose);
    // if (numrow * numcol == 1) {
    //     p.start();
    // }
    //GPU_status stat = cublasGetMatrix(numrow, numcol, sizeof(float), elements.get(), 
    //                numrow, ret.elements.get(), numrow);
    cudaError_t stat = cudaMemcpy(ret.elements.get(), (float *) elements.get(), sizeof(float) * numrow * numcol, ::cudaMemcpyDeviceToHost);
    // if (numrow * numcol == 1) {
    //     p.end();
    // }
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory upload to host failed, {}.", stat);
        ERROR_OUT;
    }
    STATIC_TOC;
    return ret;
}


float Matrix<CUDAfloat>::norm() const
{
    float res;
    CuBLASErrorCheck(
        cublasSnrm2(handle, numrow * numcol, (float *) elements.get(), 1, &res)
    )
    return res;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::dot(const Matrix<CUDAfloat>& B) const
{
    STATIC_TIC;
    //check if the dimensions are compatible
    if (num_col() != B.num_row())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<CUDAfloat> C("dot", num_row(), B.num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f;
    CuBLASErrorCheck( 
        cublasSgemm(handle, transA, transB,
        num_row(), B.num_col(), num_col(), &one, (float *) elements.get(), numrow,
        (float *) B.elements.get(), B.numrow, &one, (float *) C.elements.get(), C.numrow) 
    )
    STATIC_TOC;
    return C;
}
//s1*M + a
Matrix<CUDAfloat> Matrix<CUDAfloat>::add(float a, float s1) const
{
    Matrix<CUDAfloat> C("add", numrow, numcol, transpose);
    fill(C, a);

    CuBLASErrorCheck(  
        cublasSaxpy(handle, numrow * numcol, &s1, (float *) elements.get(), 1, (float *) C.elements.get(), 1)
    )
    return C;
}

//rvalue s1*M + a
void Matrix<CUDAfloat>::add(float a, float s1)
{
    size_t numElem = numrow * numcol;
    addKernel<<<cudaConfig(numElem)>>>((float*) elements.get(), s1, a, numElem);
}

//M = s1*M
void Matrix<CUDAfloat>::scale(float s1){
    //original content lM is replaced to store the scaled result.
    size_t numelems = numrow * numcol;
    CuBLASErrorCheck( 
        cublasSscal(handle, numelems, (const float *) &s1, (float *)elements.get(), 1)
    )
}

//s1*M + s2*B
Matrix<CUDAfloat> Matrix<CUDAfloat>::add(const Matrix<CUDAfloat>& B, float s1, float s2) const
{
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<CUDAfloat> C("add", num_row(), num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    CuBLASErrorCheck(
        cublasSgeam(handle, transA, transB, num_row(), num_col(), &s1, (float *) elements.get(), numrow,
        &s2, (float *) B.elements.get(), B.numrow, (float *) C.elements.get(), C.numrow)
    )

    return C;
}
//in place s1*M + s2*B
void Matrix<CUDAfloat>::add(const Matrix<CUDAfloat>& B, float s1, float s2) {
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = transpose != B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    CuBLASErrorCheck(
        cublasSgeam(handle, transA, transB, numrow, numcol, &s1, (float*)elements.get(), numrow,
            &s2, (float*)B.elements.get(), B.numrow, (float*)elements.get(), numrow)
    )

}

// elementwise division
__global__ void divKernel(float *d_out, float *d_in, size_t numElements)
{

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] /= d_in[i];
    }
}


// inplace division
__global__ void divKernel(float *d_out, float d_in1, float *d_in2, size_t numElements)
{

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = d_in1 / d_in2[i];
    }
}

// M_new = M / l
Matrix<CUDAfloat> Matrix<CUDAfloat>::eleminv(double l) const{
    Matrix<CUDAfloat> result("elem_rec", numrow, numcol, transpose);

    size_t numElem = numrow * numcol;
    divKernel<<<cudaConfig(numElem)>>>((float*) result.elements.get(), (float)l, (float*) elements.get(), numElem);

    return result;
}

// M = M / l
void Matrix<CUDAfloat>::eleminv(double l){
    LOG_DEBUG("rval l/rM called!");

    size_t numElem = numrow * numcol;
    divKernel <<<cudaConfig(numElem)>>> ((float*) elements.get(), (float)l, (float*) elements.get(), numElem);
}

const Matrix<CUDAfloat> Matrix<CUDAfloat>::T() const
{
    string newname = name + "_T";
    Matrix<CUDAfloat> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}
//similar to MATLAB's sum function, sum the matrix along the specified dimension
Matrix<CUDAfloat> sum(const Matrix<CUDAfloat>& M, int dim)
{
    int transM = M.transpose;
    if (dim == 0)
    {
        transM = !transM;
    }

    Matrix<CUDAfloat> sumM("sumM", transM ? M.numcol : M.numrow, 1, 0);
    Matrix<CUDAfloat> ones("ones", transM ? M.numrow : M.numcol, 1, 0);
    ones.ones();

    cublasOperation_t cudaTransM = transM ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f, zero = 0.0f;
    CuBLASErrorCheck(
        cublasSgemv(M.handle, cudaTransM, M.numrow, M.numcol,
        &one, (float *) M.elements.get(), M.numrow, (float *) ones.elements.get(),
        1, &zero, (float *) sumM.elements.get(), 1)
    )

    if (dim == 0)
    {
        sumM.transpose = true;
    }
    return sumM;
}
ostream& operator <<(ostream& os, const Matrix<CUDAfloat>& M) {
    using namespace std;
    Matrix<float> hostM = M.to_host();
    // write obj to stream
    os << hostM.get_name() << " " << hostM.num_row() << " by " << hostM.num_col();
    for (size_t i = 0; i < hostM.num_row(); i++)
    {
        os << endl;
        for (size_t j = 0; j < hostM.num_col(); j++)
        {
            os << hostM.elem(i, j) << " ";
        }
    }
    return os;
}

curandStatus_t GPUSampler::stats = CURAND_STATUS_NOT_INITIALIZED;
curandGenerator_t GPUSampler::gen = NULL;


Matrix<CUDAfloat> Matrix<CUDAfloat>::randn(size_t m, size_t n) {
    STATIC_TIC;
    auto& gen = GPUSampler::gen;
    Matrix<CUDAfloat> M("randn", m, n);

    if (m * n % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (m * n + 1) * sizeof(float));
        curandStatus_t curand_stat = curandGenerateNormal(gen, p, m * n + 1, 0.0f, 1.0f);
        if (curand_stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("sample random nums failed, {}.", curand_stat);
            ERROR_OUT;
        }
        cudaError_t stat = cudaMemcpy((float *) M.elements.get(), p, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory copy failed, {}.", stat);
            ERROR_OUT;
        }
        cudaFree(p);
    }
    else {
        // generate normal 
        auto stat = curandGenerateNormal(gen, (float *) M.elements.get(), M.numcol * M.numrow, 0.0f, 1.0f);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", stat);
            ERROR_OUT;
        }
    }
    STATIC_TOC;
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::rand(size_t m, size_t n) {
    static Profiler profiler("rand gen"); profiler.start();
    auto& gen = GPUSampler::gen;
    Matrix<CUDAfloat> M("randn", m, n);

    if (m * n % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (m * n + 1) * sizeof(float));
        curandStatus_t curand_stat = curandGenerateUniform(gen, p, m * n + 1);
        if (curand_stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("sample random nums failed, {}.", curand_stat);
            ERROR_OUT;
        }
        cudaError_t stat = cudaMemcpy((float *) M.elements.get(), p, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory copy failed, {}.", stat);
            ERROR_OUT;
        }
        cudaFree(p);
    }
    else {
        // generate normal 
        auto stat = curandGenerateUniform(gen, (float *) M.elements.get(), M.numcol * M.numrow);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", stat);
            ERROR_OUT;
        }
    }
    profiler.end();
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::ones(size_t m, size_t n)
{
    STATIC_TIC;
    Matrix<CUDAfloat> M("ones", m, n); fill(M, 1.0f);
    STATIC_TOC;
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::zeros(size_t m, size_t n)
{
    Matrix<CUDAfloat> M("zeros", m, n);
    // cudaError_t stat = cudaMemset((float *) M.elements.get(), 0, M.numcol * M.numrow * sizeof(float));
    // if (stat != cudaSuccess) {
    //     LOG_ERROR("device memory set zero failed, {}.", cudaGetErrorString(stat));
    // }
    return M;
}
