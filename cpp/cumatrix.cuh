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

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "core.hpp"
#include "matrix.hpp"
#include "operators.hpp"

#pragma region ERROR_HANDLING
#define CudaErrorCheck(ans) \
    { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        LOG_ERROR("CUDA ERROR: {} {}:{}", cudaGetErrorString(code), file, line);
        ERROR_OUT;
    }
}

#define CuBLASErrorCheck(ans) \
    { CuBLASAssert((ans), __FUNCTION__, __FILE__, __LINE__); }
inline void CuBLASAssert(cublasStatus_t code, const char* func,
                         const char* file, int line, bool abort = true) {
    if (code != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("CUBLAS ERROR: {}, {}, {}:{}", cublasGetStatusString(code),
                  func, file, line);
        ERROR_OUT;
    } else {
        LOG_DEBUG("CUBLAS OK, {}, {}:{}", func, file, line);
    }
}

#pragma endregion

#define threadsPerBlock 1024
#define cudaConfig(numElem)                                          \
    ((unsigned int)numElem + threadsPerBlock - 1) / threadsPerBlock, \
        threadsPerBlock

typedef cublasHandle_t GPU_handle;

// allocate CUDA memory
template <>
inline CUDAfloat* Memory<CUDAfloat>::_alloc(size_t size) {
    STATIC_TIC;
    // if size = 0, allocate 1 element, this is to avoid error when size = 0
    if(size == 0) size = 1; 
    CUDAfloat* ptr = NULL;
    cudaMalloc(&ptr, size * sizeof(CUDAfloat));
    //CudaErrorCheck(cudaMalloc(&ptr, size * sizeof(CUDAfloat)));
    if (ptr == NULL) throw std::bad_alloc();
    
    STATIC_TOC;
    return ptr;
}

// free CUDA memory
template <>
inline void Memory<CUDAfloat>::_free(CUDAfloat* ptr) {
    cudaFree(ptr);
}

__global__ void copyKernel(float* d_out, float* d_in, size_t numElements,
                           size_t numrow, size_t rowstart, size_t rowend,
                           size_t colstart, size_t colend, bool direction);
__global__ void copySliceAssignKernel(float* dst, const float* src,
                                       size_t rows, size_t cols,
                                       size_t dst_numrow, bool dst_transpose,
                                       size_t src_numrow, bool src_transpose,
                                       size_t rstart, size_t cstart);
__global__ void copyDexKernel(float* d_out, const float* d_in, const size_t* d_rowidx,
                               const size_t* d_colidx, size_t num_rows_out,
                               size_t num_cols_out, size_t numrow_in,
                               bool transposed);
__global__ void addKernel(float* d_out, float s1, float a, size_t numElements);

template <>
class Matrix<CUDAfloat> {
    size_t numcol;
    size_t numrow;
    bool transpose;
    std::string name;

    std::shared_ptr<CUDAfloat[]> elements;

    // NOTE: each matrix does have its own handle, however, for now,
    // they are all assigned to the global handle.
    GPU_handle handle;

    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol, int trans,
                      std::shared_ptr<CUDAfloat[]> elements) {
        this->name = name;
        this->numrow = numrow;
        this->numcol = numcol;
        this->transpose = trans;
        this->elements = elements;
        this->handle = global_handle;
    }

    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol,
                      int trans);
   public:
    static GPU_handle global_handle;

    // constructors
    explicit Matrix<CUDAfloat>(const Matrix<float>& M);
    Matrix<CUDAfloat>(const char* name, size_t numrow, size_t numcol)
        : Matrix<CUDAfloat>(name, numrow, numcol, 0){};
    // Empty state for non-affine layers: no device allocation or fill kernel.
    explicit Matrix<CUDAfloat>(std::nullptr_t)
        : numcol(0), numrow(0), transpose(false), name("empty"), elements(nullptr),
          handle(global_handle) {}
    Matrix<CUDAfloat>() : Matrix<CUDAfloat>("un_init", 2, 2, 0){};

    // copy and move constructors, assignment operators
    Matrix<CUDAfloat>(const Matrix<CUDAfloat>& M);
    Matrix<CUDAfloat>(Matrix<CUDAfloat>&& M) noexcept;
    Matrix<CUDAfloat>& operator=(const Matrix<CUDAfloat>& M);
    Matrix<CUDAfloat>& operator=(Matrix<CUDAfloat>&& M) noexcept;

    inline size_t idx(size_t i, size_t j) const {
        return transpose ? i * numrow + j : j * numrow + i;
    }
    // access matrix info
    inline CUDAfloat elem(size_t i, size_t j) const {
        return elements[idx(i, j)];
    }
    inline CUDAfloat& elem(size_t i, size_t j) { return elements[idx(i, j)]; }
    inline CUDAfloat operator()(size_t i, size_t j) const {
        return elements[idx(i, j)];
    }
    inline CUDAfloat& operator()(size_t i, size_t j) {
        return elements[idx(i, j)];
    }

    inline size_t num_col() const { return transpose ? numrow : numcol; }
    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline size_t get_transpose() const { return transpose; }
    std::string get_name() const { return name; }
    const CUDAfloat* data() const { return elements.get(); }

    // matrix filler
    void ones();
    void zeros();

    static Matrix<CUDAfloat> randn(size_t m, size_t n);
    static Matrix<CUDAfloat> rand(size_t m, size_t n);
    static Matrix<CUDAfloat> ones(size_t m, size_t n);
    static Matrix<CUDAfloat> zeros(size_t m, size_t n);

    // basic matrix ops
    Matrix<CUDAfloat> dot(const Matrix<CUDAfloat>& B) const;

    Matrix<CUDAfloat> add(const Matrix<CUDAfloat>& B, float s1, float s2) const;
    void add(const Matrix<CUDAfloat>& B, float s1, float s2);

    Matrix<CUDAfloat> add(float a, float s1) const;
    void add(float a, float s1);

    Matrix<CUDAfloat> scale(float s1) const { return add(0, s1); }
    void scale(float s1);

    void eleminv(double l);
    Matrix<CUDAfloat> eleminv(double l) const;

    float norm() const;
    const Matrix<CUDAfloat> T() const;

    // upload to host mem
    Matrix<float> to_host() const;

    ////slicing matrix
    // Matrix<float> slice(int rstart, int rend, int cstart, int cend) const {
    // return to_host().slice(rstart, rend, cstart, cend); } Matrix<float>
    // slice(const idxlist& rowidx, const idxlist& colidx) const { return
    // to_host().slice(rowidx, colidx); }
    Matrix<CUDAfloat> slice(size_t rstart, size_t rend, size_t cstart,
                            size_t cend) const {
        size_t numElem = (rend - rstart) * (cend - cstart);
        if (transpose) {
            size_t tmp = rstart;
            rstart = cstart;
            cstart = tmp;
            tmp = rend;
            rend = cend;
            cend = tmp;
        }
        Matrix<CUDAfloat> M("submatrix", rend - rstart, cend - cstart,
                            transpose);
        copyKernel<<<cudaConfig(numElem)>>>(
            (float*)M.elements.get(), (float*)elements.get(), numElem, numrow,
            rstart, rend, cstart, cend, true);
        return M;
    }

    void slice(size_t rstart, size_t rend, size_t cstart, size_t cend,
               const Matrix<CUDAfloat>& M) {
        if (rstart > rend || cstart > cend || rend > num_row() || cend > num_col())
            throw std::out_of_range("CUDA slice assignment indices out of range");
        const size_t rows = rend - rstart, cols = cend - cstart;
        if (M.num_row() != rows || M.num_col() != cols)
            throw std::invalid_argument("CUDA slice assignment dimensions do not match target range");
        const size_t limit = std::numeric_limits<size_t>::max();
        if ((cols && rows > limit / cols) || rows * cols > limit / sizeof(float))
            throw std::length_error("CUDA slice assignment size overflows");
        const size_t count = rows * cols;
        if (!count) return;
        // T() shares storage; preserve the source before any overlapping write.
        if (elements.get() == M.elements.get()) {
            Matrix<CUDAfloat> snapshot(M);
            slice(rstart, rend, cstart, cend, snapshot);
            return;
        }
        const unsigned blocks = unsigned(std::min(size_t(4096), (count - 1) / 256 + 1));
        copySliceAssignKernel<<<blocks, 256>>>(
            (float*)elements.get(), (const float*)M.elements.get(), rows, cols,
            numrow, transpose, M.numrow, M.transpose, rstart, cstart);
        CudaErrorCheck(cudaGetLastError());
    }

    Matrix<CUDAfloat> rows(size_t rstart, size_t rend) const {
        return slice(rstart, rend, 0, num_col());
    }
    void rows(size_t rstart, size_t rend, const Matrix<CUDAfloat>& M) {
        return slice(rstart, rend, 0, num_col(), M);
    }
    // Matrix<float> rows(idxlist rlist) const { return to_host().rows(rlist); }
    Matrix<CUDAfloat> rows(idxlist rlist) const {
        return slice(rlist, seq(num_col()));
    }
    Matrix<CUDAfloat> columns(size_t cstart, size_t cend) const {
        return slice(0, num_row(), cstart, cend);
    }
    void columns(size_t cstart, size_t cend, const Matrix<CUDAfloat>& M) {
        return slice(0, num_row(), cstart, cend, M);
    }
    // Matrix<float> columns(idxlist clist) const { return
    // to_host().columns(clist); }
    Matrix<CUDAfloat> columns(idxlist clist) const {
        return slice(seq(num_row()), clist);
    }

    Matrix<CUDAfloat> slice(const idxlist& rowidx,const idxlist& colidx) const {
        const size_t nr=rowidx.size(),nc=colidx.size();
        for(size_t row:rowidx)if(row>=num_row())throw std::out_of_range("CUDA gather row index");
        for(size_t col:colidx)if(col>=num_col())throw std::out_of_range("CUDA gather column index");
        const size_t limit=std::numeric_limits<size_t>::max();
        if((nc && nr>limit/nc) || nr*nc>limit/sizeof(float) || nr>limit/sizeof(size_t) || nc>limit/sizeof(size_t))
            throw std::length_error("CUDA gather size overflows");
        const size_t count=nr*nc;
        // No redundant memset, and no zero-block launch for empty selections.
        auto storage=std::shared_ptr<CUDAfloat[]>(Memory<CUDAfloat>::allocate(count),[](CUDAfloat* p){Memory<CUDAfloat>::free(p);});
        Matrix<CUDAfloat> result("submatrix",nr,nc,0,std::move(storage));
        if(!count)return result;
        auto indices=[](const idxlist& input){
            size_t* device=nullptr;CudaErrorCheck(cudaMalloc(&device,input.size()*sizeof(size_t)));
            std::shared_ptr<size_t> owner(device,[](size_t* p){if(p)CudaErrorCheck(cudaFree(p));});
            CudaErrorCheck(cudaMemcpy(device,input.data(),input.size()*sizeof(size_t),cudaMemcpyHostToDevice));
            return owner;
        };
        auto rows=indices(rowidx),cols=indices(colidx);
        const unsigned blocks=unsigned(std::min(size_t(4096),(count-1)/256+1));
        copyDexKernel<<<blocks,256>>>((float*)result.elements.get(),(const float*)elements.get(),rows.get(),cols.get(),nr,nc,numrow,transpose);
        CudaErrorCheck(cudaGetLastError());
        return result;
    }

    // our friends
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

    // columns operations
    template <class Function>
    friend Matrix<CUDAfloat> reduce(Function func, const Matrix<CUDAfloat>& M, int dim, int k);  
    template <class Function>
    // lvalue elementwise op
    friend Matrix<CUDAfloat> elemwise(Function func, const Matrix<CUDAfloat>& M);
    template <class Function>
    // rvalue elementwise op
    friend Matrix<CUDAfloat> elemwise(Function func, Matrix<CUDAfloat>&& M);

    friend void copy(Matrix<CUDAfloat>& dest, const Matrix<CUDAfloat>& src);
    friend Matrix<CUDAfloat>& fill(Matrix<CUDAfloat>& M, double a);
    friend class MatrixView<CUDAfloat>;
    friend Matrix<CUDAfloat> hstack(
        std::vector<MatrixView<CUDAfloat>> matrices);
    friend const Matrix<CUDAfloat> vstack(
        std::vector<MatrixView<CUDAfloat>> matrices);
    friend Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1,
                                   const Matrix<CUDAfloat>& M2);
    friend Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1,
                                   Matrix<CUDAfloat>&& M2);
    friend Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat>&& M1,
                                  const Matrix<CUDAfloat>& M2);
    
    friend void read<CUDAfloat>(FILE *fp, Matrix<CUDAfloat>& M);
};

struct GPUSampler {
    static curandStatus_t stats;
    static curandGenerator_t gen;

    // random sampler
    GPUSampler(int seed) {
        // random generator
        curandStatus_t stats =
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
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
        stats = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand set seed failed");
            ERROR_OUT;
        }
        // A reset starts at the beginning even after an earlier nonzero offset.
        stats = curandSetGeneratorOffset(gen, 0);
        if (stats != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand reset generator offset failed");
            ERROR_OUT;
        }
    }

    // deconstruct the sampler
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

// operators, math functions
Matrix<CUDAfloat> sum(const Matrix<CUDAfloat>& M, int dim);
std::ostream& operator<<(std::ostream& os, const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> exp(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> exp(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> log(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> tanh(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> tanh(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> d_tanh(Matrix<CUDAfloat>&& M);
Matrix<CUDAfloat> d_tanh(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> square(const Matrix<CUDAfloat>& M);
Matrix<CUDAfloat> square(Matrix<CUDAfloat>&& M);

template <class Function>
__global__ void reduce_kernel(Function func, float* vecdes, float* vec,
                              size_t lenvec, size_t lenvecdes, size_t numvecs) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numvecs) {
        func(&vec[i * lenvec], &vecdes[i * lenvecdes], lenvec, lenvecdes);
    }
}

template <class Function>
Matrix<CUDAfloat> reduce(Function func, const Matrix<CUDAfloat>& M, int dim,
                         int k) {
    // STATIC_TIC;
    bool trans = false;
    if (dim == 0 && !M.transpose) {
        trans = false;
    } else if (dim == 1 && !M.transpose) {
        trans = true;
    } else if (dim == 0 && M.transpose) {
        trans = true;
    } else {
        trans = false;
    }

    if (!trans) {
        Matrix<CUDAfloat> result("resM", k, M.numcol);
        reduce_kernel<<<cudaConfig(M.numcol)>>>(
            func, (float*)result.elements.get(), (float*)M.elements.get(),
            M.numrow, k, M.numcol);
        if (M.transpose) {
            return result.T();
        } else {
            return result;
        }
    } else {
        // transpose the matrix
        Matrix<CUDAfloat> t("tzeros", M.numcol, M.numrow);
        t += M.transpose ? M : M.T();

        Matrix<CUDAfloat> result("resM", k, t.numcol);
        reduce_kernel<<<cudaConfig(t.numcol)>>>(
            func, (float*)result.elements.get(), (float*)t.elements.get(),
            t.numrow, k, t.numcol);
        if (M.transpose) {
            return result;
        } else {
            return result.T();
        }
    }
}

template <class Function>
__global__ void elemwise_kernel(Function func, float* vecdes, float* vec,
                                size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = func(vec[i]);
    }
}

template <class Function>
__global__ void Inplace_elemwise_kernel(Function func, float* vecdes,
                                        size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = func(vecdes[i]);
    }
}

template <class Function>
Matrix<CUDAfloat> elemwise(Function func, const Matrix<CUDAfloat>& M) {
    // STATIC_TIC;

    Matrix<CUDAfloat> result("resM", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col();
    elemwise_kernel<<<cudaConfig(numElem)>>>(
        func, (float*)result.elements.get(), (float*)M.elements.get(), numElem);
    // TODO: check cuda kernel error

    // STATIC_TOC;
    return result;
}

template <class Function>
Matrix<CUDAfloat> elemwise(Function func, Matrix<CUDAfloat>&& M) {
    // STATIC_TIC;

    size_t numElem = M.num_row() * M.num_col();
    Inplace_elemwise_kernel<<<cudaConfig(numElem)>>>(
        func, (float*)M.elements.get(), numElem);
    // TODO: check cuda kernel error
    //  STATIC_TOC;
    return std::move(M);
}

Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1,
                        const Matrix<CUDAfloat>& M2);
Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat>& M1, Matrix<CUDAfloat>&& M2);
Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat>&& M1, const Matrix<CUDAfloat>& M2);
Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat>&& M1, Matrix<CUDAfloat>&& M2);
Matrix<CUDAfloat>& fill(Matrix<CUDAfloat>& M, double a);
void copy(Matrix<CUDAfloat>& dest, const Matrix<CUDAfloat>& src);
Matrix<CUDAfloat> hstack(std::vector<MatrixView<CUDAfloat>> matrices);
const Matrix<CUDAfloat> vstack(std::vector<MatrixView<CUDAfloat>> matrices);

// specialization, as they will be called by 
// fun(Matrix<D> M) where D is a template parameter

template <>
void write(FILE *fp, const Matrix<CUDAfloat>& M);
template <>
void read(FILE *fp, Matrix<CUDAfloat>& M);

#endif