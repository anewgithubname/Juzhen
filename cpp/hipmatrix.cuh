/**
 * @file hipmatrix.cuh
 * @brief Matrix<ROCMfloat> specialization (phase-2 vertical slice).
 */

#ifndef HIPMATRIX_CUH
#define HIPMATRIX_CUH

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include "core.hpp"
#include "hipbackend.hpp"
#include "matrix.hpp"
#include "operators.hpp"
#include <vector>

#define rocmThreadsPerBlock 1024
#define rocmConfig(numElem) \
    ((unsigned int)(numElem) + rocmThreadsPerBlock - 1) / rocmThreadsPerBlock, \
        rocmThreadsPerBlock

template <>
inline ROCMfloat* Memory<ROCMfloat>::_alloc(size_t size) {
    if (size == 0) size = 1;
    float* ptr = nullptr;
    int rc = Juzhen::RocmMalloc(&ptr, size);
    if (rc != 0 || ptr == nullptr) {
        throw std::bad_alloc();
    }
    return reinterpret_cast<ROCMfloat*>(ptr);
}

template <>
inline void Memory<ROCMfloat>::_free(ROCMfloat* ptr) {
    Juzhen::RocmFree(reinterpret_cast<float*>(ptr));
}

template <>
class Matrix<ROCMfloat> {
    size_t numcol;
    size_t numrow;
    bool transpose;
    std::string name;

    std::shared_ptr<ROCMfloat[]> elements;

    Matrix<ROCMfloat>(const char* name, size_t numrow, size_t numcol, int trans,
                      std::shared_ptr<ROCMfloat[]> elements);
    Matrix<ROCMfloat>(const char* name, size_t numrow, size_t numcol, int trans);

   public:
    explicit Matrix<ROCMfloat>(const Matrix<float>& M);
    Matrix<ROCMfloat>(const char* name, size_t numrow, size_t numcol)
        : Matrix<ROCMfloat>(name, numrow, numcol, 0) {}
    Matrix<ROCMfloat>() : Matrix<ROCMfloat>("un_init", 2, 2, 0) {}

    Matrix<ROCMfloat>(const Matrix<ROCMfloat>& M);
    Matrix<ROCMfloat>(Matrix<ROCMfloat>&& M) noexcept;
    Matrix<ROCMfloat>& operator=(const Matrix<ROCMfloat>& M);
    Matrix<ROCMfloat>& operator=(Matrix<ROCMfloat>&& M) noexcept;

    inline size_t idx(size_t i, size_t j) const {
        return transpose ? i * numrow + j : j * numrow + i;
    }

    inline ROCMfloat elem(size_t i, size_t j) const { return elements[idx(i, j)]; }
    inline ROCMfloat& elem(size_t i, size_t j) { return elements[idx(i, j)]; }

    inline size_t num_col() const { return transpose ? numrow : numcol; }
    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline size_t get_transpose() const { return transpose; }
    std::string get_name() const { return name; }
    const ROCMfloat* data() const { return elements.get(); }

    void ones();
    void zeros();

    static Matrix<ROCMfloat> randn(size_t m, size_t n);
    static Matrix<ROCMfloat> rand(size_t m, size_t n);
    static Matrix<ROCMfloat> ones(size_t m, size_t n);
    static Matrix<ROCMfloat> zeros(size_t m, size_t n);

    Matrix<ROCMfloat> dot(const Matrix<ROCMfloat>& B) const;

    Matrix<ROCMfloat> add(const Matrix<ROCMfloat>& B, float s1, float s2) const;
    void add(const Matrix<ROCMfloat>& B, float s1, float s2);

    Matrix<ROCMfloat> add(float a, float s1) const;
    void add(float a, float s1);

    Matrix<ROCMfloat> scale(float s1) const { return add(0, s1); }
    void scale(float s1);

    void eleminv(double l);
    Matrix<ROCMfloat> eleminv(double l) const;

    float norm() const;
    const Matrix<ROCMfloat> T() const;

    Matrix<float> to_host() const;

    Matrix<ROCMfloat> slice(size_t rstart, size_t rend, size_t cstart, size_t cend) const;
    void slice(size_t rstart, size_t rend, size_t cstart, size_t cend, const Matrix<ROCMfloat>& M);

    Matrix<ROCMfloat> rows(size_t rstart, size_t rend) const;
    void rows(size_t rstart, size_t rend, const Matrix<ROCMfloat>& M);
    Matrix<ROCMfloat> columns(size_t cstart, size_t cend) const;
    void columns(size_t cstart, size_t cend, const Matrix<ROCMfloat>& M);

    friend Matrix<ROCMfloat> sum(const Matrix<ROCMfloat>& M, int dim);
    friend Matrix<ROCMfloat> hadmd(const Matrix<ROCMfloat>& M1,
                                   const Matrix<ROCMfloat>& M2);
    friend Matrix<ROCMfloat> hadmd(const Matrix<ROCMfloat>& M1,
                                   Matrix<ROCMfloat>&& M2);
    friend Matrix<ROCMfloat> hadmd(Matrix<ROCMfloat>&& M1,
                                   const Matrix<ROCMfloat>& M2);
    friend Matrix<ROCMfloat> hadmd(Matrix<ROCMfloat>&& M1,
                                   Matrix<ROCMfloat>&& M2);
    friend Matrix<ROCMfloat> exp(const Matrix<ROCMfloat>& M);
    friend Matrix<ROCMfloat> exp(Matrix<ROCMfloat>&& M);
    friend Matrix<ROCMfloat> log(const Matrix<ROCMfloat>& M);
    friend Matrix<ROCMfloat> tanh(const Matrix<ROCMfloat>& M);
    friend Matrix<ROCMfloat> tanh(Matrix<ROCMfloat>&& M);
    friend Matrix<ROCMfloat> d_tanh(const Matrix<ROCMfloat>& M);
    friend Matrix<ROCMfloat> d_tanh(Matrix<ROCMfloat>&& M);
    friend Matrix<ROCMfloat> square(const Matrix<ROCMfloat>& M);
    friend Matrix<ROCMfloat> square(Matrix<ROCMfloat>&& M);
    friend void copy(Matrix<ROCMfloat>& dest, const Matrix<ROCMfloat>& src);
    friend Matrix<ROCMfloat>& fill(Matrix<ROCMfloat>& M, double a);
    friend class MatrixView<ROCMfloat>;
    friend Matrix<ROCMfloat> hstack(std::vector<MatrixView<ROCMfloat>> matrices);
    friend Matrix<ROCMfloat> vstack(std::vector<MatrixView<ROCMfloat>> matrices);

    template <class Function>
    friend Matrix<ROCMfloat> reduce(Function func, const Matrix<ROCMfloat>& M, int dim, int k);
    template <class Function>
    friend Matrix<ROCMfloat> elemwise(Function func, const Matrix<ROCMfloat>& M);
    template <class Function>
    friend Matrix<ROCMfloat> elemwise(Function func, Matrix<ROCMfloat>&& M);

    friend std::ostream& operator<<(std::ostream& os, const Matrix<ROCMfloat>& M);
    friend void write<ROCMfloat>(FILE* fp, const Matrix<ROCMfloat>& M);
    friend void read<ROCMfloat>(FILE* fp, Matrix<ROCMfloat>& M);
};

Matrix<ROCMfloat> sum(const Matrix<ROCMfloat>& M, int dim);
std::ostream& operator<<(std::ostream& os, const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> exp(const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> exp(Matrix<ROCMfloat>&& M);
Matrix<ROCMfloat> log(const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> tanh(const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> tanh(Matrix<ROCMfloat>&& M);
Matrix<ROCMfloat> d_tanh(const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> d_tanh(Matrix<ROCMfloat>&& M);
Matrix<ROCMfloat> square(const Matrix<ROCMfloat>& M);
Matrix<ROCMfloat> square(Matrix<ROCMfloat>&& M);
void copy(Matrix<ROCMfloat>& dest, const Matrix<ROCMfloat>& src);
Matrix<ROCMfloat>& fill(Matrix<ROCMfloat>& M, double a);
Matrix<ROCMfloat> hstack(std::vector<MatrixView<ROCMfloat>> matrices);
Matrix<ROCMfloat> vstack(std::vector<MatrixView<ROCMfloat>> matrices);

#ifdef __HIPCC__

template <class Function>
__global__ void rocm_reduce_kernel(Function func, float* vecdes, float* vec,
                                   size_t lenvec, size_t lenvecdes, size_t numvecs) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numvecs) {
        func(&vec[i * lenvec], &vecdes[i * lenvecdes], lenvec, lenvecdes);
    }
}

template <class Function>
Matrix<ROCMfloat> reduce(Function func, const Matrix<ROCMfloat>& M, int dim,
                         int k) {
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
        Matrix<ROCMfloat> result("resM", k, M.numcol);
        rocm_reduce_kernel<<<rocmConfig(M.numcol)>>>(
            func, (float*)result.elements.get(), (float*)M.elements.get(),
            M.numrow, k, M.numcol);
        if (M.transpose) {
            return result.T();
        } else {
            return result;
        }
    } else {
        Matrix<ROCMfloat> t("tzeros", M.numcol, M.numrow);
        t += M.transpose ? M : M.T();

        Matrix<ROCMfloat> result("resM", k, t.numcol);
        rocm_reduce_kernel<<<rocmConfig(t.numcol)>>>(
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
__global__ void rocm_elemwise_kernel(Function func, float* vecdes, float* vec,
                                     size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        vecdes[i] = func(vec[i]);
    }
}

template <class Function>
__global__ void rocm_inplace_elemwise_kernel(Function func, float* vecdes,
                                             size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        vecdes[i] = func(vecdes[i]);
    }
}

template <class Function>
Matrix<ROCMfloat> elemwise(Function func, const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> result("resM", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col();
    rocm_elemwise_kernel<<<rocmConfig(numElem)>>>(
        func, (float*)result.elements.get(), (float*)M.elements.get(), numElem);
    return result;
}

template <class Function>
Matrix<ROCMfloat> elemwise(Function func, Matrix<ROCMfloat>&& M) {
    size_t numElem = M.num_row() * M.num_col();
    rocm_inplace_elemwise_kernel<<<rocmConfig(numElem)>>>(
        func, (float*)M.elements.get(), numElem);
    return std::move(M);
}

#endif // __HIPCC__

#endif
