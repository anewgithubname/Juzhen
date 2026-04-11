/**
 * @file hipmatrix.cpp
 * @brief ROCm/HIP vertical-slice matrix ops (GEMM).
 */

#include "hipmatrix.cuh"
#include "hipbackend.hpp"
#include "helper.hpp"

#include <algorithm>
#include <cstring>

#ifdef ROCM_HIP

#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#define JUZHEN_ROCM_BLAS_AVAILABLE 1
#else
#define JUZHEN_ROCM_BLAS_AVAILABLE 0
#endif

#if JUZHEN_ROCM_BLAS_AVAILABLE
namespace {

hipblasHandle_t GlobalHipblasHandle() {
    static hipblasHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        auto st = hipblasCreate(&handle);
        if (st != HIPBLAS_STATUS_SUCCESS) {
            LOG_ERROR("hipblasCreate failed with code {}", static_cast<int>(st));
            return nullptr;
        }
        initialized = true;
    }
    return handle;
}

}  // namespace
#endif

Matrix<ROCMfloat>::Matrix(const char* name, size_t numrow, size_t numcol, int trans,
                          std::shared_ptr<ROCMfloat[]> elements) {
    this->name = name;
    this->numrow = numrow;
    this->numcol = numcol;
    this->transpose = trans;
    this->elements = elements;
}

Matrix<ROCMfloat>::Matrix(const char* name, size_t numrow, size_t numcol, int trans) {
    this->name = name;
    this->numrow = numrow;
    this->numcol = numcol;
    this->transpose = trans;
    elements.reset(Memory<ROCMfloat>::allocate(numrow * numcol),
                   [](ROCMfloat* p) { Memory<ROCMfloat>::free(p); });
}

Matrix<ROCMfloat>::Matrix(const Matrix<float>& M)
    : Matrix<ROCMfloat>(M.get_name().c_str(), M.num_row(), M.num_col(), M.get_transpose()) {
    Juzhen::RocmMemcpyH2D(reinterpret_cast<float*>(elements.get()), M.data(), M.num_row() * M.num_col());
}

Matrix<ROCMfloat>::Matrix(const Matrix<ROCMfloat>& M)
    : Matrix<ROCMfloat>(M.name.c_str(), M.numrow, M.numcol, M.transpose) {
    Juzhen::RocmMemcpyD2D(reinterpret_cast<float*>(elements.get()),
                          reinterpret_cast<const float*>(M.elements.get()),
                          M.numrow * M.numcol);
}

Matrix<ROCMfloat>::Matrix(Matrix<ROCMfloat>&& M) noexcept {
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
}

Matrix<ROCMfloat>& Matrix<ROCMfloat>::operator=(const Matrix<ROCMfloat>& M) {
    if (this == &M) return *this;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements.reset(Memory<ROCMfloat>::allocate(numrow * numcol),
                   [](ROCMfloat* p) { Memory<ROCMfloat>::free(p); });
    Juzhen::RocmMemcpyD2D(reinterpret_cast<float*>(elements.get()),
                          reinterpret_cast<const float*>(M.elements.get()),
                          M.numrow * M.numcol);
    return *this;
}

Matrix<ROCMfloat>& Matrix<ROCMfloat>::operator=(Matrix<ROCMfloat>&& M) noexcept {
    if (this == &M) return *this;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
    return *this;
}

void Matrix<ROCMfloat>::ones() { Juzhen::RocmFill(reinterpret_cast<float*>(elements.get()), num_row() * num_col(), 1.0f); }

void Matrix<ROCMfloat>::zeros() { Juzhen::RocmFill(reinterpret_cast<float*>(elements.get()), num_row() * num_col(), 0.0f); }

Matrix<ROCMfloat> Matrix<ROCMfloat>::randn(size_t m, size_t n) {
    Matrix<ROCMfloat> M("randn", m, n);
    Juzhen::RocmRandNormal(reinterpret_cast<float*>(M.elements.get()), m * n, 0.0f, 1.0f, 0);
    return M;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::rand(size_t m, size_t n) {
    Matrix<ROCMfloat> M("rand", m, n);
    Juzhen::RocmRandUniform(reinterpret_cast<float*>(M.elements.get()), m * n, 0);
    return M;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::ones(size_t m, size_t n) {
    Matrix<ROCMfloat> M("ones", m, n);
    M.ones();
    return M;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::zeros(size_t m, size_t n) {
    Matrix<ROCMfloat> M("zeros", m, n);
    M.zeros();
    return M;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::dot(const Matrix<ROCMfloat>& B) const {
    Matrix<ROCMfloat> C("dot", num_row(), B.num_col());
    int rc = Juzhen::RocmGemm(reinterpret_cast<const float*>(elements.get()),
                             reinterpret_cast<const float*>(B.elements.get()),
                             reinterpret_cast<float*>(C.elements.get()),
                             static_cast<int>(num_row()),
                             static_cast<int>(B.num_col()),
                             static_cast<int>(num_col()),
                             transpose,
                             B.transpose,
                             static_cast<int>(numrow),
                             static_cast<int>(B.numrow),
                             static_cast<int>(C.numrow));
    if (rc != 0) {
        LOG_ERROR("RocmGemm failed with code {}", rc);
        ERROR_OUT;
    }
    return C;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::add(const Matrix<ROCMfloat>& B, float s1, float s2) const {
    if (num_row() != B.num_row() || num_col() != B.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    Matrix<ROCMfloat> out("add", numrow, numcol, transpose);
    int rc = Juzhen::RocmAxpby(reinterpret_cast<float*>(out.elements.get()),
                              reinterpret_cast<const float*>(elements.get()),
                              reinterpret_cast<const float*>(B.elements.get()),
                              num_row() * num_col(),
                              s1,
                              s2);
    if (rc != 0) {
        LOG_ERROR("RocmAxpby failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

void Matrix<ROCMfloat>::add(const Matrix<ROCMfloat>& B, float s1, float s2) {
    if (num_row() != B.num_row() || num_col() != B.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    int rc = Juzhen::RocmAxpby(reinterpret_cast<float*>(elements.get()),
                               reinterpret_cast<const float*>(elements.get()),
                               reinterpret_cast<const float*>(B.elements.get()),
                               num_row() * num_col(),
                               s1,
                               s2);
    if (rc != 0) {
        LOG_ERROR("RocmAxpby failed with code {}", rc);
        ERROR_OUT;
    }
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::add(float a, float s1) const {
    Matrix<ROCMfloat> out(*this);
    int rc = Juzhen::RocmAffineInplace(reinterpret_cast<float*>(out.elements.get()),
                                      out.num_row() * out.num_col(),
                                      s1,
                                      a);
    if (rc != 0) {
        LOG_ERROR("RocmAffineInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

void Matrix<ROCMfloat>::add(float a, float s1) {
    int rc = Juzhen::RocmAffineInplace(reinterpret_cast<float*>(elements.get()),
                                      num_row() * num_col(),
                                      s1,
                                      a);
    if (rc != 0) {
        LOG_ERROR("RocmAffineInplace failed with code {}", rc);
        ERROR_OUT;
    }
}

void Matrix<ROCMfloat>::scale(float s1) { add(0.0f, s1); }

void Matrix<ROCMfloat>::eleminv(double l) {
    int rc = Juzhen::RocmElemInvInplace(reinterpret_cast<float*>(elements.get()),
                                       numrow * numcol,
                                       static_cast<float>(l));
    if (rc != 0) {
        LOG_ERROR("RocmElemInvInplace failed with code {}", rc);
        ERROR_OUT;
    }
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::eleminv(double l) const {
    Matrix<ROCMfloat> out(*this);
    out.eleminv(l);
    return out;
}

float Matrix<ROCMfloat>::norm() const {
#if JUZHEN_ROCM_BLAS_AVAILABLE
    hipblasHandle_t handle = GlobalHipblasHandle();
    if (handle == nullptr) {
        LOG_ERROR("hipBLAS handle unavailable for ROCm norm");
        ERROR_OUT;
    }

    float result = 0.0f;
    auto st = hipblasSnrm2(handle,
                           static_cast<int>(numrow * numcol),
                           reinterpret_cast<const float*>(elements.get()),
                           1,
                           &result);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        LOG_ERROR("hipblasSnrm2 failed with code {}", static_cast<int>(st));
        ERROR_OUT;
    }
    return result;
#else
    return to_host().norm();
#endif
}

const Matrix<ROCMfloat> Matrix<ROCMfloat>::T() const {
    std::string newname = name + "_T";
    return Matrix<ROCMfloat>(newname.c_str(), numrow, numcol, !transpose, elements);
}

Matrix<float> Matrix<ROCMfloat>::to_host() const {
    Matrix<float> H(name.c_str(), numrow, numcol);
    H.transpose = transpose;
    Juzhen::RocmMemcpyD2H(H.elements.get(), reinterpret_cast<const float*>(elements.get()), numrow * numcol);
    return H;
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::slice(size_t rstart, size_t rend, size_t cstart, size_t cend) const {
    if (rstart > rend || cstart > cend || rend > num_row() || cend > num_col()) {
        throw std::out_of_range("slice indices out of range");
    }

    const size_t rows = rend - rstart;
    const size_t cols = cend - cstart;
    Matrix<ROCMfloat> out("slice", rows, cols, 0);

    int rc = Juzhen::RocmSliceExtract(reinterpret_cast<float*>(out.elements.get()),
                                      out.numrow,
                                      reinterpret_cast<const float*>(elements.get()),
                                      numrow,
                                      transpose,
                                      rstart,
                                      cstart,
                                      rows,
                                      cols);
    if (rc != 0) {
        LOG_ERROR("RocmSliceExtract failed with code {}", rc);
        ERROR_OUT;
    }

    return out;
}

void Matrix<ROCMfloat>::slice(size_t rstart, size_t rend, size_t cstart, size_t cend,
                              const Matrix<ROCMfloat>& M) {
    if (rstart > rend || cstart > cend || rend > num_row() || cend > num_col()) {
        throw std::out_of_range("slice indices out of range");
    }

    const size_t rows = rend - rstart;
    const size_t cols = cend - cstart;
    if (M.num_row() != rows || M.num_col() != cols) {
        throw std::invalid_argument("slice assignment matrix dimensions do not match target range");
    }

    int rc = Juzhen::RocmSliceAssign(reinterpret_cast<float*>(elements.get()),
                                     numrow,
                                     transpose,
                                     rstart,
                                     cstart,
                                     reinterpret_cast<const float*>(M.elements.get()),
                                     M.numrow,
                                     M.transpose,
                                     rows,
                                     cols);
    if (rc != 0) {
        LOG_ERROR("RocmSliceAssign failed with code {}", rc);
        ERROR_OUT;
    }
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::rows(size_t rstart, size_t rend) const {
    return slice(rstart, rend, 0, num_col());
}

void Matrix<ROCMfloat>::rows(size_t rstart, size_t rend, const Matrix<ROCMfloat>& M) {
    slice(rstart, rend, 0, num_col(), M);
}

Matrix<ROCMfloat> Matrix<ROCMfloat>::columns(size_t cstart, size_t cend) const {
    return slice(0, num_row(), cstart, cend);
}

void Matrix<ROCMfloat>::columns(size_t cstart, size_t cend, const Matrix<ROCMfloat>& M) {
    slice(0, num_row(), cstart, cend, M);
}

Matrix<ROCMfloat> exp(const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> out(M);
    Juzhen::RocmExpInplace(reinterpret_cast<float*>(out.elements.get()), out.num_row() * out.num_col());
    return out;
}

Matrix<ROCMfloat> exp(Matrix<ROCMfloat>&& M) {
    Juzhen::RocmExpInplace(reinterpret_cast<float*>(M.elements.get()), M.num_row() * M.num_col());
    return std::move(M);
}

Matrix<ROCMfloat> log(const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> out(M);
    Juzhen::RocmLogInplace(reinterpret_cast<float*>(out.elements.get()), out.num_row() * out.num_col());
    return out;
}

Matrix<ROCMfloat> tanh(const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> out("tanhM", M.numrow, M.numcol, M.transpose);
    int rc = Juzhen::RocmTanh(reinterpret_cast<float*>(out.elements.get()),
                              reinterpret_cast<const float*>(M.elements.get()),
                              M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmTanh failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

Matrix<ROCMfloat> tanh(Matrix<ROCMfloat>&& M) {
    int rc = Juzhen::RocmTanhInplace(reinterpret_cast<float*>(M.elements.get()),
                                     M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmTanhInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return std::move(M);
}

Matrix<ROCMfloat> d_tanh(const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> out("d_tanhM", M.numrow, M.numcol, M.transpose);
    int rc = Juzhen::RocmDTanh(reinterpret_cast<float*>(out.elements.get()),
                               reinterpret_cast<const float*>(M.elements.get()),
                               M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmDTanh failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

Matrix<ROCMfloat> d_tanh(Matrix<ROCMfloat>&& M) {
    int rc = Juzhen::RocmDTanhInplace(reinterpret_cast<float*>(M.elements.get()),
                                      M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmDTanhInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return std::move(M);
}

Matrix<ROCMfloat> square(const Matrix<ROCMfloat>& M) {
    Matrix<ROCMfloat> out("square", M.numrow, M.numcol, M.transpose);
    int rc = Juzhen::RocmSquare(reinterpret_cast<float*>(out.elements.get()),
                                reinterpret_cast<const float*>(M.elements.get()),
                                M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmSquare failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

Matrix<ROCMfloat> square(Matrix<ROCMfloat>&& M) {
    int rc = Juzhen::RocmSquareInplace(reinterpret_cast<float*>(M.elements.get()),
                                       M.num_row() * M.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmSquareInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return std::move(M);
}

Matrix<ROCMfloat>& fill(Matrix<ROCMfloat>& M, double a) {
    int rc = Juzhen::RocmFill(reinterpret_cast<float*>(M.elements.get()),
                              M.num_row() * M.num_col(),
                              static_cast<float>(a));
    if (rc != 0) {
        LOG_ERROR("RocmFill failed with code {}", rc);
        ERROR_OUT;
    }
    return M;
}

void copy(Matrix<ROCMfloat>& dest, const Matrix<ROCMfloat>& src) {
    dest.numrow = src.numrow;
    dest.numcol = src.numcol;
    dest.transpose = src.transpose;
    dest.elements.reset(Memory<ROCMfloat>::allocate(src.numrow * src.numcol),
                        [](ROCMfloat* p) { Memory<ROCMfloat>::free(p); });
    int rc = Juzhen::RocmCopy(reinterpret_cast<float*>(dest.elements.get()),
                              reinterpret_cast<const float*>(src.elements.get()),
                              src.num_row() * src.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmCopy failed with code {}", rc);
        ERROR_OUT;
    }
}

Matrix<ROCMfloat> hstack(std::vector<MatrixView<ROCMfloat>> matrices) {
    auto t = std::remove_if(matrices.begin(), matrices.end(),
                            [](const MatrixView<ROCMfloat>& m) {
                                return m.num_row() == 0 || m.num_col() == 0;
                            });
    matrices.erase(t, matrices.end());

    if (matrices.empty()) {
        throw std::invalid_argument("hstack: input list is empty or contains only empty matrices");
    }

    size_t num_row = matrices[0].num_row();
    size_t num_col = 0;
    for (size_t i = 0; i < matrices.size(); i++) {
        if (matrices[i].num_row() != num_row) {
            throw std::invalid_argument("hstack: all matrices must have the same row count");
        }
        num_col += matrices[i].num_col();
    }

    Matrix<ROCMfloat> result("hstack", num_row, num_col, 0);

    size_t col_index = 0;
    for (const auto& mv : matrices) {
        int rc = Juzhen::RocmStackCopyCols(reinterpret_cast<float*>(result.elements.get()),
                                           num_row,
                                           col_index,
                                           reinterpret_cast<const float*>(mv.data()),
                                           mv.get_transpose() ? mv.num_col() : mv.num_row(),
                                           mv.get_transpose() ? mv.num_row() : mv.num_col(),
                                           mv.get_transpose() != 0);
        if (rc != 0) {
            LOG_ERROR("RocmStackCopyCols failed with code {}", rc);
            ERROR_OUT;
        }
        col_index += mv.num_col();
    }

    return result;
}

Matrix<ROCMfloat> vstack(std::vector<MatrixView<ROCMfloat>> matrices) {
    auto t = std::remove_if(matrices.begin(), matrices.end(),
                            [](const MatrixView<ROCMfloat>& m) {
                                return m.num_row() == 0 || m.num_col() == 0;
                            });
    matrices.erase(t, matrices.end());

    if (matrices.empty()) {
        throw std::invalid_argument("vstack: input list is empty or contains only empty matrices");
    }

    size_t num_row = 0;
    size_t num_col = matrices[0].num_col();
    for (size_t i = 0; i < matrices.size(); i++) {
        if (matrices[i].num_col() != num_col) {
            throw std::invalid_argument("vstack: all matrices must have the same column count");
        }
        num_row += matrices[i].num_row();
    }

    Matrix<ROCMfloat> result("vstack", num_row, num_col, 0);

    size_t row_index = 0;
    for (const auto& mv : matrices) {
        int rc = Juzhen::RocmStackCopyRows(reinterpret_cast<float*>(result.elements.get()),
                                           num_row,
                                           row_index,
                                           reinterpret_cast<const float*>(mv.data()),
                                           mv.get_transpose() ? mv.num_col() : mv.num_row(),
                                           mv.get_transpose() ? mv.num_row() : mv.num_col(),
                                           mv.get_transpose() != 0);
        if (rc != 0) {
            LOG_ERROR("RocmStackCopyRows failed with code {}", rc);
            ERROR_OUT;
        }
        row_index += mv.num_row();
    }

    return result;
}

Matrix<ROCMfloat> sum(const Matrix<ROCMfloat>& M, int dim) {
    CBLAS_TRANSPOSE transM = M.transpose ? CblasTrans : CblasNoTrans;
    if (dim == 0) {
        transM = (transM == CblasTrans) ? CblasNoTrans : CblasTrans;
    }

    Matrix<ROCMfloat> sumM("sumM", transM == CblasTrans ? M.numcol : M.numrow, 1, 0);
    Matrix<ROCMfloat> ones("ones", transM == CblasTrans ? M.numrow : M.numcol, 1, 0);
    ones.ones();

    gemv(transM,
         static_cast<int>(M.numrow),
         static_cast<int>(M.numcol),
         1.0f,
         const_cast<ROCMfloat*>(M.elements.get()),
         static_cast<int>(M.numrow),
         ones.elements.get(),
         1,
         0.0f,
         sumM.elements.get(),
         1);

    if (dim == 0) {
        sumM.transpose = 1;
    }
    return sumM;
}

Matrix<ROCMfloat> hadmd(const Matrix<ROCMfloat>& M1, const Matrix<ROCMfloat>& M2) {
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<ROCMfloat> out("hadmd", M1.numrow, M1.numcol, M1.transpose);
    int rc = Juzhen::RocmHadamard(reinterpret_cast<float*>(out.elements.get()),
                                 reinterpret_cast<const float*>(M1.elements.get()),
                                 reinterpret_cast<const float*>(M2.elements.get()),
                                 M1.num_row() * M1.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmHadamard failed with code {}", rc);
        ERROR_OUT;
    }
    return out;
}

Matrix<ROCMfloat> hadmd(const Matrix<ROCMfloat>& M1, Matrix<ROCMfloat>&& M2) {
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    int rc = Juzhen::RocmHadamardInplace(reinterpret_cast<float*>(M2.elements.get()),
                                        reinterpret_cast<const float*>(M1.elements.get()),
                                        M1.num_row() * M1.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmHadamardInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return std::move(M2);
}

Matrix<ROCMfloat> hadmd(Matrix<ROCMfloat>&& M1, const Matrix<ROCMfloat>& M2) {
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    int rc = Juzhen::RocmHadamardInplace(reinterpret_cast<float*>(M1.elements.get()),
                                        reinterpret_cast<const float*>(M2.elements.get()),
                                        M1.num_row() * M1.num_col());
    if (rc != 0) {
        LOG_ERROR("RocmHadamardInplace failed with code {}", rc);
        ERROR_OUT;
    }
    return std::move(M1);
}

Matrix<ROCMfloat> hadmd(Matrix<ROCMfloat>&& M1, Matrix<ROCMfloat>&& M2) {
    return hadmd(M1, std::move(M2));
}

int Juzhen::RocmGemm(const float* A_device, const float* B_device, float* C_device,
                     int m, int n, int k,
                     bool transA, bool transB,
                     int lda, int ldb, int ldc,
                     float alpha, float beta) {
#if JUZHEN_ROCM_BLAS_AVAILABLE
    if (m <= 0 || n <= 0 || k <= 0) return -1;
    if (A_device == nullptr || B_device == nullptr || C_device == nullptr) return -1;

    hipblasHandle_t handle = GlobalHipblasHandle();
    if (handle == nullptr) return -1;

    hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    auto st = hipblasSgemm(handle,
                      opA,
                      opB,
                      m,
                      n,
                      k,
                      &alpha,
                      A_device,
                      lda,
                      B_device,
                      ldb,
                      &beta,
                      C_device,
                      ldc);
    return st == HIPBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(st);
#else
    (void)A_device;
    (void)B_device;
    (void)C_device;
    (void)m;
    (void)n;
    (void)k;
    (void)transA;
    (void)transB;
    (void)lda;
    (void)ldb;
    (void)ldc;
    (void)alpha;
    (void)beta;
    LOG_ERROR("hipBLAS headers not found; RocmGemm is unavailable.");
    return -1;
#endif
}

int Juzhen::RocmGemmNN(const float* A_device, const float* B_device, float* C_device,
                       int m, int n, int k, float alpha, float beta) {
    return Juzhen::RocmGemm(A_device, B_device, C_device,
                            m, n, k,
                            false, false,
                            m, k, m,
                            alpha, beta);
}

int Juzhen::RocmGemv(const float* A_device, const float* x_device, float* y_device,
                     int m, int n,
                     bool transA,
                     int lda,
                     int incx,
                     int incy,
                     float alpha,
                     float beta) {
#if JUZHEN_ROCM_BLAS_AVAILABLE
    if (m <= 0 || n <= 0) return -1;
    if (A_device == nullptr || x_device == nullptr || y_device == nullptr) return -1;
    if (incx <= 0 || incy <= 0) return -1;

    hipblasHandle_t handle = GlobalHipblasHandle();
    if (handle == nullptr) return -1;

    hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto st = hipblasSgemv(handle,
                           opA,
                           m,
                           n,
                           &alpha,
                           A_device,
                           lda,
                           x_device,
                           incx,
                           &beta,
                           y_device,
                           incy);
    return st == HIPBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(st);
#else
    (void)A_device;
    (void)x_device;
    (void)y_device;
    (void)m;
    (void)n;
    (void)transA;
    (void)lda;
    (void)incx;
    (void)incy;
    (void)alpha;
    (void)beta;
    LOG_ERROR("hipBLAS headers not found; RocmGemv is unavailable.");
    return -1;
#endif
}

std::ostream& operator<<(std::ostream& os, const Matrix<ROCMfloat>& M) {
    os << M.to_host();
    return os;
}

template <>
void write(FILE* fp, const Matrix<ROCMfloat>& M) {
    write(fp, M.to_host());
}

template <>
void read(FILE* fp, Matrix<ROCMfloat>& M) {
    Matrix<float> tmp("tmp", M.num_row(), M.num_col());
    read(fp, tmp);
    M.numrow = tmp.numrow;
    M.numcol = tmp.numcol;
    M.transpose = tmp.transpose;
    Juzhen::RocmMemcpyH2D(reinterpret_cast<float*>(M.elements.get()), tmp.elements.get(),
                          tmp.num_row() * tmp.num_col());
}

#endif
