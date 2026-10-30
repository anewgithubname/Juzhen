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

#ifndef MPSMATRIX_HPP
#define MPSMATRIX_HPP

#include "core.hpp"
#include "metal/MPSWrapper.h"

// memory management

template<>
inline MPSfloat* Memory<MPSfloat>::_alloc(size_t size) {
    // Metal cannot create empty buffers; allocate one element, as CUDA does.
    if (size == 0) size = 1;
    return (MPSfloat *)mpsMalloc(size * sizeof(MPSfloat));
}

template<>
inline void Memory<MPSfloat>::_free(MPSfloat* ptr) {
    mpsFree((float *) ptr);
}

template<>
class Matrix<MPSfloat> {
public:
    size_t numcol;
    size_t numrow;
    bool transpose;
    std::string name;

    std::shared_ptr<MPSfloat[]> elements;

    Matrix<MPSfloat>(const char* name, size_t numrow, size_t numcol, int trans,
        std::shared_ptr<MPSfloat[]> elements) {
    this->name = name;
    this->numrow = numrow;
    this->numcol = numcol;
    this->transpose = trans;
    this->elements = elements;
    }

    Matrix<MPSfloat>(const char* name, size_t numrow, size_t numcol,
            int trans);


public: 
    // constructors
    explicit Matrix<MPSfloat>(const Matrix<float>& M);
    Matrix<MPSfloat>(const char* name, size_t numrow, size_t numcol)
        : Matrix<MPSfloat>(name, numrow, numcol, 0){};
    // Empty state for non-affine layers: no device allocation or fill kernel.
    explicit Matrix<MPSfloat>(std::nullptr_t)
        : numcol(0), numrow(0), transpose(false), name("empty"), elements(nullptr) {}
    Matrix<MPSfloat>() : Matrix<MPSfloat>("un_init", 2, 2, 0){};

    // copy and move constructors, assignment operators
    Matrix<MPSfloat>(const Matrix<MPSfloat>& M);
    Matrix<MPSfloat>(Matrix<MPSfloat>&& M) noexcept;
    Matrix<MPSfloat>& operator=(const Matrix<MPSfloat>& M);
    Matrix<MPSfloat>& operator=(Matrix<MPSfloat>&& M) noexcept;

    inline size_t num_col() const { return transpose ? numrow : numcol; }

    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline size_t get_transpose() const { return transpose; }
    std::string get_name() const { return name; }
    const MPSfloat* data() const { return elements.get(); }
    inline MPSfloat elem(size_t i, size_t j) const {
        return transpose ? elements[j * numcol + i] : elements[i * numcol + j];
    }
    inline MPSfloat& elem(size_t i, size_t j) {
        return transpose ? elements[j * numcol + i] : elements[i * numcol + j];
    }
    // Host access to shared storage; call mpsSynchronize() first if GPU work
    // writing this matrix may still be pending.
    inline MPSfloat operator()(size_t i, size_t j) const { return elem(i, j); }
    inline MPSfloat& operator()(size_t i, size_t j) { return elem(i, j); }
    
    Matrix<float> to_host() const;
    void zeros();
    void ones();
    static Matrix<MPSfloat> rand(size_t m, size_t n);
    static Matrix<MPSfloat> randn(size_t m, size_t n);
    static Matrix<MPSfloat> zeros(size_t m, size_t n);
    static Matrix<MPSfloat> ones(size_t m, size_t n);

    // basic matrix ops
    Matrix<MPSfloat> dot(const Matrix<MPSfloat>& B) const;
    Matrix<MPSfloat> add(const Matrix<MPSfloat>& B, float s1, float s2) const;
    void add(const Matrix<MPSfloat>& B, float s1, float s2);
    Matrix<MPSfloat> add(float a, float s1) const;
    void add(float a, float s1);
    void scale(float s1);
    Matrix<MPSfloat> scale(float s1) const { return add(0, s1); }
    void eleminv(double l);
    Matrix<MPSfloat> eleminv(double l) const;
    float norm() const;
    // 2D submatrix (GPU block copy, stays in the command stream). Read form
    // returns the block; write form copies M into the
    // [rstart,rend) x [cstart,cend) region.
    Matrix<MPSfloat> slice(size_t rstart, size_t rend, size_t cstart,
                           size_t cend) const;
    void slice(size_t rstart, size_t rend, size_t cstart, size_t cend,
               const Matrix<MPSfloat>& M);
    // Indexed gather: result(i, j) = (*this)(rowidx[i], colidx[j]).
    Matrix<MPSfloat> slice(const idxlist& rowidx, const idxlist& colidx) const;
    Matrix<MPSfloat> rows(size_t start, size_t end) const {
        return slice(start, end, 0, num_col());
    }
    void rows(size_t start, size_t end, const Matrix<MPSfloat>& M) {
        slice(start, end, 0, num_col(), M);
    }
    Matrix<MPSfloat> rows(idxlist rlist) const { return slice(rlist, seq(num_col())); }
    Matrix<MPSfloat> columns(size_t start, size_t end) const {
        return slice(0, num_row(), start, end);
    }
    void columns(size_t start, size_t end, const Matrix<MPSfloat>& M) {
        slice(0, num_row(), start, end, M);
    }
    Matrix<MPSfloat> columns(idxlist clist) const { return slice(seq(num_row()), clist); }
    const Matrix<MPSfloat> T() const;

    friend Matrix<MPSfloat> hadmd(const Matrix<MPSfloat>& M1,
        const Matrix<MPSfloat>& M2);
    friend Matrix<MPSfloat> hadmd(const Matrix<MPSfloat>& M1,
        Matrix<MPSfloat>&& M2);
    friend Matrix<MPSfloat> hadmd(Matrix<MPSfloat>&& M1,
       const Matrix<MPSfloat>& M2);

    friend Matrix<float> topk(const Matrix<MPSfloat>& M, int k, int dim);
    friend Matrix<MPSfloat> tanh(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> tanh(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> d_tanh(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> d_tanh(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> sum(const Matrix<MPSfloat>& M, int dim);
    friend Matrix<MPSfloat> exp(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> exp(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> log(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> log(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> square(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> square(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> sqrt(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> sqrt(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> relu(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> relu(Matrix<MPSfloat>&& M);
    friend Matrix<MPSfloat> d_relu(const Matrix<MPSfloat>& M);
    friend Matrix<MPSfloat> d_relu(Matrix<MPSfloat>&& M);
};

std::ostream& operator <<(std::ostream& os, const Matrix<MPSfloat>& M);
Matrix<MPSfloat> sum(const Matrix<MPSfloat>& M, int dim);
Matrix<float> topk(const Matrix<MPSfloat>& M, int k, int dim);
Matrix<MPSfloat> tanh(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> tanh(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> d_tanh(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> d_tanh(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> square(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> square(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> sqrt(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> sqrt(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> relu(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> relu(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> d_relu(const Matrix<MPSfloat>& M);
Matrix<MPSfloat> d_relu(Matrix<MPSfloat>&& M);
Matrix<MPSfloat> clamp(const Matrix<MPSfloat>& M, float lo, float hi);
Matrix<MPSfloat> clamp(Matrix<MPSfloat>&& M, float lo, float hi);

Matrix<MPSfloat>& fill(Matrix<MPSfloat>& M, double a);
// dest takes src's shape and layout in fresh storage.
void copy(Matrix<MPSfloat>& dest, const Matrix<MPSfloat>& src);

Matrix<MPSfloat> hstack(std::vector<MatrixView<MPSfloat>> matrices);

Matrix<MPSfloat> vstack(std::vector<MatrixView<MPSfloat>> matrices);

// Seeds the device generator behind rand() and randn(), like the CUDA
// GPUSampler. setseed restarts both streams from the beginning.
struct GPUSampler {
    explicit GPUSampler(int seed) { setseed(seed); }
    void setseed(int seed) { mpsSetSeed(seed); }
};

// Matrix files hold host (column-major) storage, so a checkpoint written on
// one backend loads on any other.
template <>
void write(FILE* fp, const Matrix<MPSfloat>& M);
template <>
void read(FILE* fp, Matrix<MPSfloat>& M);

// ── host-fallback generic ops ───────────────────────────────────────────────
// reduce/elemwise take an arbitrary C++ lambda, which cannot be JIT-compiled
// into a Metal kernel. On Apple Silicon's unified memory we round-trip through
// the host and reuse the CPU implementations (Matrix<float>). These overloads
// are more specialized than the generic Matrix<D> templates, so they win
// overload resolution for Matrix<MPSfloat>. Heavy GEMMs stay on the GPU.
template <class Function>
Matrix<MPSfloat> reduce(Function func, const Matrix<MPSfloat>& M, int dim, int k) {
    return Matrix<MPSfloat>(reduce(func, M.to_host(), dim, k));
}
template <class Function>
Matrix<MPSfloat> elemwise(Function func, const Matrix<MPSfloat>& M) {
    return Matrix<MPSfloat>(elemwise(func, M.to_host()));
}
template <class Function>
Matrix<MPSfloat> elemwise(Function func, Matrix<MPSfloat>&& M) {
    // The shared Metal buffer is CPU-accessible once pending GPU work ends.
    // Keep the rvalue's allocation, including its physical transpose layout.
    mpsSynchronize();
    auto* values = reinterpret_cast<float*>(M.elements.get());
    for (size_t i = 0; i < M.numrow * M.numcol; ++i)
        values[i] = func(values[i]);
    return std::move(M);
}

#endif