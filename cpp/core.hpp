/**
 * @file core.hpp
 * @brief Core Components
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
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

#ifndef CORE_HPP
#define CORE_HPP

#include "helper.hpp"
#include "memory.hpp"

template <class D>
class Matrix;

/**
 * @brief non-owning view of a matrix.
 *
 * @tparam D the data type
 */
template <class D>
class MatrixView {
    const D *elements;
    size_t numcol;
    size_t numrow;
    bool transpose;

    inline size_t idx(size_t i, size_t j) const {
        return transpose ? i * numrow + j : j * numrow + i;
    }

   public:
    MatrixView(const Matrix<D> &matrix)
        : elements(matrix.elements.get()),
          numrow(matrix.numrow),
          numcol(matrix.numcol),
          transpose(matrix.transpose) {}

    inline size_t num_col() const { return transpose ? numrow : numcol; }
    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline size_t get_transpose() const { return transpose; }
    inline const D *data() const { return elements; }
    inline D elem(size_t i, size_t j) const { return elements[idx(i, j)]; }

    // hack, you need this to access transpose
    friend const Matrix<CUDAfloat> vstack(
        std::vector<MatrixView<CUDAfloat>> matrices);
};

template <class D>
class Matrix {
   protected:
    size_t numcol;
    size_t numrow;
    bool transpose;
    std::string name;

    std::shared_ptr<D[]> elements;

    Matrix(const char *name, size_t numrow, size_t numcol, int trans,
           std::shared_ptr<D[]> elements) {
        this->name = name;
        this->numrow = numrow;
        this->numcol = numcol;
        this->transpose = trans;
        this->elements = elements;
    }

    inline size_t idx(size_t i, size_t j) const {
        return transpose ? i * numrow + j : j * numrow + i;
    }

    Matrix(const char *name, size_t numrow, size_t numcol, int trans);

   public:
    // constructors
    Matrix()
        : numcol(0),
          numrow(0),
          elements(nullptr),
          transpose(0),
          name("uninit") {}
    Matrix(const char *name, size_t numrow, size_t numcol)
        : Matrix(name, numrow, numcol, 0) {}
    Matrix(const char *name, std::vector<std::vector<double>> elems);
    Matrix(const char *name, size_t numrow, size_t numcol,
           std::shared_ptr<D[]> elements)
        : Matrix(name, numrow, numcol, 0, elements){};

    Matrix(const Matrix &M);
    Matrix(Matrix &&M) noexcept;

    Matrix<D> &operator=(const Matrix<D> &M);
    Matrix<D> &operator=(Matrix<D> &&M) noexcept;

    // access matrix info
    inline D elem(size_t i, size_t j) const { return elements[idx(i, j)]; }
    inline D &elem(size_t i, size_t j) { return elements[idx(i, j)]; }
    inline D operator()(size_t i, size_t j) const {
        return elements[idx(i, j)];
    }
    inline D &operator()(size_t i, size_t j) { return elements[idx(i, j)]; }

    inline size_t num_col() const { return transpose ? numrow : numcol; }
    inline size_t num_row() const { return transpose ? numcol : numrow; }
    inline int get_transpose() const { return transpose; }
    std::string get_name() const { return name; }
    const D *data() const { return elements.get(); }
    const std::shared_ptr<D[]> get_shared_ptr() const { return elements; }

    // Matrix Fillers
    void zeros() { memset(elements.get(), 0, sizeof(D) * numrow * numcol); }
    void ones();

    // Matrix Operations / performance notes
    Matrix<D> dot(const Matrix<D> &B) const;  // using cblas_dgemm
    Matrix<D> add(const Matrix<D> &B, D s1,
                  D s2) const;  // using double for loop
    void add(const Matrix<D> &B, D s1, D s2);
    Matrix<D> add(D b, D s1) const;  // using single for loop
    void add(D b, D s1);
    Matrix<D> scale(D s1) const { return add(0, s1); }
    void scale(D s1) { add(0, s1); }

    void eleminv(double l);
    Matrix<D> eleminv(double l) const;

    Matrix<D> inv();  // using LU decomposition
    D norm() const;   // using single for loop
    Matrix<D> T() const;

    // Matrix slicers
    Matrix<D> columns(size_t start, size_t end) const;
    void columns(size_t start, size_t end, const Matrix<D> &M);
    Matrix<D> columns(idxlist cols) const;
    Matrix<D> rows(size_t start, size_t end) const;
    void rows(size_t start, size_t end, const Matrix<D> &M);
    Matrix<D> rows(idxlist rows) const;
    Matrix<D> slice(size_t rstart, size_t rend, size_t cstart,
                    size_t cend) const;
    void slice(size_t rstart, size_t rend, size_t cstart, size_t cend,
               const Matrix<D> &M);
    Matrix<D> slice(const idxlist &rowidx, const idxlist &colidx) const;

    static Matrix<D> randn(size_t m, size_t n);
    static Matrix<D> rand(size_t m, size_t n);
    static Matrix<D> ones(size_t m, size_t n);
    static Matrix<D> zeros(size_t m, size_t n);
    // Matrix's friends
    template <class Data>
    friend Matrix<Data> sum(const Matrix<Data> &M, int dim);
    template <class Data>
    friend Matrix<Data> exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> exp(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> log(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> tanh(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> tanh(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> d_tanh(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> d_tanh(Matrix<Data> &&M);

    template <class Data>
    friend Matrix<Data> atan_exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> atan_exp(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> d_atan_exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> d_atan_exp(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> sin(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> sin(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> cos(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> cos(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> square(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> square(Matrix<Data> &&M);

    template <class Data, class Function>
    friend Matrix<Data> elemwise(Function func, Matrix<Data> &&M);
    template <class Data, class Function>
    friend Matrix<Data> elemwise(Function func, const Matrix<Data> &M);
    template <class Data, class Function>
    friend Matrix<Data> reduce(Function func, const Matrix<Data> &M, int dim,
                               int k);

    template <class Data>
    friend class MatrixView;
    template <class Data>
    friend Matrix<Data> hstack(std::vector<MatrixView<Data>> matrices);
    template <class Data>
    friend Matrix<Data> vstack(std::vector<MatrixView<Data>> matrices);
    template <class Data>
    friend Matrix<Data> hadmd(const Matrix<Data> &M1, const Matrix<Data> &M2);

    template <class Data>
    friend void read(FILE *f, Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> read(std::string filename);
    template <class Data>
    friend void write(std::string filename, const Matrix<Data> &M);
    template <class Data>
    friend void write(FILE *f, const Matrix<Data> &M);

    friend class Memory<D>;
    friend class Matrix<CUDAfloat>;
};

template <class D>
Matrix<D>::Matrix(const char *name, size_t numrow, size_t numcol, int trans) {
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    // TODO: Not nice using reset, should change
    elements.reset(Memory<D>::allocate(numrow * numcol),
                   [](D *p) { Memory<D>::free(p); });
}

template <class D>
Matrix<D>::Matrix(const char *name, std::vector<std::vector<double>> elems) {
    this->name = name;
    numrow = elems.size();
    numcol = elems.front().size();
    transpose = 0;

    // TODO: Not nice using reset, should change
    elements.reset(Memory<D>::allocate(numrow * numcol),
                   [](D *p) { Memory<D>::free(p); });

    // double for loop may have some performance issues,
    // but should be OK for small matrices.
    // remember to always loop over columns first.
    for (size_t j = 0; j < numcol; j++) {
        for (size_t i = 0; i < numrow; i++) {
            elem(i, j) = elems[i][j];
        }
    }
}

template <class D>
Matrix<D>::Matrix(const Matrix<D> &M) {
    LOG_DEBUG("copy construction called, {}!", M.name);
    numcol = M.numcol;
    numrow = M.numrow;
    transpose = M.transpose;
    elements.reset(Memory<D>::allocate(numrow * numcol),
                   [](D *p) { Memory<D>::free(p); });

    memcpy(elements.get(), M.elements.get(), numrow * numcol * sizeof(D));
    name = "copy of " + M.name;
}

template <class D>
Matrix<D>::Matrix(Matrix<D> &&M) noexcept {
    LOG_DEBUG("move constructor called");
    this->numcol = M.numcol;
    this->numrow = M.numrow;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
    this->name = M.name;
}

template <class D>
Matrix<D> &Matrix<D>::operator=(const Matrix<D> &M) {
    LOG_DEBUG("copy assignment called!");
    if (this == &M) return *this;
    numcol = M.numcol;
    numrow = M.numrow;
    transpose = M.transpose;
    elements.reset(Memory<D>::allocate(numrow * numcol),
                   [](D *p) { Memory<D>::free(p); });

    memcpy(elements.get(), M.elements.get(), numrow * numcol * sizeof(D));
    name = "copy of " + M.name;
    return *this;
}

template <class D>
Matrix<D> &Matrix<D>::operator=(Matrix<D> &&M) noexcept {
    LOG_DEBUG("move assignment called");
    if (this == &M) return *this;
    this->numcol = M.numcol;
    this->numrow = M.numrow;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
    this->name = M.name;
    return *this;
}

template <class D>
void Matrix<D>::ones() {
    size_t numelems = num_row() * num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++) elements[i] = 1.0;
}

template <class D>
D Matrix<D>::norm() const {
    D norm = 0;
    size_t numelems = num_row() * num_col();
#pragma clang loop vectorize(enable)
#pragma ivdep
    for (size_t i = 0; i < numelems; i++) norm += elements[i] * elements[i];

    return sqrt(norm);
}
//"fake" matrix transposition. It does not copy the data.
template <class D>
Matrix<D> Matrix<D>::T() const {
    std::string newname = name + "_T";
    Matrix<D> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}

template <class D>
Matrix<D> Matrix<D>::slice(size_t rstart, size_t rend, size_t cstart,
                           size_t cend) const {
    Matrix<D> M("submatrix", rend - rstart, cend - cstart, 0);
    for (size_t j = cstart; j < cend; j++) {
        for (size_t i = rstart; i < rend; i++) {
            M.elem(i - rstart, j - cstart) = elem(i, j);
        }
    }
    return M;
}

template <class D>
void Matrix<D>::slice(size_t rstart, size_t rend, size_t cstart, size_t cend,
                      const Matrix<D> &M) {
    for (size_t j = cstart; j < cend; j++) {
        for (size_t i = rstart; i < rend; i++) {
            elem(i, j) = M.elem(i - rstart, j - cstart);
        }
    }
}

template <class D>
Matrix<D> Matrix<D>::slice(const idxlist &rowidx, const idxlist &colidx) const {
    // TODO: the size of matrix is int, which may not be enough
    Matrix<D> M("submatrix", rowidx.size(), colidx.size(), 0);
    size_t i = 0;
    for (auto it = rowidx.begin(); it != rowidx.end(); it++) {
        size_t r = *it;
        size_t j = 0;
        for (auto it2 = colidx.begin(); it2 != colidx.end(); it2++) {
            size_t c = *it2;
            M.elem(i, j) = elem(r, c);
            j++;
        }
        i++;
    }
    return M;
}

template <class D>
Matrix<D> Matrix<D>::dot(const Matrix<D> &B) const {
    STATIC_TIC;
    Matrix<D> C("dot", num_row(), B.num_col(), 0);
    C.zeros();
#ifdef NO_CBLAS
    size_t n = C.num_row();
    size_t m = C.num_col();
    size_t k = num_col();

#pragma omp simd
#pragma ivdep
    for (size_t i = 0; i < n; i++) {
#pragma ivdep
        for (size_t j = 0; j < m; j++) {
#pragma ivdep
            for (size_t l = 0; l < k; l++) {
                C.elem(i, j) += elem(i, l) * B.elem(l, j);
            }
        }
    }
#else
    CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = B.transpose ? CblasTrans : CblasNoTrans;

    gemm(transA, transB, num_row(), B.num_col(), num_col(), 1.0f,
         elements.get(), numrow, B.elements.get(), B.numrow, 1.0f,
         C.elements.get(), C.numrow);
#endif
    STATIC_TOC;
    return C;
}

/*
Matrix inversion using Lapack, found and modified from:
https://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c/41474839
*/
template <class D>
Matrix<D> Matrix<D>::inv() {
    Matrix<D> inv("inv_M", numrow, numcol, 0);
    for (size_t i = 0; i < numrow * numcol; i++) {
        inv.elements[i] = elements[i];
    }

    int N = (int)num_row();
    int error = 0;
    int *pivot = (int *)malloc(
        N *
        sizeof(
            int));  // LAPACK requires MIN(M,N), here M==N, so N will do fine.
    int Nwork = 2 * N * N;
    D *workspace = (D *)malloc(Nwork * sizeof(D));

    /*  LU factorisation */
    getrf_(&N, &N, inv.elements.get(), &N, pivot, &error);

    if (error != 0) {
        // NSLog(@"Error 1");
        std::cout << "Error 1" << std::endl;
        ::free(pivot);
        ::free(workspace);
        exit(1);
    }

    /*  matrix inversion */
    getri_(&N, inv.elements.get(), &N, pivot, workspace, &Nwork, &error);

    if (error != 0) {
        // NSLog(@"Error 2");
        std::cout << "Error 2" << std::endl;
        ::free(pivot);
        ::free(workspace);
        exit(1);
    }

    ::free(pivot);
    ::free(workspace);

    return inv;
}

// C = s1*A + s2*B
template <class D>
Matrix<D> Matrix<D>::add(const Matrix<D> &B, D s1, D s2) const {
    Matrix<D> C("add", num_row(), num_col(), 0);

#ifdef INTEL_MKL
    C.zeros();
    char transA = transpose ? 'T' : 'N';
    char transB = B.transpose ? 'T' : 'N';

    omatadd(transA, transB, num_row(), num_col(), s1, elements.get(), numrow,
            s2, B.elements.get(), B.numrow, C.elements.get(), C.numrow);
#else
    for (size_t j = 0; j < num_col(); j++) {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_row(); i++) {
            // NOTE: Perhaps there is better way of doing this
            //  considering the cache hit rate.
            // NOTE: there was a mistake here
            C.elem(i, j) = s1 * elem(i, j) + s2 * B.elem(i, j);
        }
    }
#endif
    return C;
}

// A = s1*A + s2*B
template <class D>
void Matrix<D>::add(const Matrix<D> &B, D s1, D s2) {
    for (size_t j = 0; j < num_col(); j++) {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_row(); i++) {
            // NOTE: Perhaps there is better way of doing this
            //  considering the cache hit rate.
            // NOTE: there was a mistake here
            elem(i, j) = s1 * elem(i, j) + s2 * B.elem(i, j);
        }
    }
}

// C = s1*A + b
template <class D>
Matrix<D> Matrix<D>::add(D b, D s1) const {
#ifdef INTEL_MKL
    Matrix<D> C("add", num_row(), num_col(), 0);
    C.zeros();
    auto e = Matrix<D>::ones(num_row(), num_col());
    char transA = transpose ? 'T' : 'N';
    char transB = 'N';

    omatadd(transA, transB, num_row(), num_col(), s1, elements.get(), numrow, b,
            e.elements.get(), e.numrow, C.elements.get(), C.numrow);
#else

    Matrix<D> C("add", numrow, numcol, transpose);
    size_t numelems = num_row() * num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++) {
        C.elements[i] = s1 * elements[i] + b;
    }
#endif
    return C;
}

// A = s1*A + b
template <class D>
void Matrix<D>::add(D b, D s1) {
    size_t numelems = num_row() * num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++) {
        elements[i] = s1 * elements[i] + b;
    }
}

template <class D>
void Matrix<D>::eleminv(double l) {
    for (size_t i = 0; i < numrow * numcol; i++) elements[i] = l / elements[i];
}

template <class D>
Matrix<D> Matrix<D>::eleminv(double l) const {
    Matrix<D> M((name + "reci").c_str(), numrow, numcol, transpose);
    for (size_t i = 0; i < numrow * numcol; i++)
        M.elements[i] = l / elements[i];
    return M;
}

template <class D>
void read(FILE *f, Matrix<D> &M){
    // read int variables to the file.
    size_t numrow = getw(f);
    size_t numcol = getw(f);
    size_t transpose = getw(f);

    int bytesread = fread(M.elements.get(), sizeof(D), numcol * numrow, f);
}

/*
 * Read Matrix from a File
 *
 */

template <class D>
Matrix<D> read(std::string filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    // read int variables to the file.
    size_t numrow = getw(f);
    size_t numcol = getw(f);
    size_t transpose = getw(f);

    Matrix<D> A(filename.c_str(), numrow, numcol, transpose);
    int bytesread = fread(A.elements.get(), sizeof(D), numcol * numrow, f);
    fclose(f);

    return A;
}

template <class D>
void write(FILE *f, const Matrix<D> &M) {
    // write int variables to the file.
    putw(M.numrow, f);
    putw(M.numcol, f);
    putw(M.transpose, f);
    fwrite(M.elements.get(), sizeof(CUDAfloat), M.numcol * M.numrow, f);
}

template <class D>
/*
    Write matrix M to file
    M: the matrix to be written
    filename: name of the file to be created.
    */
void write(std::string filename, const Matrix<D> &M) {
    FILE *f = fopen(filename.c_str(), "wb");
    // write int variables to the file.

    write(f, M);

    fclose(f);
}

#endif
