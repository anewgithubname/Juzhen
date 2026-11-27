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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <fstream>
#include <cstring>
#include <type_traits>

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
        if constexpr (std::is_same_v<D, MPSfloat>) {
            return transpose ? j * numcol + i : i * numcol + j;
        } else {
            return transpose ? i * numrow + j : j * numrow + i;
        }
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

// out = f(a, b), element by element, for two matrices of the same shape.
// a and out are stored the same way (numrow x numcol, column-major, and out
// may be a itself); b is stored either like them or as their transpose.
// Stored alike, it is one pass over the storage. Otherwise b is read against
// its storage order, and the work goes tile by tile so that b's tile stays in
// the cache.
template <class D, class F>
inline void zip_storage(D *out, const D *a, const D *b, size_t numrow,
                        size_t numcol, bool b_is_transposed, F f) {
    const size_t n = numrow * numcol;
    if (!b_is_transposed) {
        if (out == a && out != b) {
            // In place (A += B). The loop below is compiled with a test for
            // overlapping storage that sends this case to its one-at-a-time
            // version; here the compiler is told that only b is separate.
            D *__restrict o = out;
            const D *__restrict q = b;
            for (size_t k = 0; k < n; k++) o[k] = f(o[k], q[k]);
            return;
        }
        for (size_t k = 0; k < n; k++) out[k] = f(a[k], b[k]);
        return;
    }
    // A += A.T: b is the storage being written; read it from a copy.
    std::vector<D> kept;
    if (b == out) {
        kept.assign(b, b + n);
        b = kept.data();
    }
    // 16 x 16: small enough that the tile of b stays cached even when the
    // number of rows is a power of two and its pieces share cache sets.
    const size_t tile = 16;
    for (size_t j0 = 0; j0 < numcol; j0 += tile) {
        const size_t j1 = std::min(j0 + tile, numcol);
        for (size_t i0 = 0; i0 < numrow; i0 += tile) {
            const size_t i1 = std::min(i0 + tile, numrow);
            for (size_t j = j0; j < j1; j++)
                for (size_t i = i0; i < i1; i++)
                    out[j * numrow + i] = f(a[j * numrow + i], b[i * numcol + j]);
        }
    }
}

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
    // Explicit empty storage; keep the historical default constructor unchanged.
    explicit Matrix(std::nullptr_t)
        : numcol(0), numrow(0), transpose(false), name("empty"), elements(nullptr) {}
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
    Matrix<D> scale(D s1) const;  // a multiplication alone, not s1*A + 0
    void scale(D s1);

    void eleminv(double l);
    Matrix<D> eleminv(double l) const;
    Matrix<D> elemdiv(const Matrix<D> &B) const;  // this ./ B in one pass

    D norm() const;   // using single for loop
    const Matrix<D> T() const;

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
    friend Matrix<Data> sum_direct(const Matrix<Data> &M, int dim);
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
    friend Matrix<Data> hadmd(const Matrix<Data> &M1, Matrix<Data> &&M2);
    template <class Data>
    friend Matrix<Data> hadmd(Matrix<Data> &&M1, const Matrix<Data> &M2);
    friend Matrix<float> topk(const Matrix<MPSfloat>& M, int k, int dim);

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
    friend class Matrix<MPSfloat>;
    friend class Matrix<ROCMfloat>;
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
    for (size_t i = 0; i < numelems; i++) elements[i] = 1.0;
}

template <class D>
D Matrix<D>::norm() const {
    D norm = 0;
    size_t numelems = num_row() * num_col();
    for (size_t i = 0; i < numelems; i++) norm += elements[i] * elements[i];

    return sqrt(norm);
}
//"fake" matrix transposition. It does not copy the data.
template <class D>
const Matrix<D> Matrix<D>::T() const {
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
    // column by column, so that the result is written in storage order
    D *out = M.elements.get();
    for (size_t c : colidx)
        for (size_t r : rowidx) *out++ = elem(r, c);
    return M;
}

template <class D>
Matrix<D> Matrix<D>::dot(const Matrix<D> &B) const {
    STATIC_TIC;
    // check if dimensions are compatible
    if (num_col() != B.num_row()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<D> C("dot", num_row(), B.num_col(), 0);
    C.zeros();
    CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = B.transpose ? CblasTrans : CblasNoTrans;

    gemm(transA, transB, num_row(), B.num_col(), num_col(), 1.0f,
         elements.get(), numrow, B.elements.get(), B.numrow, 1.0f,
         C.elements.get(), C.numrow);
    STATIC_TOC;
    return C;
}

// C = s1*A + s2*B
template <class D>
Matrix<D> Matrix<D>::add(const Matrix<D> &B, D s1, D s2) const {
    // check if dimensions are compatible
    if (num_col() != B.num_col() || num_row() != B.num_row()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    // The result is stored like the two operands when they are stored alike
    // (transposed or not), and without the transpose flag otherwise.
    const bool swap = transpose && !B.transpose;
    const Matrix<D> &x = swap ? B : *this, &y = swap ? *this : B;
    const D sx = swap ? s2 : s1, sy = swap ? s1 : s2;
    Matrix<D> C("add", x.numrow, x.numcol, x.transpose);
    zip_storage(C.elements.get(), x.elements.get(), y.elements.get(), x.numrow,
                x.numcol, x.transpose != y.transpose,
                [sx, sy](D p, D q) { return sx * p + sy * q; });
    return C;
}

// A = s1*A + s2*B
template <class D>
void Matrix<D>::add(const Matrix<D> &B, D s1, D s2) {
    // check if dimensions are compatible
    if (num_col() != B.num_col() || num_row() != B.num_row()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    zip_storage(elements.get(), elements.get(), B.elements.get(), numrow,
                numcol, transpose != B.transpose,
                [s1, s2](D p, D q) { return s1 * p + s2 * q; });
}

// C = s1*A + b
template <class D>
Matrix<D> Matrix<D>::add(D b, D s1) const {
    Matrix<D> C("add", numrow, numcol, transpose);
    // plain pointers: through the shared pointers the loop is vectorized in
    // some places it is inlined and not in others
    const size_t numelems = numrow * numcol;
    const D *in = elements.get();
    D *__restrict out = C.elements.get();
    // A + b and b - A are one operation per element, not a product and a sum
    // (the values are the same: 1*x is x and -1*x + b is b - x exactly)
    if (s1 == D(1)) {
        for (size_t i = 0; i < numelems; i++) out[i] = in[i] + b;
    } else if (s1 == D(-1)) {
        for (size_t i = 0; i < numelems; i++) out[i] = b - in[i];
    } else {
        for (size_t i = 0; i < numelems; i++) out[i] = s1 * in[i] + b;
    }
    return C;
}

// A = s1*A + b
template <class D>
void Matrix<D>::add(D b, D s1) {
    const size_t numelems = numrow * numcol;
    D *p = elements.get();
    if (s1 == D(1)) {
        for (size_t i = 0; i < numelems; i++) p[i] = p[i] + b;
    } else if (s1 == D(-1)) {
        for (size_t i = 0; i < numelems; i++) p[i] = b - p[i];
    } else {
        for (size_t i = 0; i < numelems; i++) p[i] = s1 * p[i] + b;
    }
}

// C = s1*A
template <class D>
Matrix<D> Matrix<D>::scale(D s1) const {
    Matrix<D> C("scale", numrow, numcol, transpose);
    const size_t numelems = numrow * numcol;
    const D *in = elements.get();
    D *__restrict out = C.elements.get();
    for (size_t i = 0; i < numelems; i++) {
        out[i] = s1 * in[i];
    }
    return C;
}

// A = s1*A
template <class D>
void Matrix<D>::scale(D s1) {
    const size_t numelems = numrow * numcol;
    D *p = elements.get();
    for (size_t i = 0; i < numelems; i++) {
        p[i] = s1 * p[i];
    }
}

// The division is done in float for a float matrix (as on the GPUs): dividing
// in double, one element at a time, is several times slower.
template <class D>
void Matrix<D>::eleminv(double l) {
    using Q = std::conditional_t<std::is_same_v<D, float>, float, double>;
    const Q s = (Q)l;
    const size_t n = numrow * numcol;
    D *p = elements.get();
    for (size_t i = 0; i < n; i++) p[i] = s / p[i];
}

template <class D>
Matrix<D> Matrix<D>::eleminv(double l) const {
    Matrix<D> M((name + "reci").c_str(), numrow, numcol, transpose);
    using Q = std::conditional_t<std::is_same_v<D, float>, float, double>;
    const Q s = (Q)l;
    const size_t n = numrow * numcol;
    const D *in = elements.get();
    D *__restrict out = M.elements.get();
    for (size_t i = 0; i < n; i++) out[i] = s / in[i];
    return M;
}

// this ./ B in one pass over the storage. The result is stored like the two
// operands when they are stored alike, and without the transpose flag
// otherwise (see Matrix<D>::add).
template <class D>
Matrix<D> Matrix<D>::elemdiv(const Matrix<D> &B) const {
    if (num_row() != B.num_row() || num_col() != B.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    if (transpose && !B.transpose) {
        Matrix<D> C("elemdiv", B.numrow, B.numcol, B.transpose);
        zip_storage(C.elements.get(), B.elements.get(), elements.get(), B.numrow,
                    B.numcol, true, [](D q, D p) { return p / q; });
        return C;
    }
    Matrix<D> C("elemdiv", numrow, numcol, transpose);
    zip_storage(C.elements.get(), elements.get(), B.elements.get(), numrow,
                numcol, transpose != B.transpose,
                [](D p, D q) { return p / q; });
    return C;
}

// Helpers for the existing three-native-int matrix format. Keep stream reads
// composable: validate available payload when seekable, but preserve pipe input.
namespace matrix_io_detail {

// log the reason and abort the read
[[noreturn]] inline void fail(const char* reason) {
    LOG_ERROR("matrix/data read: {}", reason);
    ERROR_OUT;
}

// the three ints at the start of every .matrix file
struct Header {
    size_t rows;
    size_t cols;
    int transpose;  // 0 or 1
};

// read and sanity-check the header
inline Header header(FILE* file) {
    if (!file)
        fail("null input stream");

    int fields[3];
    if (std::fread(fields, sizeof(int), 3, file) != 3)
        fail("incomplete matrix header");

    const int rows = fields[0], cols = fields[1], transpose = fields[2];
    if (rows < 0 || cols < 0 || (transpose != 0 && transpose != 1))
        fail("invalid matrix header");

    return {size_t(rows), size_t(cols), transpose};
}

// rows * cols, failing if the matrix would be too big to address in bytes
template <class D>
size_t count(size_t rows, size_t cols) {
    const size_t max_bytes = std::min(std::numeric_limits<size_t>::max(),
                                      size_t(std::numeric_limits<std::ptrdiff_t>::max()));
    const size_t max_elements = max_bytes / sizeof(D);
    if (cols && rows > max_elements / cols)
        fail("matrix byte size overflows");
    return rows * cols;
}

// does the file have at least `bytes` left after the current position?
// Streams that cannot seek (e.g. a pipe) are assumed fine; the caller's
// fread still checks every item.
inline bool enough_bytes(FILE* file, size_t bytes) {
    const long position = std::ftell(file);
    if (position < 0)
        return true;

    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::clearerr(file);
        return true;
    }
    const long end = std::ftell(file);

    // go back to where we were
    if (std::fseek(file, position, SEEK_SET) != 0)
        fail("cannot restore input position");

    return end >= position && size_t(end - position) >= bytes;
}

// number of elements to read, after checking the file really holds them
template <class D>
size_t payload(FILE* file, size_t rows, size_t cols) {
    const size_t n = count<D>(rows, cols);
    if (!enough_bytes(file, n * sizeof(D)))
        fail("truncated matrix payload");
    return n;
}

// deleter so std::unique_ptr<FILE, FileCloser> closes the file automatically
struct FileCloser {
    void operator()(FILE* file) const {
        if (file)
            std::fclose(file);
    }
};

}  // namespace matrix_io_detail

template <class D>
void read(FILE* file,Matrix<D>& destination) {
    const auto header=matrix_io_detail::header(file);
    const size_t n=matrix_io_detail::payload<D>(file,header.rows,header.cols);
    // Preserve the stream reader's standalone allocation policy. Pool reuse
    // changes buffer placement and can slow cached reads substantially.
    std::shared_ptr<D[]> loaded(new D[n]);
    if(n && std::fread(loaded.get(),sizeof(D),n,file)!=n)matrix_io_detail::fail("incomplete matrix payload");
    // Publish only after a complete read; the array owner uses delete[].
    destination.elements=std::move(loaded);
    destination.numrow=header.rows;
    destination.numcol=header.cols;
    destination.transpose=header.transpose;
}

template <class D>
Matrix<D> read(std::string filename) {
    std::unique_ptr<FILE,matrix_io_detail::FileCloser> file(std::fopen(filename.c_str(),"rb"));
    if(!file)matrix_io_detail::fail("cannot open matrix file");
    const auto header=matrix_io_detail::header(file.get());
    const size_t n=matrix_io_detail::payload<D>(file.get(),header.rows,header.cols);
    // Filename reads retain their existing pooled allocation policy.
    Matrix<D> loaded(filename.c_str(),header.rows,header.cols,header.transpose);
    if(n && std::fread(loaded.elements.get(),sizeof(D),n,file.get())!=n)matrix_io_detail::fail("incomplete matrix payload");
    return loaded;
}

template <class D>
void write(FILE *f, const Matrix<D> &M) {
    // write int variables to the file.
    putw(M.numrow, f);
    putw(M.numcol, f);
    putw(M.transpose, f);
    fwrite(M.elements.get(), sizeof(D), M.numcol * M.numrow, f);
}

template <class D>
/*
    Write matrix M to file
    M: the matrix to be written
    filename: name of the file to be created.
    */
void write(std::string filename, const Matrix<D> &M) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        LOG_ERROR("write: cannot open file %s", filename.c_str());
        return;
    }

    write(f, M);

    fclose(f);
}

/*
    write matrix to csv file
 */
template <class T>
void writetocsv(const std::string& filename, const MatrixView<T>& M) {
    using namespace std;
    
    std::cout << "writing to " << filename << std::endl;

    ofstream myfile;

    myfile.open(filename);
    for (int i = 0; i < M.num_col(); i++){
        for (int j = 0; j < M.num_row(); j++){
            myfile << M.elem(j, i) << ",";
        }
        myfile << std::endl;
    }
    myfile.close();

}

/*
    read matrix from csv file
*/
template <class T>
Matrix<T> readfromcsv(const std::string& filename, int n, int d) {
    std::cout << "reading from " << filename << std::endl;
    if(n<0 || d<0)matrix_io_detail::fail("negative CSV dimensions");
    std::ifstream file(filename);
    if(!file)matrix_io_detail::fail("cannot open CSV file");
    matrix_io_detail::count<T>(size_t(n),size_t(d));
    Matrix<T> ret = Matrix<T>::zeros(n, d);
    std::string line;
    int i = 0;
    while (std::getline(file, line) && i < n) {
        std::stringstream ss(line);
        std::string token;
        int j = 0;
        while (std::getline(ss, token, ',') && j < d) {
            ret.elem(i, j) = std::stof(token);
            j++;
        }
        i++;
    }
    return ret;
}


#endif
