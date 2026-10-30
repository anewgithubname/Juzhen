#include <cmath>
#include <limits>
#include <string>
#include "operators.hpp"
#include "mpsmatrix.hpp"
#include "matrix.hpp"

// Accelerate's SIMD out-of-place transpose: C (M x N, row-major) = A^T where A
// is N x M row-major. Declared directly because <Accelerate/Accelerate.h>
// would pull in a cblas.h that clashes with the OpenBLAS one already included.
extern "C" void vDSP_mtrans(const float* A, long strideA, float* C, long strideC,
                            unsigned long M, unsigned long N);

Matrix<MPSfloat>::Matrix(const Matrix<float>& M)
{
    static Profiler profiler("upload to MPS memory");
    this->name = "mps_" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    MPSfloat* p = Memory<MPSfloat>::allocate( numcol * numrow);
    elements.reset(p, [](MPSfloat* p) {
        Memory<MPSfloat>::free(p);
    });

    mpsSynchronize(); // before the upload, make sure all the previous operations are done
    profiler.start();
    //change column major to row major
    vDSP_mtrans((const float*)M.elements.get(), 1, (float*)elements.get(), 1,
                numrow, numcol);
    profiler.end();
}

Matrix<MPSfloat>::Matrix(const char* name, size_t numrow, size_t numcol, int trans)
{
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    MPSfloat* p = Memory<MPSfloat>::allocate( numcol * numrow);
    elements.reset(p, [](MPSfloat* p) {
        Memory<MPSfloat>::free(p);
    });

}

Matrix<MPSfloat>::Matrix(const Matrix<MPSfloat>& M) {
    LOG_DEBUG("mps copy constructor called");
    this->name = "copy of" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<MPSfloat[]>(
        Memory<MPSfloat>::allocate(  numcol * numrow), [](MPSfloat* p) {
            Memory<MPSfloat>::free(p);
        });

    mpsSynchronize();
    memcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float));

}

Matrix<MPSfloat>::Matrix(Matrix<MPSfloat>&& M) noexcept {
    LOG_DEBUG("mps move constructor called");
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
}

Matrix<MPSfloat>& Matrix<MPSfloat>::operator=(const Matrix<MPSfloat>& M) {
    if (this == &M) return *this;
    LOG_DEBUG("mps copy assignment called");
    this->name = "copy of " + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<MPSfloat[]>(
        Memory<MPSfloat>::allocate(  numcol * numrow), [](MPSfloat* p) {
            Memory<MPSfloat>::free(p);
        });
    
    mpsSynchronize();
    memcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float));

    return *this;
}

Matrix<MPSfloat>& Matrix<MPSfloat>::operator=(Matrix<MPSfloat>&& M) noexcept {
    if (this == &M) return *this;
    LOG_DEBUG("mps move assignment called");
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = M.elements;
    M.elements = nullptr;
    return *this;
}

Matrix<float> Matrix<MPSfloat>::to_host() const
{
    Matrix<float> M((name + "->host").c_str(), numrow, numcol, transpose);
    mpsSynchronize();
    //change row major to column major
    vDSP_mtrans((const float*)elements.get(), 1, (float*)M.elements.get(), 1,
                numcol, numrow);
    return M;
}

Matrix<MPSfloat> Matrix<MPSfloat>::dot(const Matrix<MPSfloat>& B) const
{
    STATIC_TIC;
    //check if the dimensions are compatible
    if (num_col() != B.num_row())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<MPSfloat> C("dot", num_row(), B.num_col());
    mpsGemm((float*) elements.get(), (float*) B.elements.get(), (float*) C.elements.get(),
            numrow, numcol, B.numrow, B.numcol, transpose, B.get_transpose());
    return C;
}

//s1*M + s2*B
Matrix<MPSfloat> Matrix<MPSfloat>::add(const Matrix<MPSfloat>& B, float s1, float s2) const
{
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<MPSfloat> C("add", numrow, numcol, transpose);

    mpsAdd((float*) B.elements.get(), (float*) elements.get(), (float*) C.elements.get(),
           B.numrow, B.numcol, transpose != B.transpose, s2, s1);

    return C;
}
//in place s1*M + s2*B
void Matrix<MPSfloat>::add(const Matrix<MPSfloat>& B, float s1, float s2) {
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsAdd((float*) B.elements.get(), (float*) elements.get(), (float*) elements.get(),
           B.numrow, B.numcol, transpose != B.transpose, s2, s1);

}

//s1*M + a
Matrix<MPSfloat> Matrix<MPSfloat>::add(float a, float s1) const
{
    Matrix<MPSfloat> C("add", numrow, numcol, transpose);

    mpsAx_b((float*) elements.get(), s1, a, (float*) C.elements.get(), numrow * numcol);
    return C;
}

//rvalue s1*M + a
void Matrix<MPSfloat>::add(float a, float s1)
{
    mpsAx_b((float*) elements.get(), s1, a, (float*) elements.get(), numrow * numcol);
}

//M = s1*M
void Matrix<MPSfloat>::scale(float s1){
    //original content lM is replaced to store the scaled result.
    mpsAx_b((float*) elements.get(), s1, 0, (float*) elements.get(), numrow * numcol);
}

// M_new = M / l
Matrix<MPSfloat> Matrix<MPSfloat>::eleminv(double l) const 
{
    // std::cout << "eleminv 1" << std::endl;
    Matrix<MPSfloat> M("reci", numrow, numcol, transpose);
    mpsElemInv((float*) elements.get(), (float*) M.elements.get(), numrow * numcol, l);
    return M;
}

// M = M / l
void Matrix<MPSfloat>::eleminv(double l)
{
    // std::cout << "eleminv 2" << std::endl;
    mpsElemInv((float*) elements.get(), (float*) elements.get(), numrow * numcol, l);
}

// Accumulates in double over the shared buffer, so one sync replaces a
// device reduction plus a readback.
float Matrix<MPSfloat>::norm() const
{
    mpsSynchronize();
    const float* values = (const float*)elements.get();
    double total = 0;
    for (size_t i = 0; i < numrow * numcol; ++i) total += double(values[i]) * values[i];
    return (float)std::sqrt(total);
}

const Matrix<MPSfloat> Matrix<MPSfloat>::T() const
{
    std::string newname = name + "_T";
    Matrix<MPSfloat> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}

// C = M1 .* M2
Matrix<MPSfloat> hadmd(const Matrix<MPSfloat>& M1, const Matrix<MPSfloat>& M2){
    // check if the size of the two matrices are the same
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    Matrix<MPSfloat> result("hadmd", M1.numrow, M1.numcol, M1.transpose);
    mpsProduct((float*) M2.elements.get(), (float*) M1.elements.get(), (float*) result.elements.get(),
               M2.numrow, M2.numcol, M1.transpose != M2.transpose);
    return result;
}

// M2 = M1 .* M2
Matrix<MPSfloat> hadmd(const Matrix<MPSfloat>& M1, Matrix<MPSfloat>&& M2){
    // check if the size of the two matrices are the same
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsProduct((float*) M1.elements.get(), (float*) M2.elements.get(), (float*) M2.elements.get(),
               M1.numrow, M1.numcol, M1.transpose != M2.transpose);
    return std::move(M2);
}

// M1 = M1 .* M2
Matrix<MPSfloat> hadmd(Matrix<MPSfloat>&& M1, const Matrix<MPSfloat>& M2){
    // check if the size of the two matrices are the same
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsProduct((float*) M2.elements.get(), (float*) M1.elements.get(), (float*) M1.elements.get(),
               M2.numrow, M2.numcol, M1.transpose != M2.transpose);
    return std::move(M1);
}

void Matrix<MPSfloat>::zeros()
{
    mpsFill((float*) elements.get(), numrow * numcol, 0.0);
}
void Matrix<MPSfloat>::ones()
{
    mpsFill((float*) elements.get(), numrow * numcol, 1.0);
}
Matrix<MPSfloat> Matrix<MPSfloat>::rand(size_t m, size_t n)
{
    Matrix<MPSfloat> M("rand", m, n);
    mpsRand((float*) M.elements.get(), m * n);
    return M;
}
Matrix<MPSfloat> Matrix<MPSfloat>::randn(size_t m, size_t n)
{
    // Box-Muller over the device uniform stream; finite by construction.
    Matrix<MPSfloat> M("randn", m, n);
    mpsRandn((float*) M.elements.get(), m * n);
    return M;
}

Matrix<MPSfloat> Matrix<MPSfloat>::zeros(size_t m, size_t n)
{
    Matrix<MPSfloat> M("zeros", m, n);
    mpsFill((float*) M.elements.get(), m * n, 0.0);
    return M;
}

Matrix<MPSfloat> Matrix<MPSfloat>::ones(size_t m, size_t n)
{
    Matrix<MPSfloat> M("ones", m, n);
    mpsFill((float*) M.elements.get(), m * n, 1.0);
    return M;
}

Matrix<MPSfloat> tanh(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> tanhM("tanhM", M.numrow, M.numcol, M.transpose);
    mpsTanh((float*) M.elements.get(), (float*) tanhM.elements.get(), M.numrow * M.numcol);
    return tanhM;
}
Matrix<MPSfloat> tanh(Matrix<MPSfloat>&& M){
    mpsTanh((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}
Matrix<MPSfloat> d_tanh(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> d_tanhM("d_tanhM", M.numrow, M.numcol, M.transpose);
    mpsdTanh((float*) M.elements.get(), (float*) d_tanhM.elements.get(), M.numrow * M.numcol);
    return d_tanhM;
}
Matrix<MPSfloat> d_tanh(Matrix<MPSfloat>&& M){
    mpsdTanh((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> sqrt(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> sqrtM("sqrtM", M.numrow, M.numcol, M.transpose);
    mpsSqrt((float*) M.elements.get(), (float*) sqrtM.elements.get(), M.numrow * M.numcol);
    return sqrtM;
}
Matrix<MPSfloat> sqrt(Matrix<MPSfloat>&& M){
    mpsSqrt((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> exp(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> expM("expM", M.numrow, M.numcol, M.transpose);
    mpsExp((float*) M.elements.get(), (float*) expM.elements.get(), M.numrow * M.numcol);
    return expM;
}

Matrix<MPSfloat> exp(Matrix<MPSfloat>&& M){
    mpsExp((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> log(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> logM("logM", M.numrow, M.numcol, M.transpose);
    mpsLog((float*) M.elements.get(), (float*) logM.elements.get(), M.numrow * M.numcol);
    return logM;
}

Matrix<MPSfloat> log(Matrix<MPSfloat>&& M){
    mpsLog((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> square(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> squareM("squareM", M.numrow, M.numcol, M.transpose);
    mpsSquare((float*) M.elements.get(), (float*) squareM.elements.get(), M.numrow * M.numcol);
    return squareM;
}

Matrix<MPSfloat> square(Matrix<MPSfloat>&& M){
    mpsSquare((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> relu(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> reluM("reluM", M.numrow, M.numcol, M.transpose);
    mpsRelu((float*) M.elements.get(), (float*) reluM.elements.get(), M.numrow * M.numcol);
    return reluM;
}

Matrix<MPSfloat> relu(Matrix<MPSfloat>&& M){
    mpsRelu((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> d_relu(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> d_reluM("d_reluM", M.numrow, M.numcol, M.transpose);
    mpsdRelu((float*) M.elements.get(), (float*) d_reluM.elements.get(), M.numrow * M.numcol);
    return d_reluM;
}
Matrix<MPSfloat> d_relu(Matrix<MPSfloat>&& M){
    mpsdRelu((float*) M.elements.get(), (float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat>& fill(Matrix<MPSfloat>& M, double a) {
    mpsFill((float*) M.elements.get(), M.numrow * M.numcol, (float) a);
    return M;
}

void copy(Matrix<MPSfloat>& dest, const Matrix<MPSfloat>& src) {
    const size_t count = src.numrow * src.numcol;
    auto storage = std::shared_ptr<MPSfloat[]>(Memory<MPSfloat>::allocate(count),
                                               [](MPSfloat* p) { Memory<MPSfloat>::free(p); });
    // Copy the physical buffer as a single row; the layout flag carries over.
    mpsCopyMatrixBlock((const float*) src.elements.get(), (float*) storage.get(),
                       1, (int) count, false, 0, 0, 1, (int) count,
                       1, (int) count, false, 0, 0);
    dest.numrow = src.numrow;
    dest.numcol = src.numcol;
    dest.transpose = src.transpose;
    dest.elements = std::move(storage);
}

Matrix<MPSfloat> hstack(std::vector<MatrixView<MPSfloat>> matrices) {
    if (matrices.empty()) {
        throw std::invalid_argument("hstack: input list is empty");
    }
    const size_t num_row = matrices[0].num_row();
    size_t num_col = 0;
    for (size_t i = 0; i < matrices.size(); ++i) {
        if (matrices[i].num_row() != num_row) {
            throw std::invalid_argument("hstack: all matrices must have the same row count");
        }
        num_col += matrices[i].num_col();
    }
    Matrix<MPSfloat> out("hstack", num_row, num_col, 0);
    size_t col_index = 0;
    for (size_t i = 0; i < matrices.size(); ++i) {
        mpsCopyMatrixBlock((const float*)matrices[i].data(), (float*)out.elements.get(),
                           (int)matrices[i].num_row(), (int)matrices[i].num_col(), (bool)matrices[i].get_transpose(),
                           0, 0,
                           (int)matrices[i].num_row(), (int)matrices[i].num_col(),
                           (int)out.num_row(), (int)out.num_col(), (bool)out.get_transpose(),
                           0, (int)col_index);
        col_index += matrices[i].num_col();
    }
    return out;
}

Matrix<MPSfloat> vstack(std::vector<MatrixView<MPSfloat>> matrices) {
    if (matrices.empty()) {
        throw std::invalid_argument("vstack: input list is empty");
    }
    size_t num_row = 0;
    const size_t num_col = matrices[0].num_col();
    for (size_t i = 0; i < matrices.size(); ++i) {
        if (matrices[i].num_col() != num_col) {
            throw std::invalid_argument("vstack: all matrices must have the same column count");
        }
        num_row += matrices[i].num_row();
    }
    Matrix<MPSfloat> out("vstack", num_row, num_col, 0);
    size_t row_index = 0;
    for (size_t i = 0; i < matrices.size(); ++i) {
        mpsCopyMatrixBlock((const float*)matrices[i].data(), (float*)out.elements.get(),
                           (int)matrices[i].num_row(), (int)matrices[i].num_col(), (bool)matrices[i].get_transpose(),
                           0, 0,
                           (int)matrices[i].num_row(), (int)matrices[i].num_col(),
                           (int)out.num_row(), (int)out.num_col(), (bool)out.get_transpose(),
                           (int)row_index, 0);
        row_index += matrices[i].num_row();
    }
    return out;
}

Matrix<MPSfloat> sum(const Matrix<MPSfloat>& M, int dim){
    int transM = M.transpose;
    if (dim == 0)
    {
        transM = !transM;
    }

    Matrix<MPSfloat> sumM("sumM", transM ? M.numcol : M.numrow, 1, 0);
    Matrix<MPSfloat> ones = Matrix<MPSfloat>::ones(transM ? M.numrow : M.numcol, 1);

    mpsGemv((float*) M.elements.get(), (float*) ones.elements.get(), 
            (float*) sumM.elements.get(), M.numrow, M.numcol, transM);

    if (dim == 0)
    {
        sumM.transpose = true;
    }
    return sumM;
}

Matrix<float> topk(const Matrix<MPSfloat>& M, int k, int dim){

    // STATIC_TIC;
    bool trans = false;
    if (dim == 1 && !M.transpose) {
        trans = false;
    } else if (dim == 0 && !M.transpose) {
        trans = true;
    } else if (dim == 1 && M.transpose) {
        trans = true;
    } else {
        trans = false;
    }

    if (!trans){
        Matrix<float> result("resM", M.numrow, k);
        Matrix<MPSfloat> kval("kval", M.numrow, k);
        mpsTopk((float*) M.elements.get(), result.elements.get(), (float *) kval.elements.get(), M.numrow, M.numcol, k);

        if (M.transpose)
        {
            return result.T();
        }
        else
        {
            return result;
        }
    }else{
        // transpose the matrix
        auto t = Matrix<MPSfloat>::zeros(M.numcol, M.numrow);
        if (M.get_transpose())
            t += M;
        else
            t += M.T();

        Matrix<float> result("resM", t.numrow, k);
        Matrix<MPSfloat> kval("kval", t.numrow, k);
        mpsTopk((float*) t.elements.get(), result.elements.get(), (float *) kval.elements.get(), t.numrow, t.numcol, k);

        if (M.transpose) {
            return result;
        } else {
            return result.T();
        }
    }

}

Matrix<MPSfloat> Matrix<MPSfloat>::slice(size_t rstart, size_t rend,
                                         size_t cstart, size_t cend) const {
    if (rstart > rend || cstart > cend || rend > num_row() || cend > num_col())
        throw std::out_of_range("MPS slice indices out of range");
    Matrix<MPSfloat> out("submatrix", rend - rstart, cend - cstart);
    mpsCopyMatrixBlock((const float*)elements.get(), (float*)out.elements.get(),
                       (int)num_row(), (int)num_col(), (bool)get_transpose(),
                       (int)rstart, (int)cstart,
                       (int)(rend - rstart), (int)(cend - cstart),
                       (int)out.num_row(), (int)out.num_col(), (bool)out.get_transpose(),
                       0, 0);
    return out;
}

void Matrix<MPSfloat>::slice(size_t rstart, size_t rend, size_t cstart,
                             size_t cend, const Matrix<MPSfloat>& M) {
    if (rstart > rend || cstart > cend || rend > num_row() || cend > num_col())
        throw std::out_of_range("MPS slice assignment indices out of range");
    const size_t rows = rend - rstart, cols = cend - cstart;
    if (M.num_row() != rows || M.num_col() != cols)
        throw std::invalid_argument("MPS slice assignment dimensions do not match target range");
    if (!rows || !cols) return;
    // T() shares storage; preserve the source before any overlapping write.
    if (elements.get() == M.elements.get()) {
        Matrix<MPSfloat> snapshot(nullptr);
        copy(snapshot, M);  // device copy: no host synchronization
        slice(rstart, rend, cstart, cend, snapshot);
        return;
    }
    mpsCopyMatrixBlock((const float*)M.elements.get(), (float*)elements.get(),
                       (int)M.num_row(), (int)M.num_col(), (bool)M.get_transpose(),
                       0, 0,
                       (int)rows, (int)cols,
                       (int)num_row(), (int)num_col(), (bool)get_transpose(),
                       (int)rstart, (int)cstart);
}

Matrix<MPSfloat> Matrix<MPSfloat>::slice(const idxlist& rowidx, const idxlist& colidx) const {
    for (size_t row : rowidx) if (row >= num_row()) throw std::out_of_range("MPS gather row index");
    for (size_t col : colidx) if (col >= num_col()) throw std::out_of_range("MPS gather column index");
    const size_t limit = std::numeric_limits<int>::max();
    if (rowidx.size() > limit || colidx.size() > limit ||
        (colidx.size() && rowidx.size() > limit / colidx.size()))
        throw std::length_error("MPS gather size overflows");
    Matrix<MPSfloat> result("submatrix", rowidx.size(), colidx.size());
    if (rowidx.empty() || colidx.empty()) return result;
    const std::vector<unsigned> rows(rowidx.begin(), rowidx.end()), cols(colidx.begin(), colidx.end());
    mpsGather((const float*)elements.get(), (float*)result.elements.get(),
              rows.data(), (int)rows.size(), cols.data(), (int)cols.size(),
              (int)num_row(), (int)num_col(), (bool)transpose);
    return result;
}

template <>
void write(FILE* fp, const Matrix<MPSfloat>& M) {
    write(fp, M.to_host());
}

template <>
void read(FILE* fp, Matrix<MPSfloat>& M) {
    Matrix<float> tmp("tmp", M.num_row(), M.num_col());
    read(fp, tmp);
    if (ferror(fp) || feof(fp)) return;
    // Upload into fresh device storage with the file's shape and layout flag.
    Matrix<MPSfloat> restored(tmp);
    restored.name = M.name;
    M = std::move(restored);
}

std::ostream& operator <<(std::ostream& os, const Matrix<MPSfloat>& M) {
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