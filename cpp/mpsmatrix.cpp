#include <string>
#include "mpsmatrix.hpp"

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

    profiler.start();
    //change column major to row major
    for (size_t i = 0; i < numrow; i++)
    {
        for (size_t j = 0; j < numcol; j++)
        {
            elements[i * numcol + j] = M.elements[j * numrow + i];
        }
    }
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
    for (size_t i = 0; i < numrow; i++)
    {
        for (size_t j = 0; j < numcol; j++)
        {
            M.elements[j * numrow + i] = elements[i * numcol + j];
        }
    }
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

Matrix<MPSfloat> Matrix<MPSfloat>::T() const
{
    std::string newname = name + "_T";
    Matrix<MPSfloat> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}

Matrix<MPSfloat> Matrix<MPSfloat>::randn(size_t m, size_t n)
{
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