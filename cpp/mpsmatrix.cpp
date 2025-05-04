#include <string>
#include "operators.hpp"
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

    mpsSynchronize(); // before the upload, make sure all the previous operations are done
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

//s1*M + s2*B
Matrix<MPSfloat> Matrix<MPSfloat>::add(const Matrix<MPSfloat>& B, float s1, float s2) const
{
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<MPSfloat> C("add", numrow, numcol, transpose);

    mpsAx_b((float*) elements.get(), 1.0, 0, (float*) C.elements.get(), numrow * numcol);
    mpsAdd((float*) B.elements.get(), (float *) C.elements.get(), B.numrow, B.numcol, transpose != B.transpose, s2, s1);
    

    return C;
}
//in place s1*M + s2*B
void Matrix<MPSfloat>::add(const Matrix<MPSfloat>& B, float s1, float s2) {
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsAdd((float*) B.elements.get(), (float*) elements.get(), B.numrow, B.numcol, transpose != B.transpose, s2, s1);

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
    mpsAx_b((float*) elements.get(), 1.0, 0, (float*) M.elements.get(), numrow * numcol);
    mpsElemInv((float*) M.elements.get(), numrow * numcol, l);
    return M;
}

// M = M / l
void Matrix<MPSfloat>::eleminv(double l)
{
    // std::cout << "eleminv 2" << std::endl; 
    mpsElemInv((float*) elements.get(), numrow * numcol, l);
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
    mpsAx_b((float*) M1.elements.get(), 1.0, 0.0, (float*) result.elements.get(), M1.numrow * M1.numcol);

    mpsProduct((float*) M2.elements.get(), (float*) result.elements.get(), M2.numrow, M2.numcol, M1.transpose != M2.transpose);
    return result;
}

// M2 = M1 .* M2
Matrix<MPSfloat> hadmd(const Matrix<MPSfloat>& M1, Matrix<MPSfloat>&& M2){
    // check if the size of the two matrices are the same
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsProduct((float*) M1.elements.get(), (float*) M2.elements.get(), M1.numrow, M1.numcol, M1.transpose != M2.transpose);
    return std::move(M2);
}

// M1 = M1 .* M2
Matrix<MPSfloat> hadmd(Matrix<MPSfloat>&& M1, const Matrix<MPSfloat>& M2){
    // check if the size of the two matrices are the same
    if (M1.num_row() != M2.num_row() || M1.num_col() != M2.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    mpsProduct((float*) M2.elements.get(), (float*) M1.elements.get(), M2.numrow, M2.numcol, M1.transpose != M2.transpose);
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

Matrix<MPSfloat> tanh(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> tanhM("tanhM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) tanhM.elements.get(), M.numrow * M.numcol);
    mpsTanh((float*) tanhM.elements.get(), M.numrow * M.numcol);
    return tanhM;
}
Matrix<MPSfloat> tanh(Matrix<MPSfloat>&& M){
    mpsTanh((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}
Matrix<MPSfloat> d_tanh(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> d_tanhM("d_tanhM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) d_tanhM.elements.get(), M.numrow * M.numcol);
    mpsdTanh((float*) d_tanhM.elements.get(), M.numrow * M.numcol);
    return d_tanhM;
}
Matrix<MPSfloat> d_tanh(Matrix<MPSfloat>&& M){
    mpsdTanh((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> sqrt(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> sqrtM("sqrtM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) sqrtM.elements.get(), M.numrow * M.numcol);
    mpsSqrt((float*) sqrtM.elements.get(), M.numrow * M.numcol);
    return sqrtM;
}
Matrix<MPSfloat> sqrt(Matrix<MPSfloat>&& M){
    mpsSqrt((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> exp(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> expM("expM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) expM.elements.get(), M.numrow * M.numcol);
    mpsExp((float*) expM.elements.get(), M.numrow * M.numcol);
    return expM;
}

Matrix<MPSfloat> exp(Matrix<MPSfloat>&& M){
    mpsExp((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> log(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> logM("logM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) logM.elements.get(), M.numrow * M.numcol);
    mpsLog((float*) logM.elements.get(), M.numrow * M.numcol);
    return logM;
}

Matrix<MPSfloat> log(Matrix<MPSfloat>&& M){
    mpsLog((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> square(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> squareM("squareM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) squareM.elements.get(), M.numrow * M.numcol);
    mpsSquare((float*) squareM.elements.get(), M.numrow * M.numcol);
    return squareM;
}

Matrix<MPSfloat> square(Matrix<MPSfloat>&& M){
    mpsSquare((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> relu(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> reluM("reluM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) reluM.elements.get(), M.numrow * M.numcol);
    mpsRelu((float*) reluM.elements.get(), M.numrow * M.numcol);
    return reluM;
}

Matrix<MPSfloat> relu(Matrix<MPSfloat>&& M){
    mpsRelu((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
}

Matrix<MPSfloat> d_relu(const Matrix<MPSfloat>& M){
    Matrix<MPSfloat> d_reluM("d_reluM", M.numrow, M.numcol, M.transpose);
    mpsAx_b((float*) M.elements.get(), 1.0, 0, (float*) d_reluM.elements.get(), M.numrow * M.numcol);
    mpsdRelu((float*) d_reluM.elements.get(), M.numrow * M.numcol);
    return d_reluM;
}
Matrix<MPSfloat> d_relu(Matrix<MPSfloat>&& M){
    mpsdRelu((float*) M.elements.get(), M.numrow * M.numcol);
    return std::move(M);
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