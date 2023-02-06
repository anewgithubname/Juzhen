/**
 * @file matrix.hpp
 * @brief Stand alone or less important member functions for the Matrix class. 
 * @author Song Liu (song.liu@bristol.ac.uk)
 * 
 *   Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#ifndef MATRIX_H
#define MATRIX_H
#include "core.hpp"
#include <concepts>

template<typename D>
concept Addable = requires(Matrix<D> &&A, const Matrix<D> &B, const Matrix<D> &C, D d)
{
    d = d+1;
    A.add(B,1,1); //self add
    {C.add(B,1,1)} ->  std::same_as<Matrix<D>>; // const add
};

template<typename D>
concept ScalarAddable = requires(Matrix<D> &&A, const Matrix<D> &C, D d)
{
    d = d+1;
    A.add(1,1); //self add
    {C.add(1,1)} ->  std::same_as<Matrix<D>>; // const add
};

template<typename D>
concept Scalable = requires(Matrix<D> &&A, const Matrix<D> &C, D d)
{
    d = d+1;
    A.scale((D) 1); //self add
    {C.scale((D) 1)} ->  std::same_as<Matrix<D>>; // const add
};

template<typename D>
concept Multiplicable = requires(const Matrix<D> && A, const Matrix<D> &B, D d)
{
    d = d + 1;
    {A.dot(B)} ->  std::same_as<Matrix<D>>; // const add
};


//matrix filler funcitons. 
template<class D>
Matrix<D> Matrix<D>::ones(size_t m, size_t n)
{
    static Profiler p("ones"); p.start();
    Matrix<D> M("ones", m, n);
#pragma ivdep
    for (size_t i = 0; i < m*n; i++)
    {
        M.elements[i] = 1; 
    }
    p.end();
    return M;
}

template<class D>
inline Matrix<D> Matrix<D>::zeros(size_t m, size_t n)
{
    Matrix<D> M("zeros", m, n);
#pragma ivdep
    for (size_t i = 0; i < m * n; i++)
    {
        M.elements[i] = 0;
    }
    return M;
}

template<class D>
inline Matrix<D> Matrix<D>::randn(size_t m, size_t n)
{
    using namespace std;
	normal_distribution<> d(0, 1);
    Matrix<D> M("randn", m, n);
	//cannot be vectorized, due to the implementation of std::random.
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
	for (size_t i = 0; i < numelems; i++)
		M.elements[i] = (D) d(randomnumber_gen);
    
    return M;
}

template<class D>
inline Matrix<D> Matrix<D>::rand(size_t m, size_t n)
{
    using namespace std;
    uniform_real_distribution<> d(0, 1);
    Matrix<D> M("rand", m, n);
    //cannot be vectorized, due to the implementation of std::random.
    for (size_t i = 0; i < M.num_row() * M.num_col(); i++)
        M.elements[i] = (D) d(randomnumber_gen);

    return M;
}

//sum function as the "sum" in MATLAB
template <class D>
Matrix<D> sum(const Matrix<D> &M, int dim)
{
#ifdef NO_CBLAS
    size_t n = M.num_row();
    size_t m = M.num_col();

    if (dim == 0)
    {
        Matrix<D> sumM("sumM", 1, M.num_col(), 0);
        sumM.zeros();

        for (size_t j = 0; j < m; j++)
        {
            #pragma clang loop vectorize(enable)
            for (size_t i = 0; i < n; i++)
            {
                sumM.elem(0, j) += M.elem(i, j);
            }
        }
        return sumM;
    }else{
        Matrix<D> sumM("sumM", M.num_row(), 1, 0);
        sumM.zeros();
        
        for (size_t j = 0; j < m; j++)
        {
            #pragma clang loop vectorize(enable)
            for (size_t i = 0; i < n; i++)
            {
                sumM.elem(i, 0) += M.elem(i, j);
            }
        }
        return sumM;
    }
#else
    int transM = M.transpose;
    if (dim == 0)
    {
        transM = !transM; 
    }

    Matrix<D> sumM("sumM", transM?M.numcol:M.numrow, 1, 0);
    Matrix<D> ones("ones", transM?M.numrow:M.numcol, 1, 0);
    ones.ones();
    CBLAS_TRANSPOSE cBlasTransM = transM ? CblasTrans : CblasNoTrans;
    gemv(cBlasTransM, M.numrow, M.numcol,
                1.0f, M.elements.get(), M.numrow, ones.elements.get(),
                1, 0.0f, sumM.elements.get(), 1);
    if (dim == 0)
    {
        sumM.transpose = 1;
    }
    return sumM;
#endif
}
//hstack function as the "hstack" in NumPy
template <class D>
Matrix<D> hstack(const std::vector<Matrix<D>> &matrices)
{
    size_t num_row = matrices[0].num_row();
    size_t num_col = 0;
    for (size_t i = 0; i < matrices.size(); i++)
    {
        num_col += matrices[i].num_col();
    }
    Matrix<D> result("hstack", num_row, num_col, 0);

    size_t col_index = 0;
    for (size_t i = 0; i < matrices.size(); i++)
    {
        for (size_t k = 0; k < matrices[i].num_col(); k++)
        {
            for (size_t j = 0; j < matrices[i].num_row(); j++)
            {
                result.elem(j, col_index + k) = matrices[i].elem(j, k);
            }
        }
        col_index += matrices[i].num_col();
    }

    return result;
}

template <class D> 
Matrix<D> hstack(std::vector<Matrix<D>>&& matrices)
{
    //rval hstack
    return hstack(matrices);
}

//vstack function as the "vstack" in NumPy
template <class D>
Matrix<D> vstack(const std::vector<Matrix<D>> &matrices)
{
    size_t num_row = 0;
    size_t num_col = matrices[0].num_col();
    for (size_t i = 0; i < matrices.size(); i++)
    {
        num_row += matrices[i].num_row();
    }
    Matrix<D> result("vstack", num_row, num_col, 0);

    size_t row_index = 0;
    for (size_t i = 0; i < matrices.size(); i++)
    {
        for (size_t k = 0; k < matrices[i].num_col(); k++)
        {
            for (size_t j = 0; j < matrices[i].num_row(); j++)
            {
                result.elem(row_index + j, k) = matrices[i].elem(j, k);
            }
        }
        row_index += matrices[i].num_row();
    }

    return result;
}

template <class D>
Matrix<D> vstack(std::vector<Matrix<D>>&& matrices)
{
    //rval vstack
    return vstack(matrices);
}

//return a copy of some continuous rows
template <class D>
Matrix<D> Matrix<D>::rows(size_t start, size_t end) const
{
    return slice(start, end, 0, num_col());
}
//assign continuous rows
template <class D>
void Matrix<D>::rows(size_t start, size_t end, const Matrix<D> &M)
{
    slice(start, end, 0, num_col(), M);
}

//return a copy of some continuous columns
template <class D>
Matrix<D> Matrix<D>::columns(size_t start, size_t end) const
{
    return slice(0, num_row(), start, end);
}
//assign continuous columns
template <class D>
void Matrix<D>::columns(size_t start, size_t end, const Matrix<D> &M)
{
    slice(0, num_row(), start, end, M);
}

//return a copy of some non-continuous columns
template <class D>
Matrix<D> Matrix<D>::columns(idxlist cols) const
{
    return slice(seq(num_row()), cols);
}
//return a copy of some non-continuous rows
template <class D>
Matrix<D> Matrix<D>::rows(idxlist rows) const
{
    return slice(rows, seq(num_col()));
}

template <class D, class Function>
Matrix<D> elemwise(Function func, const Matrix<D> &M) {
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> res("res", M.numrow, M.numcol, M.transpose);
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < numelems; i++)
        res.elements[i] = func(M.elements[i]);

    return res;
}

template <class D, class Function>
Matrix<D> elemwise(Function func, Matrix<D> &&M) {
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = func(M.elements[i]);

    return std::move(M);
}

template<class D, class Function>
Matrix<D> reduce(Function func, const Matrix<D>& M, int dim, int k)
{   
    // STATIC_TIC;
    bool trans = false;
    if (dim == 0 && !M.transpose) {
        trans = false;
    }else if(dim == 1 && !M.transpose){
        trans = true;
    }else if(dim == 0 && M.transpose){
        trans = true; 
    }else{
        trans = false;
    }

    if (!trans) {
        Matrix<D> result("resM", k, M.numcol);
        // reduce_kernel << <cudaConfig(M.numcol) >> > (func, (float*)result.elements.get(), (float*)M.elements.get(), M.numrow, k, M.numcol);
        for (size_t i = 0; i < M.numcol; i++) {
            func(&M.elements.get()[i*M.numrow], &result.elements.get()[i*k], M.numrow, k);
        }
        if(M.transpose){
            return result.T();
        }else{
            return result;
        }
    }
    else {
        //transpose the matrix
        Matrix<D> t("tzeros", M.numcol, M.numrow);
        t.zeros();
        t += M.transpose? M : M.T();

        Matrix<D> result("resM", k, t.numcol);
        // reduce_kernel << <cudaConfig(t.numcol) >> > (func, (float*)result.elements.get(), (float*)t.elements.get(), t.numrow, k, t.numcol);
        for (size_t i = 0; i < t.numcol; i++) {
            func(&t.elements.get()[i*t.numrow], &result.elements.get()[i*k], t.numrow, k);
        }
        if(M.transpose){
            return result; 
        }else{
            return result.T();
        }
    }
    
}

template <class D>
Matrix<D> exp(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> expM("expM", M.numrow, M.numcol, M.transpose);
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < numelems; i++)
        expM.elements[i] = exp(M.elements[i]);

    return expM;
}

template <class D>
Matrix<D> exp(Matrix<D> &&M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = exp(M.elements[i]);

    return std::move(M);
}

//TODO: add rvalue version in the future
template <class D>
Matrix<D> log(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> logM("logM", M.numrow, M.numcol, M.transpose);
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < numelems; i++)
        logM.elements[i] = log(M.elements[i]);

    return logM;
}
template <class D>
Matrix<D> tanh(const Matrix<D> &M)
{
    static Profiler p("tanh left"); p.start();
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> tanhM("tanh", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        tanhM.elements[i] = tanh(M.elements[i]);

    p.end();
    return tanhM;
}
//rvalue overload
template <class D>
Matrix<D> tanh(Matrix<D> &&M)
{
    static Profiler p("tanh right"); p.start();
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = tanh(M.elements[i]);

    p.end();
    return std::move(M);
}
template <class D>
Matrix<D> d_tanh(const Matrix<D> &M)
{
    static Profiler p("dtanh left"); p.start();
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> d_tanhM("tanh", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        d_tanhM.elements[i] = 1- tanh(M.elements[i])*tanh(M.elements[i]);

    p.end();
    return d_tanhM;
}
//rvalue overload
template <class D>
Matrix<D> d_tanh(Matrix<D> && M)
{
    static Profiler p("dtanh right"); p.start();
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = 1- tanh(M.elements[i])*tanh(M.elements[i]);
    
    p.end();
    return std::move(M);
}

//lvalue atan_exp
template <class D>
Matrix<D> atan_exp(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> atan_expM("atan_exp", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        atan_expM.elements[i] = atan(exp(M.elements[i]));

    return atan_expM;
}

//rvalue atan_exp
template <class D>
Matrix<D> atan_exp(Matrix<D> && M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = atan(exp(M.elements[i]));

    return std::move(M);
}

//lvalue d_atan_exp
template <class D>
Matrix<D> d_atan_exp(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> d_atan_expM("d_atan_exp", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        d_atan_expM.elements[i] = exp(M.elements[i]) / (1 + exp(2*M.elements[i]));

    return d_atan_expM;
}

//rvalue d_atan_exp
template <class D>
Matrix<D> d_atan_exp(Matrix<D> && M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = exp(M.elements[i]) / (1 + exp(2*M.elements[i]));

    return std::move(M);
}

//lvalue sin
template <class D>
Matrix<D> sin(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> sinM("sin", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        sinM.elements[i] = sin(M.elements[i]);

    return sinM;
}

//rvalue sin
template <class D>
Matrix<D> sin(Matrix<D> && M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = sin(M.elements[i]);

    return std::move(M);
}

//lvalue cos
template <class D>
Matrix<D> cos(const Matrix<D> &M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> cosM("cos", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        cosM.elements[i] = cos(M.elements[i]);

    return cosM;
}

//rvalue cos
template <class D>
Matrix<D> cos(Matrix<D> && M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] = cos(M.elements[i]);

    return std::move(M);
}

template<class D>
inline Matrix<D> square(const Matrix<D>& M)
{
    size_t numelems = M.num_row() * M.num_col();
    Matrix<D> res("square", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        res.elements[i] = M.elements[i] * M.elements[i];

    return res;
}

//in place square
template<class D>
inline Matrix<D> square(Matrix<D>&& M)
{
    size_t numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        M.elements[i] *= M.elements[i];

    return std::move(M);
}

template <class D>
Matrix<D> hadmd(const Matrix<D> &M1, const Matrix<D> &M2)
{
    size_t numcol = M1.num_col();
    size_t numrow = M1.num_row();
    Matrix<D> result("hadmd", M1.num_row(), M1.num_col());
    for (size_t j=0; j< numcol; j++){
#pragma ivdep
        for (size_t i=0; i< numrow; i++){
            result.elem(i, j) = M1.elem(i, j) * M2.elem(i, j);

        }
    }
    return result;
}
//rvalue overload
template <class D>
Matrix<D> hadmd(const Matrix<D> &M1, Matrix<D> &&M2)
{
    size_t numcol = M1.num_col();
    size_t numrow = M1.num_row();
    for (size_t j=0; j< numcol; j++)
#pragma ivdep
        for (size_t i=0; i< numrow; i++)
            M2.elem(i, j) *= M1.elem(i, j);
    return std::move(M2);
}
//rvalue overload
template <class D>
Matrix<D> hadmd(Matrix<D> &&M1, const Matrix<D> &M2)
{
    size_t numcol = M1.num_col();
    size_t numrow = M1.num_row();
    for (size_t j=0; j< numcol; j++)
#pragma ivdep
        for (size_t i=0; i< numrow; i++)
            M1.elem(i, j) *= M2.elem(i, j);
    return std::move(M1);
}

//operator overloads
template <class D>
std::ostream &operator<<(std::ostream &os, const Matrix<D> &M)
{
    using namespace std;
    // write obj to stream
    os <<M.get_name()<< " " << M.num_row() << " by " << M.num_col();
    for (size_t i = 0; i < M.num_row(); i++)
    {
        os << endl;
        for (size_t j = 0; j < M.num_col(); j++)
        {
            os << M.elem(i, j) << " ";
        }
    }
    return os;
}
template <Multiplicable D>
Matrix<D> operator*(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.dot(rM);
}
template <Addable D>
Matrix<D> operator+(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.add(rM, 1.0, 1.0);
}
//rvalue version of operator +
template <Addable D>
Matrix<D> operator+(Matrix<D> &&lM, const Matrix<D> &rM)
{
    lM.add(rM, 1.0, 1.0);
    return std::move(lM);
}
//rvalue version of operator +
template <Addable D>
Matrix<D> operator+(Matrix<D> &&lM, Matrix<D> &&rM)
{
    lM.add(rM, 1.0, 1.0);
    return std::move(lM);
}
//rvalue version of operator +
template <Addable D>
Matrix<D> operator+(const Matrix<D> &lM, Matrix<D> &&rM)
{
    rM.add(lM, 1.0, 1.0);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D> operator+(const Matrix<D> &lM, double r)
{
    return lM.add((D) r, (D) 1.0);
}
//rvalue version of operator +
template <ScalarAddable D>
Matrix<D> operator+(Matrix<D> &&lM, double r)
{
    lM.add((D) r, (D) 1.0);
    return std::move(lM);
}
template <ScalarAddable D>
Matrix<D> operator+(double l, const Matrix<D> &rM)
{
    return rM.add(l, 1.0);
}
template <ScalarAddable D>
Matrix<D> operator+(double l, Matrix<D> &&rM)
{
    rM.add(l, 1.0);
    return std::move(rM);
}
template <Addable D>
Matrix<D> operator-(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.add(rM, 1.0, -1.0);
}
//rvalue version of operator -
template <Addable D>
Matrix<D> operator-(const Matrix<D> &lM, Matrix<D> &&rM)
{
    rM.add(lM, -1.0, 1.0);
    return std::move(rM);
}
//rvalue version of operator -
template <Addable D>
Matrix<D> operator-(Matrix<D> &&lM, const Matrix<D> &rM)
{
    lM.add(rM, 1.0, -1.0);
    return std::move(lM);
}
//rvalue version of operator -
template <Addable D>
Matrix<D> operator-(Matrix<D> &&lM, Matrix<D> &&rM)
{
    lM.add(rM, 1.0, -1.0);
    return std::move(lM);
}
template <ScalarAddable D>
Matrix<D> operator-(const Matrix<D> &lM, double r)
{
    return lM.add( - r, 1.0);
}
//rvalue version of operator -
template <ScalarAddable D>
Matrix<D> operator-(Matrix<D> &&lM, double r)
{
    lM.add( - (D) r, 1.0);
    return std::move(lM);
}
template <ScalarAddable D>
Matrix<D> operator-(double l, const Matrix<D> &rM)
{
    return rM.add(l, -1.0);
}
template <ScalarAddable D>
Matrix<D> operator-(double l, Matrix<D> &&rM)
{
    rM.add(l, -1.0);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D> operator-(const Matrix<D> &rM)
{
    return rM.add(0, -1.0);
}
template <ScalarAddable D>
Matrix<D> operator-(Matrix<D> &&rM)
{
    rM.add(0, -1.0);
    return std::move(rM);
}
template <class D>
Matrix<D> operator/(const Matrix<D> &lM, const Matrix<D> &rM)
{
    size_t numcol = lM.num_col();
    size_t numrow = lM.num_row();
    Matrix<D> result("elem_div", lM.num_row(), lM.num_col(), 0);
    for(size_t j = 0; j< numcol; j++)
    {
#pragma ivdep
        for(size_t i = 0; i< numrow; i++)
        {
            result.elem(i, j) = lM.elem(i, j) / rM.elem(i, j);
        }
    }
    return result;
}
//rvalue division
template <class D>
Matrix<D> operator/(Matrix<D> &&lM, Matrix<D> &&rM)
{
    size_t numcol = lM.num_col();
    size_t numrow = lM.num_row();
    // printf("rval /");
    for(size_t j = 0; j< numcol; j++)
    {
#pragma ivdep
        for(size_t i = 0; i< numrow; i++)
        {
            lM.elem(i, j) /= rM.elem(i, j);
        }
    }
    return std::move(lM);
}
template <class D>
Matrix<D> operator/(double l, const Matrix<D> &rM)
{
    size_t numelems = rM.num_row() * rM.num_col();
    Matrix<D> result("elem_div", rM.num_row(), rM.num_col(), 0);
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        result.elements[i] = l / rM.elements[i];

    return result;
}
//rvalue division
template <class D>
Matrix<D> operator/(double l, Matrix<D> &&rM)
{
    size_t numelems = rM.num_row() * rM.num_col();
#pragma ivdep
    for (size_t i = 0; i < numelems; i++)
        rM.elements[i] = l / rM.elements[i];

    return std::move(rM);
}
template <Scalable D>
Matrix<D> operator/(const Matrix<D> &lM, double r)
{
    return lM.scale((D) (1.0 / r));
}
//rvalue division
template <Scalable D>
Matrix<D> operator/(Matrix<D>&& lM, double r)
{
    // cout << "rval /" << endl;
    lM.scale( (D) (1 / r));
    return std::move(lM);
}
template <Scalable D>
Matrix<D> operator*(const Matrix<D> &lM, double r)
{
    return lM.scale((D)r);
}
template <Scalable D>
Matrix<D> operator*(Matrix<D> &&lM, double r)
{
    lM.scale((D)r);
    return std::move(lM);
}
template <Scalable D>
Matrix<D> operator*(double l, const Matrix<D> &rM)
{
    return rM.scale((D)l);
}
//rvalue *
template <Scalable D>
Matrix<D> operator*(double l, Matrix<D> &&rM)
{
    rM.scale(l);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D>& operator+=(Matrix<D> &lM, double r)
{
    lM.add(r,  1);
    return lM;
}
template <ScalarAddable D>
Matrix<D>& operator-=(Matrix<D> &lM, double r)
{
    lM.add(-r, 1);
    return lM;
}
template <Addable D>
Matrix<D>& operator+=(Matrix<D> &lM, const Matrix<D> &rM)
{
    lM.add(rM, 1.0, 1.0);
    return lM;
}

template <Addable D>
Matrix<D>& operator-=(Matrix<D> &lM, const Matrix<D> &rM)
{
    lM.add(rM, 1.0, -1.0);
    return lM;
}

#endif