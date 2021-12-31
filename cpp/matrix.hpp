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

//sum function as the "sum" in MATLAB
template <class D>
Matrix<D> sum(const Matrix<D> &M, int dim)
{
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
}
//hstack function as the "hstack" in NumPy
template <class D>
Matrix<D> hstack(const std::vector<Matrix<D>> &matrices)
{
    int num_row = matrices[0].num_row();
    int num_col = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        num_col += matrices[i].num_col();
    }
    Matrix<D> result("hstack", num_row, num_col, 0);

    int col_index = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        for (int k = 0; k < matrices[i].num_col(); k++)
        {
            for (int j = 0; j < matrices[i].num_row(); j++)
            {
                result.elem(j, col_index + k) = matrices[i].elem(j, k);
            }
        }
        col_index += matrices[i].num_col();
    }

    return result;
}

//vstack function as the "vstack" in NumPy
template <class D>
Matrix<D> vstack(const std::vector<Matrix<D>> &matrices)
{
    int num_row = 0;
    int num_col = matrices[0].num_col();
    for (int i = 0; i < matrices.size(); i++)
    {
        num_row += matrices[i].num_row();
    }
    Matrix<D> result("vstack", num_row, num_col, 0);

    int row_index = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        for (int k = 0; k < matrices[i].num_col(); k++)
        {
            for (int j = 0; j < matrices[i].num_row(); j++)
            {
                result.elem(row_index + j, k) = matrices[i].elem(j, k);
            }
        }
        row_index += matrices[i].num_row();
    }

    return result;
}

//return a copy of some continuous rows
template <class D>
Matrix<D> Matrix<D>::rows(int start, int end) const
{
    return slice(start, end, 0, num_col());
}
//return a copy of some continuous columns
template <class D>
Matrix<D> Matrix<D>::columns(int start, int end) const
{
    return slice(0, num_row(), start, end);
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
// TODO: Add rvalue versions in the future. 
template <class D>
Matrix<D> exp(const Matrix<D> &M)
{
    int numelems = M.num_row() * M.num_col();
    Matrix<D> expM("expM", M.numrow, M.numcol, M.transpose);
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < numelems; i++)
        expM.elements[i] = exp(M.elements[i]);

    return expM;
}

template <class D>
Matrix<D> log(const Matrix<D> &M)
{
    int numelems = M.num_row() * M.num_col();
    Matrix<D> logM("logM", M.numrow, M.numcol, M.transpose);
#pragma ivdep
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < numelems; i++)
        logM.elements[i] = log(M.elements[i]);

    return logM;
}
template <class D>
Matrix<D> tanh(const Matrix<D> &M)
{
    int numelems = M.num_row() * M.num_col();
    Matrix<D> tanhM("tanh", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        tanhM.elements[i] = tanh(M.elements[i]);

    return tanhM;
}
//rvalue overload
template <class D>
Matrix<D> tanh(Matrix<D> &&M)
{
    int numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        M.elements[i] = tanh(M.elements[i]);

    return std::move(M);
}
template <class D>
Matrix<D> d_tanh(const Matrix<D> &M)
{
    int numelems = M.num_row() * M.num_col();
    Matrix<D> d_tanhM("tanh", M.numrow, M.numcol, M.transpose);
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        d_tanhM.elements[i] = 1- tanh(M.elements[i])*tanh(M.elements[i]);

    return d_tanhM;
}
//rvalue overload
template <class D>
Matrix<D> d_tanh(Matrix<D> && M)
{
    int numelems = M.num_row() * M.num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        M.elements[i] = 1- tanh(M.elements[i])*tanh(M.elements[i]);

    return std::move(M);
}

template <class D>
Matrix<D> hadmd(const Matrix<D> &M1, const Matrix<D> &M2)
{
    int numcol = M1.num_col();
    int numrow = M1.num_row();
    Matrix<D> result("hadmd", M1.num_row(), M1.num_col());
    for (int j=0; j< numcol; j++){
#pragma ivdep
        for (int i=0; i< numrow; i++){
            // result.elem(i, j) = M1.elem(i, j) * M2.elem(i, j);
            result.elements[i*numcol+j ] = M1.elements[i*numcol+j] * M2.elements[i*numcol+j];
        }
    }
    return result;
}
//rvalue overload
template <class D>
Matrix<D> hadmd(const Matrix<D> &M1, Matrix<D> &&M2)
{
    int numcol = M1.num_col();
    int numrow = M1.num_row();
    for (int j=0; j< numcol; j++)
#pragma ivdep
        for (int i=0; i< numrow; i++)
            M2.elem(i, j) *= M1.elem(i, j);
    return std::move(M2);
}
//rvalue overload
template <class D>
Matrix<D> hadmd(Matrix<D> &&M1, const Matrix<D> &M2)
{
    int numcol = M1.num_col();
    int numrow = M1.num_row();
    for (int j=0; j< numcol; j++)
#pragma ivdep
        for (int i=0; i< numrow; i++)
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
    for (int i = 0; i < M.num_row(); i++)
    {
        os << endl;
        for (int j = 0; j < M.num_col(); j++)
        {
            os << M.elem(i, j) << " ";
        }
    }
    return os;
}
template <class D>
Matrix<D> operator*(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.dot(rM);
}
template <class D>
Matrix<D> operator+(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.add(rM, 1.0, 1.0);
}
//rvalue version of operator +
template <class D>
Matrix<D> operator+(Matrix<D> &&lM, const Matrix<D> &rM)
{
    return std::move(lM+=rM);
}
//rvalue version of operator +
template <class D>
Matrix<D> operator+(Matrix<D> &&lM, Matrix<D> &&rM)
{
    return std::move(lM += rM);
}
template <class D>
Matrix<D> operator+(const Matrix<D> &lM, const D &r)
{
    return lM.add(r, 1.0);
}
template <class D>
Matrix<D> operator+(const D &l, const Matrix<D> &rM)
{
    return rM.add(l, 1.0);
}
template <class D>
Matrix<D> operator-(const Matrix<D> &lM, const Matrix<D> &rM)
{
    return lM.add(rM, 1.0, -1.0);
}
template <class D>
Matrix<D> operator-(const Matrix<D> &lM, const D &r)
{
    return lM.add(-r, 1.0);
}
template <class D>
Matrix<D> operator-(const D &l, const Matrix<D> &rM)
{
    return rM.add(l, -1.0);
}
template <class D>
Matrix<D> operator-(const Matrix<D> &rM)
{
    return rM.add(0, -1.0);
}
template <class D>
Matrix<D> operator/(const Matrix<D> &lM, const Matrix<D> &rM)
{
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    Matrix<D> result("elem_div", lM.num_row(), lM.num_col(), 0);
    for(int j = 0; j< numcol; j++)
    {
#pragma ivdep
        for(int i = 0; i< numrow; i++)
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
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    // printf("rval /");
    for(int j = 0; j< numcol; j++)
    {
#pragma ivdep
        for(int i = 0; i< numrow; i++)
        {
            lM.elem(i, j) /= rM.elem(i, j);
        }
    }
    return std::move(lM);
}
template <class D>
Matrix<D> operator/(const D &l, const Matrix<D> &rM)
{
    int numelems = rM.num_row() * rM.num_col();
    Matrix<D> result("elem_div", rM.num_row(), rM.num_col(), 0);
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        result.elements[i] = l / rM.elements[i];

    return result;
}
//rvalue division
template <class D>
Matrix<D> operator/(const D &l, Matrix<D> &&rM)
{
    int numelems = rM.num_row() * rM.num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        rM.elements[i] = l / rM.elements[i];

    return std::move(rM);
}
template <class D>
Matrix<D> operator/(const Matrix<D> &lM, const double &r)
{
    return lM.scale(1.0 / r);
}
//rvalue division
template <class D>
Matrix<D> operator/(Matrix<D>&& lM, const D& r)
{
    // cout << "rval /" << endl;
    int numelems = lM.num_row() * lM.num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        lM.elements[i] /= r;

    return std::move(lM);
}
template <class D>
Matrix<D> operator*(const Matrix<D> &lM, const D &r)
{
    return lM.scale(r);
}
template <class D>
Matrix<D> operator*(const D &l, const Matrix<D> &rM)
{
    return rM.scale(l);
}
//rvalue *
template <class D>
Matrix<D> operator*(const D &l, Matrix<D> &&rM)
{
    int numelems = rM.num_row() * rM.num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        rM.elements[i] *= l;

    return std::move(rM);

}
template <class D>
Matrix<D>& operator+=(Matrix<D> &lM, const D &r)
{
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    for (int j = 0; j < numcol; j++)
    {
#pragma ivdep
        for (int i = 0; i < numrow; i++)
        {
            lM.elem(i, j) += r;
        }
    }
    return lM;
}
template <class D>
Matrix<D>& operator-=(Matrix<D> &lM, const D &r)
{
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    for (int j = 0; j < numcol; j++)
    {
#pragma ivdep
        for (int i = 0; i < numrow; i++)
        {
            lM.elem(i, j) -= r;
        }
    }
    return lM;
}
template <class D>
Matrix<D>& operator+=(Matrix<D> &lM, const Matrix<D> &rM)
{
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    for (int j = 0; j < numcol; j++)
    {
#pragma ivdep
        for (int i = 0; i < numrow; i++)
        {
            lM.elem(i, j) += rM.elem(i, j);
        }
    }
    return lM;
}
template <class D>
Matrix<D>& operator-=(Matrix<D> &lM, const Matrix<D> &rM)
{
    int numcol = lM.num_col();
    int numrow = lM.num_row();
    for (int j = 0; j < numcol; j++)
    {
#pragma ivdep
        for (int i = 0; i < numrow; i++)
        {
            lM.elem(i, j) -= rM.elem(i, j);
        }
    }
    return lM;
}

#endif