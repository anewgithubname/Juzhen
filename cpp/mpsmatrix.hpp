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
    return (MPSfloat *)mpsMalloc(size * sizeof(MPSfloat));
}

template<>
inline void Memory<MPSfloat>::_free(MPSfloat* ptr) {
    mpsFree((float *) ptr);
}

template<>
class Matrix<MPSfloat> {
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
    
    Matrix<float> to_host() const;
    static Matrix<MPSfloat> randn(size_t m, size_t n);
    static Matrix<MPSfloat> zeros(size_t m, size_t n);
    static Matrix<MPSfloat> ones(size_t m, size_t n);

    // basic matrix ops
    Matrix<MPSfloat> dot(const Matrix<MPSfloat>& B) const;
    Matrix<MPSfloat> T() const;

    
};

Matrix<MPSfloat> sum(const Matrix<MPSfloat>& A);
#endif