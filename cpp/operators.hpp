#pragma once
/**
 * @file operator.hpp
 * @brief Operator Definitions
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2023 Song Liu (song.liu@bristol.ac.uk)

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

#include "core.hpp"

// operator overloads

#define HAS_CONCEPTS 0

#pragma region Definitions of Concepts
// The four rules for implementing a matrix class
// Addable, Scalable, Multiplicable, ElemInvitable

#if HAS_CONCEPTS == 1
// Needs C++ 20
#include <concepts>

/**
 * @brief A concept for matrices that can be multiplied
 * 
 * @tparam D the data type
 */
template <typename D>
concept Multiplicable = requires(const Matrix<D>& A, const Matrix<D>& B) {
    { A.dot(B) } -> std::same_as<Matrix<D>>;  // multiply
};

/**
 * @brief A concept for matrices that can be added
 * 
 * Matrix can be sel-added or added to produce another matrix
 * 
 * @tparam D the data type
 */
template <typename D>
concept Addable = requires(Matrix<D>&& A, const Matrix<D>& B, const Matrix<D>& C) {
    { B.add(C, 1, 1) } -> std::same_as<Matrix<D>>;  // const add
    A.add(B, 1, 1);                         // self add
};

/**
 * @brief A concept for matrices that can be added with a scalar
 * 
 * @tparam D the data type
 */
template <typename D>
concept ScalarAddable = requires(Matrix<D>&& A, const Matrix<D>& B) {
    A.add(1, 1);                         // self add
    { B.add(1, 1) } -> std::same_as<Matrix<D>>;  // const add
};

/**
 * @brief A concept for matrices that can be scaled
 * 
 * @tparam D the data type
 */
template <typename D>
concept Scalable = requires(Matrix<D>&& A, const Matrix<D>& B) {
    A.scale(1);                         // self scale
    { B.scale(1) } -> std::same_as<Matrix<D>>;  // const scale
};

/**
 * @brief A concept for matrices that can be inversed
 * 
 * @tparam D the data type
 */
template <typename D>
concept ElemInvitable = requires(const Matrix<D>&& A, const Matrix<D>& B) {
    A.eleminv(1);                         // self reciprocal
    { B.eleminv(1) } -> std::same_as<Matrix<D>>;  // const reciprocal
};
#else
#define Multiplicable typename
#define Addable typename
#define ScalarAddable typename
#define Scalable typename
#define ElemInvitable typename
#endif
#pragma endregion

// definitions of operators
// Multiplicable matrices
// .............................................
template <Multiplicable D>
Matrix<D> operator*(const Matrix<D>& lM, const Matrix<D>& rM) {
    return lM.dot(rM);
}

// Addable matrices
// .............................................
template <Addable D>
Matrix<D> operator+(const Matrix<D>& lM, const Matrix<D>& rM) {
    return lM.add(rM, 1.0, 1.0);
}
// rvalue version of operator +
template <Addable D>
Matrix<D> operator+(Matrix<D>&& lM, const Matrix<D>& rM) {
    lM.add(rM, 1.0, 1.0);
    return std::move(lM);
}
// rvalue version of operator +
template <Addable D>
Matrix<D> operator+(const Matrix<D>& lM, Matrix<D>&& rM) {
    rM.add(lM, 1.0, 1.0);
    return std::move(rM);
}
// rvalue version of operator +
template <Addable D>
Matrix<D> operator+(Matrix<D>&& lM, Matrix<D>&& rM) {
    lM.add(rM, 1.0, 1.0);
    return std::move(lM);
}
// const reference operator -
template <Addable D>
Matrix<D> operator-(const Matrix<D>& lM, const Matrix<D>& rM) {
    return lM.add(rM, 1.0, -1.0);
}
// rvalue version of operator -
template <Addable D>
Matrix<D> operator-(const Matrix<D>& lM, Matrix<D>&& rM) {
    rM.add(lM, -1.0, 1.0);
    return std::move(rM);
}
// rvalue version of operator -
template <Addable D>
Matrix<D> operator-(Matrix<D>&& lM, const Matrix<D>& rM) {
    lM.add(rM, 1.0, -1.0);
    return std::move(lM);
}
// rvalue version of operator -
template <Addable D>
Matrix<D> operator-(Matrix<D>&& lM, Matrix<D>&& rM) {
    lM.add(rM, 1.0, -1.0);
    return std::move(lM);
}
template <Addable D>
Matrix<D>& operator+=(Matrix<D>& lM, const Matrix<D>& rM) {
    lM.add(rM, 1.0, 1.0);
    return lM;
}

template <Addable D>
Matrix<D>& operator-=(Matrix<D>& lM, const Matrix<D>& rM) {
    lM.add(rM, 1.0, -1.0);
    return lM;
}

// ScalarAddable matrices
// .............................................
template <ScalarAddable D>
Matrix<D> operator+(const Matrix<D>& lM, double r) {
    return lM.add(r, 1.0);
}
// rvalue version of operator +
template <ScalarAddable D>
Matrix<D> operator+(Matrix<D>&& lM, double r) {
    lM.add(r, 1.0);
    return std::move(lM);
}
template <ScalarAddable D>
Matrix<D> operator+(double l, const Matrix<D>& rM) {
    return rM.add(l, 1.0);
}
template <ScalarAddable D>
Matrix<D> operator+(double l, Matrix<D>&& rM) {
    rM.add(l, 1.0);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D> operator-(const Matrix<D>& lM, double r) {
    return lM.add(-r, 1.0);
}
// rvalue version of operator -
template <ScalarAddable D>
Matrix<D> operator-(Matrix<D>&& lM, double r) {
    lM.add(-r, 1.0);
    return std::move(lM);
}
template <ScalarAddable D>
Matrix<D> operator-(double l, const Matrix<D>& rM) {
    return rM.add(l, -1.0);
}
template <ScalarAddable D>
Matrix<D> operator-(double l, Matrix<D>&& rM) {
    rM.add(l, -1.0);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D> operator-(const Matrix<D>& rM) {
    return rM.add(0, -1.0);
}
template <ScalarAddable D>
Matrix<D> operator-(Matrix<D>&& rM) {
    rM.add(0, -1.0);
    return std::move(rM);
}
template <ScalarAddable D>
Matrix<D>& operator+=(Matrix<D>& lM, double r) {
    lM.add(r, 1);
    return lM;
}
template <ScalarAddable D>
Matrix<D>& operator-=(Matrix<D>& lM, double r) {
    lM.add(-r, 1);
    return lM;
}

// Scalable matrices
// .............................................

template <Scalable D>
Matrix<D> operator/(const Matrix<D>& lM, double r) {
    return lM.scale((1.0 / r));
}
// rvalue division
template <Scalable D>
Matrix<D> operator/(Matrix<D>&& lM, double r) {
    // cout << "rval /" << endl;
    lM.scale((1 / r));
    return std::move(lM);
}
template <Scalable D>
Matrix<D> operator*(const Matrix<D>& lM, double r) {
    return lM.scale(r);
}
template <Scalable D>
Matrix<D> operator*(Matrix<D>&& lM, double r) {
    lM.scale(r);
    return std::move(lM);
}
template <Scalable D>
Matrix<D> operator*(double l, const Matrix<D>& rM) {
    return rM.scale(l);
}
// rvalue *
template <Scalable D>
Matrix<D> operator*(double l, Matrix<D>&& rM) {
    rM.scale(l);
    return std::move(rM);
}

// Elementwise Invertible Matrices
// .............................................

template <ElemInvitable D>
Matrix<D> operator/(const Matrix<D>& lM, const Matrix<D>& rM) {
    Matrix<D> ret = rM.eleminv(1.0);
    return hadmd(lM, ret);
}
// rvalue division
template <ElemInvitable D>
Matrix<D> operator/(const Matrix<D>& lM, Matrix<D>&& rM) {
    rM.eleminv(1.0);
    return hadmd(lM, std::move(rM));
}
template <ElemInvitable D>
Matrix<D> operator/(double l, const Matrix<D>& rM) {
    return rM.eleminv(l);
}
// rvalue division
template <ElemInvitable D>
Matrix<D> operator/(double l, Matrix<D>&& rM) {
    rM.eleminv(l);
    return std::move(rM);
}

// stream operator
template <class D>
std::ostream& operator<<(std::ostream& os, const Matrix<D>& M) {
    using namespace std;
    // write obj to stream
    os << M.get_name() << " " << M.num_row() << " by " << M.num_col();
    for (size_t i = 0; i < M.num_row(); i++) {
        os << endl;
        for (size_t j = 0; j < M.num_col(); j++) {
            os << M.elem(i, j) << " ";
        }
    }
    return os;
}