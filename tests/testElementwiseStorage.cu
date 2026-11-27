/**
 * + - .* ./ between CPU matrices stored in either order (with and without
 * the transpose flag), against values computed element by element.
 *
 * Matrix<float>::add and hadmd run over the storage directly. This checks
 * every pairing of storage orders, the versions that write into an operand,
 * and an operand that shares the storage being written (A += A.T()).
 */
#include "../cpp/juzhen.hpp"
#include <cmath>
#include <iostream>

static float left(size_t i, size_t j) { return 0.25f * float(i) - 0.5f * float(j) + 1.0f; }
static float right(size_t i, size_t j) { return float((3 * i + 7 * j) % 11) + 2.0f; }  // never zero

// rows x cols with the given values, stored column-major (no flag) or row-major (the transpose flag)
template <class Value>
static Matrix<float> filled(size_t rows, size_t cols, bool flagged, Value value) {
    if (!flagged) {
        Matrix<float> m("plain", rows, cols);
        for (size_t j = 0; j < cols; ++j)
            for (size_t i = 0; i < rows; ++i) m.elem(i, j) = value(i, j);
        return m;
    }
    Matrix<float> t("transposed", cols, rows);
    for (size_t j = 0; j < cols; ++j)
        for (size_t i = 0; i < rows; ++i) t.elem(j, i) = value(i, j);
    return t.T();  // a copy that keeps the flag
}

static int failures = 0;

template <class Value>
static void expect(const char *what, const Matrix<float> &got, size_t rows, size_t cols, Value value,
                   float tolerance = 0.0f) {
    bool ok = got.num_row() == rows && got.num_col() == cols;
    for (size_t j = 0; ok && j < cols; ++j)
        for (size_t i = 0; ok && i < rows; ++i) {
            const float want = value(i, j);
            ok = std::fabs(got.elem(i, j) - want) <= tolerance * (1.0f + std::fabs(want));
        }
    if (!ok) {
        std::cout << "FAIL " << what << " (" << rows << " x " << cols << ")\n";
        ++failures;
    }
}

static void check(const char *what, bool ok) {
    if (!ok) {
        std::cout << "FAIL " << what << "\n";
        ++failures;
    }
}

int compute() {
    const size_t shapes[][2] = {{1, 1}, {1, 9}, {7, 1}, {3, 5}, {40, 33}, {70, 70}, {129, 65}};
    for (const auto &shape : shapes) {
        const size_t rows = shape[0], cols = shape[1];
        for (int combo = 0; combo < 4; ++combo) {
            const bool fa = combo & 1, fb = combo & 2;
            const Matrix<float> A = filled(rows, cols, fa, left), B = filled(rows, cols, fb, right);
            auto sum = [](size_t i, size_t j) { return left(i, j) + right(i, j); };
            auto difference = [](size_t i, size_t j) { return left(i, j) - right(i, j); };
            auto product = [](size_t i, size_t j) { return left(i, j) * right(i, j); };
            auto quotient = [](size_t i, size_t j) { return left(i, j) / right(i, j); };
            auto scaled = [](size_t i, size_t j) { return 0.5f * left(i, j) - 2.0f * right(i, j); };

            // new matrices
            expect("A + B", A + B, rows, cols, sum);
            expect("A - B", A - B, rows, cols, difference);
            expect("hadmd(A, B)", hadmd(A, B), rows, cols, product);
            expect("A / B", A / B, rows, cols, quotient, 1e-6f);
            expect("A.add(B, 0.5, -2)", A.add(B, 0.5f, -2.0f), rows, cols, scaled, 1e-6f);
            // stored like the operands when they agree, plainly otherwise
            check("storage of A + B", (A + B).get_transpose() == (fa && fb));
            check("storage of hadmd(A, B)", hadmd(A, B).get_transpose() == (fa && fb));

            // written into one of the operands
            Matrix<float> C = A;
            C += B;
            expect("C += B", C, rows, cols, sum);
            check("storage of C += B", C.get_transpose() == fa);
            C = A;
            C -= B;
            expect("C -= B", C, rows, cols, difference);
            C = A;
            C.add(B, 0.5f, -2.0f);
            expect("C.add(B, 0.5, -2)", C, rows, cols, scaled, 1e-6f);
            expect("temporary + B", Matrix<float>(A) + B, rows, cols, sum);
            expect("A + temporary", A + Matrix<float>(B), rows, cols, sum);
            expect("temporary - B", Matrix<float>(A) - B, rows, cols, difference);
            expect("A - temporary", A - Matrix<float>(B), rows, cols, difference);
            expect("hadmd(temporary, B)", hadmd(Matrix<float>(A), B), rows, cols, product);
            expect("hadmd(A, temporary)", hadmd(A, Matrix<float>(B)), rows, cols, product);
            expect("A / temporary", A / Matrix<float>(B), rows, cols, quotient, 1e-6f);
            expect("A is unchanged", A, rows, cols, left);
            expect("B is unchanged", B, rows, cols, right);
        }
    }

    // The right side shares the storage that is being written.
    for (size_t n : {size_t(1), size_t(3), size_t(70)}) {
        for (bool flagged : {false, true}) {
            auto symmetric = [](size_t i, size_t j) { return left(i, j) + left(j, i); };
            auto skew = [](size_t i, size_t j) { return left(i, j) - left(j, i); };
            auto squares = [](size_t i, size_t j) { return left(i, j) * left(j, i); };
            Matrix<float> A = filled(n, n, flagged, left);
            A += A.T();
            expect("A += A.T()", A, n, n, symmetric);
            A = filled(n, n, flagged, left);
            A -= A.T();
            expect("A -= A.T()", A, n, n, skew);
            A = filled(n, n, flagged, left);
            A += A;
            expect("A += A", A, n, n, [](size_t i, size_t j) { return 2 * left(i, j); });
            A = filled(n, n, flagged, left);
            const Matrix<float> view = A.T();
            expect("hadmd(A, A.T()) into A", hadmd(std::move(A), view), n, n, squares);
        }
    }

    std::cout << (failures ? "FAIL" : "PASS") << " elementwise storage orders\n";
    return failures ? 1 : 0;
}
