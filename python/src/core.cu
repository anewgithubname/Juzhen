/**
 * juzhen._core: the C++ half of the Python package in python/juzhen.
 *
 * Each build has two submodules, `cpu` (Matrix<float>) and, when Juzhen is
 * built with a GPU backend, `gpu` (Matrix<CUDAfloat>, Matrix<ROCMfloat> or
 * Matrix<MPSfloat>). Both are made by the same template, bind<D>().
 *
 * Layout. Python matrices are "one sample per row" (N x d), like numpy.
 * A C-order N x d numpy array has the same bytes as a column-major d x N
 * Juzhen matrix, so an imported array is that d x N matrix with the lazy
 * transpose flag set: no data is moved. Layers work on d x N as in C++;
 * x.T() hands them the native layout and their d x N outputs come back to
 * Python transposed the same way.
 *
 * Memory. Matrices are allocated from Juzhen's own pools and data crosses
 * into or out of Python only by copy. The pools live until the process ends.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <streambuf>

#include "../../cpp/juzhen.hpp"
#include "../../ml/checkpoint.hpp"
#include "../../ml/embedding.hpp"
#include "../../ml/layer.hpp"
#include "../../ml/loss.hpp"
#include "../../ml/optimizer.hpp"

#ifdef ROCM_HIP
#include <hip/hip_runtime.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace Juzhen;

#if defined(CUDA)
using GPU = CUDAfloat;
constexpr const char *kBackend = "cuda";
#elif defined(ROCM_HIP)
using GPU = ROCMfloat;
constexpr const char *kBackend = "rocm";
#elif defined(APPLE_SILICON)
using GPU = MPSfloat;
constexpr const char *kBackend = "metal";
#else
constexpr const char *kBackend = "cpu";
#endif
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
#define JZ_HAS_GPU 1
#endif

// The convolution layers exist for one type per build: the GPU type when
// there is a GPU backend (CUDA only with cuDNN), float otherwise.
#if defined(CUDA) && defined(CUDNN_AVAILABLE)
#define JZ_HAS_CONV 1
using ConvType = CUDAfloat;
#elif defined(ROCM_HIP) || defined(APPLE_SILICON)
#define JZ_HAS_CONV 1
using ConvType = GPU;
#elif !defined(CUDA)
#define JZ_HAS_CONV 1
using ConvType = float;
#endif

// A named namespace: nvcc does not allow extended lambdas in functions with
// internal linkage.
namespace jzpy {

template <class D>
constexpr const char *device_name() {
    return std::is_same_v<D, float> ? "cpu" : "gpu";
}

// ---------------------------------------------------------------------------
// Layout helpers

// The same storage with the transpose flag flipped; no data is copied.
template <class D>
Matrix<D> transposed(const Matrix<D> &m) {
    return m.T();
}

// A deep copy, for results that must not share storage with layer state.
template <class D>
Matrix<D> copy_of(const Matrix<D> &m) {
    return Matrix<D>(m);
}

// A deep copy stored without the transpose flag. Layers and some kernels
// read their inputs and parameters as plain column-major arrays.
template <class D>
Matrix<D> plain_copy(const Matrix<D> &m) {
    if (!m.get_transpose())
        return copy_of(m);
    Matrix<D> out = Matrix<D>::zeros(m.num_row(), m.num_col());
    out += m;
    return out;
}

// m itself when it has no transpose flag, otherwise a plain copy.
template <class D>
Matrix<D> plain(const Matrix<D> &m) {
    return m.get_transpose() ? plain_copy(m) : transposed(transposed(m));
}

// The matrix on the host (shares storage for a CPU matrix).
template <class D>
Matrix<float> host(const Matrix<D> &m) {
    if constexpr (std::is_same_v<D, float>)
        return transposed(transposed(m));
    else
        return m.to_host();
}

// A host matrix moved to device D.
template <class D>
Matrix<D> to_device(Matrix<float> &&m) {
    if constexpr (std::is_same_v<D, float>)
        return std::move(m);
    else
        return Matrix<D>(m);
}

std::string shape_str(size_t rows, size_t cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

template <class D>
std::string shape_str(const Matrix<D> &m) {
    return shape_str(m.num_row(), m.num_col());
}

void sized(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("juzhen matrices cannot be empty, got shape " + shape_str(rows, cols));
}

// ---------------------------------------------------------------------------
// numpy

using InArray = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, float, nb::ndim<2>>;

// Token ids and class numbers arrive as a flat int32 array that nn.py has
// already checked and converted; noconvert() below keeps nanobind from
// casting anything else silently. They are copied into a host Matrix<int>
// (Juzhen's own pool): `rows` x (size / rows), read column after column, so a
// C-order (batch, seq_len) numpy array is the seq_len x batch matrix the
// layers call "one sequence per column".
using IdArray = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

Matrix<int> ids_matrix(IdArray a, size_t rows = 1) {
    const size_t n = a.shape(0);
    if (n == 0)
        throw std::invalid_argument("no ids were given");
    if (rows == 0 || n % rows)
        throw std::invalid_argument("the ids do not fill whole sequences");
    Matrix<int> m("ids", rows, n / rows);
    std::memcpy(const_cast<int *>(m.data()), a.data(), n * sizeof(int32_t));
    return m;
}

// "auto" (the layer's own rule) or a standard deviation (rule "normal")
Init make_init(const std::string &rule, float std) {
    if (rule == "auto") return Init();
    if (rule == "normal") return Init::normal(std);
    throw std::invalid_argument("unknown init rule '" + rule + "'");
}

// A C-order (rows, cols) array, copied into a new matrix on device D.
template <class D>
Matrix<D> from_numpy(InArray a) {
    const size_t rows = a.shape(0), cols = a.shape(1);
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("juzhen matrices cannot be empty, got shape " + shape_str(rows, cols));
    // column-major cols x rows has the bytes of a C-order rows x cols array
    Matrix<float> m("array", cols, rows);
    std::memcpy(const_cast<float *>(m.data()), a.data(), rows * cols * sizeof(float));
    return to_device<D>(transposed(m));
}

// A new numpy array holding a copy of the matrix.
template <class D>
OutArray to_numpy(const Matrix<D> &m) {
    Matrix<float> h = host(m);
    const size_t rows = h.num_row(), cols = h.num_col();
    float *buffer = new float[rows * cols];
    std::memcpy(buffer, h.data(), rows * cols * sizeof(float));
    nb::capsule owner(buffer, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    // the storage is C-order when the transpose flag is set, Fortran-order otherwise
    if (h.get_transpose())
        return OutArray(buffer, {rows, cols}, owner, {(int64_t)cols, 1});
    return OutArray(buffer, {rows, cols}, owner, {1, (int64_t)rows});
}

template <class D>
float item(const Matrix<D> &m) {
    if (m.num_row() != 1 || m.num_col() != 1)
        throw std::invalid_argument("item() needs a 1 x 1 matrix, got shape " + shape_str(m));
    return host(m).elem(0, 0);
}

template <class D>
std::string repr(const Matrix<D> &m) {
    nb::object np = nb::module_::import_("numpy");
    nb::object text = np.attr("array2string")(nb::cast(to_numpy(m)), "separator"_a = ", ",
                                              "prefix"_a = "Matrix(", "threshold"_a = 200);
    return "Matrix(" + nb::cast<std::string>(text) + ", device='" + device_name<D>() + "')";
}

// ---------------------------------------------------------------------------
// Arithmetic with numpy-style broadcasting of (1, 1), (1, cols) and (rows, 1)

// b repeated to rows x cols (shares storage when it already has that shape)
template <class D>
Matrix<D> broadcast(const Matrix<D> &b, size_t rows, size_t cols) {
    const size_t br = b.num_row(), bc = b.num_col();
    if (br == rows && bc == cols)
        return transposed(transposed(b));
    Matrix<D> out = transposed(transposed(b));
    if (br != rows)
        out = Matrix<D>::ones(rows, 1) * out;  // (rows x 1)(1 x bc)
    if (bc != cols)
        out = out * Matrix<D>::ones(1, cols);  // (rows x 1)(1 x cols)
    return out;
}

enum class Op { add, sub, mul, div };

const char *op_symbol(Op op) {
    switch (op) {
        case Op::add: return "+";
        case Op::sub: return "-";
        case Op::mul: return "*";
        default: return "/";
    }
}

// can a dimension of size n be broadcast to size target?
bool fits(size_t n, size_t target) { return n == target || n == 1; }

template <class D>
Matrix<D> binary(const Matrix<D> &a, const Matrix<D> &b, Op op) {
    const size_t rows = std::max(a.num_row(), b.num_row());
    const size_t cols = std::max(a.num_col(), b.num_col());
    if (!fits(a.num_row(), rows) || !fits(b.num_row(), rows) || !fits(a.num_col(), cols) ||
        !fits(b.num_col(), cols))
        throw std::invalid_argument("shapes " + shape_str(a) + " and " + shape_str(b) +
                                    " cannot be broadcast together for " + op_symbol(op));
    const Matrix<D> x = broadcast(a, rows, cols), y = broadcast(b, rows, cols);
    switch (op) {
        case Op::add: return x + y;
        case Op::sub: return x - y;
        case Op::mul: return hadmd(x, y);
        default: return x / y;
    }
}

// a += b or a -= b in place; b may be broadcast to a's shape
template <class D>
void inplace(Matrix<D> &a, const Matrix<D> &b, Op op) {
    if (!fits(b.num_row(), a.num_row()) || !fits(b.num_col(), a.num_col()))
        throw std::invalid_argument("cannot broadcast " + shape_str(b) + " into " + shape_str(a) +
                                    " for " + op_symbol(op) + "=");
    Matrix<D> y = broadcast(b, a.num_row(), a.num_col());
    // A += A.T: the right side is the storage being written; use a copy of it
    if ((const void *)y.data() == (const void *)a.data() && y.get_transpose() != a.get_transpose())
        y = copy_of(y);
    if (op == Op::add)
        a += y;
    else
        a -= y;
}

template <class D>
Matrix<D> matmul(const Matrix<D> &a, const Matrix<D> &b) {
    if (a.num_col() != b.num_row())
        throw std::invalid_argument("matmul: shapes " + shape_str(a) + " and " + shape_str(b) +
                                    " are not aligned (" + std::to_string(a.num_col()) +
                                    " != " + std::to_string(b.num_row()) + ")");
    return a * b;
}

// ---------------------------------------------------------------------------
// Elementwise comparisons. The result is a matrix of 1s and 0s (floats, like
// every juzhen matrix), so it can be used in arithmetic: (A > 0) * A.

enum class Cmp { eq, ne, lt, le, gt, ge };

// x compared with the number s, element by element
template <class D>
Matrix<D> compare(const Matrix<D> &a, float s, Cmp op) {
    const int o = (int)op;
    return elemwise([=] __GPU_CPU__(float x) {
        const bool r = o == 0 ? x == s : o == 1 ? x != s : o == 2 ? x < s : o == 3 ? x <= s : o == 4 ? x > s : x >= s;
        return r ? 1.0f : 0.0f;
    }, a);
}

// Two matrices (broadcast like the arithmetic) are compared through their
// difference, which every device computes. One consequence: where both hold
// the same infinity the difference is nan, and they compare as different.
template <class D>
Matrix<D> compare(const Matrix<D> &a, const Matrix<D> &b, Cmp op) {
    const size_t rows = std::max(a.num_row(), b.num_row());
    const size_t cols = std::max(a.num_col(), b.num_col());
    static const char *symbols[] = {"==", "!=", "<", "<=", ">", ">="};
    if (!fits(a.num_row(), rows) || !fits(b.num_row(), rows) || !fits(a.num_col(), cols) ||
        !fits(b.num_col(), cols))
        throw std::invalid_argument("shapes " + shape_str(a) + " and " + shape_str(b) +
                                    " cannot be broadcast together for " + symbols[(int)op]);
    const int o = (int)op;
    return elemwise([=] __GPU_CPU__(float d) {
        const bool r = o == 0 ? d == 0 : o == 1 ? d != 0 : o == 2 ? d < 0 : o == 3 ? d <= 0 : o == 4 ? d > 0 : d >= 0;
        return r ? 1.0f : 0.0f;
    }, binary(a, b, Op::sub));
}

// if A == B: a matrix is true or false only when it holds one value
template <class D>
bool truth(const Matrix<D> &m) {
    if (m.num_row() != 1 || m.num_col() != 1)
        throw std::invalid_argument("a matrix of shape " + shape_str(m) + " is not one True or False. A comparison "
                                    "such as A == B gives a matrix of 1s and 0s; to ask whether all (or any) of "
                                    "them hold use (A == B).numpy().all() or .any()");
    return host(m).elem(0, 0) != 0;
}

// ---------------------------------------------------------------------------
// Elementwise functions and reductions

template <class D>
Matrix<D> sin_of(const Matrix<D> &m) {
    if constexpr (std::is_same_v<D, float>)
        return sin(m);
    else
        return elemwise([=] __GPU_CPU__(float x) { return sinf(x); }, m);
}

template <class D>
Matrix<D> cos_of(const Matrix<D> &m) {
    if constexpr (std::is_same_v<D, float>)
        return cos(m);
    else
        return elemwise([=] __GPU_CPU__(float x) { return cosf(x); }, m);
}

template <class D>
Matrix<D> relu_of(const Matrix<D> &m) {
    return relu(copy_of(m));
}

// sum along axis 0 (down the rows, giving 1 x cols) or 1 (giving rows x 1)
template <class D>
Matrix<D> sum_axis(const Matrix<D> &m, int axis) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1, got " + std::to_string(axis));
    return sum(m, axis);
}

// ---------------------------------------------------------------------------
// Indexing: m[rows], m[rows, cols] with ints, slices or lists of ints

struct Index {
    bool contiguous = true;
    size_t start = 0, stop = 0;  // when contiguous
    idxlist list;                // otherwise
    bool scalar = false;         // a single int

    idxlist as_list() const { return contiguous ? seq(start, stop) : list; }
    size_t size() const { return contiguous ? stop - start : list.size(); }
};

// what can index a matrix, and what was given instead
std::string bad_index(nb::handle item) {
    const std::string type = Py_TYPE(item.ptr())->tp_name;
    std::string got = "a " + type;
    if (item.is_none() || type == "ellipsis")
        got = std::string(item.is_none() ? "None" : "...") +
              " (a juzhen matrix always has two dimensions; write A[:, j] for a column)";
    else if (PyFloat_Check(item.ptr()) || type.rfind("numpy.float", 0) == 0)
        got = "a float (use int(i), or idx.astype(int) for an array)";
    else if (type == "numpy.ndarray")
        got = "an array with more than one dimension (use idx.ravel())";
    return "a matrix is indexed with whole numbers, slices or lists of whole numbers, rows first: A[i], "
           "A[i, j], A[1:3], A[[0, 2]]; got " + got;
}

// an int (or numpy integer) index into a dimension of size n; what is "row" or "column"
size_t position(nb::handle item, size_t n, const char *what) {
    // True and False would pass as 1 and 0: X[y == 2] and X[[True, False]] must not
    if (PyBool_Check(item.ptr()) || std::string(Py_TYPE(item.ptr())->tp_name) == "numpy.bool" ||
        std::string(Py_TYPE(item.ptr())->tp_name) == "numpy.bool_")
        throw nb::type_error("a matrix cannot be indexed with True or False (masks are not supported); use "
                             "row numbers, for example X[np.flatnonzero(labels == 2)] with numpy labels");
    nb::object index = nb::steal(PyNumber_Index(item.ptr()));
    if (!index.is_valid()) {
        PyErr_Clear();
        throw nb::type_error(bad_index(item).c_str());
    }
    const Py_ssize_t given = PyLong_AsSsize_t(index.ptr());
    if (given == -1 && PyErr_Occurred()) throw nb::python_error();
    const Py_ssize_t i = given < 0 ? given + (Py_ssize_t)n : given;
    if (i < 0 || i >= (Py_ssize_t)n)
        throw nb::index_error((std::string(what) + " index " + std::to_string(given) + " is out of range for a "
                               "matrix with " + std::to_string(n) + " " + what + (n == 1 ? "" : "s")).c_str());
    return (size_t)i;
}

Index parse_index(nb::handle key, size_t n, const char *what) {
    Index index;
    if (nb::isinstance<nb::slice>(key)) {
        auto [start, stop, step, length] = nb::borrow<nb::slice>(key).compute(n);
        (void)stop;
        if (step == 1) {
            index.start = start;
            index.stop = start + length;
        } else {
            index.contiguous = false;
            for (size_t k = 0; k < length; ++k)
                index.list.push_back((size_t)(start + (Py_ssize_t)k * step));
        }
    } else if (PySequence_Check(key.ptr())) {  // a list or an array of ints
        const std::string type = Py_TYPE(key.ptr())->tp_name;
        if (type == "juzhen._core.cpu.Matrix" || type == "juzhen._core.gpu.Matrix")
            throw nb::type_error("a matrix cannot be indexed with a matrix (masks are not supported); use row "
                                 "numbers, for example X[np.flatnonzero((A > 0).numpy())] for an n x 1 mask");
        index.contiguous = false;
        for (nb::handle item : key)
            index.list.push_back(position(item, n, what));
    } else {
        index.start = position(key, n, what);
        index.stop = index.start + 1;
        index.scalar = true;
    }
    if (index.size() == 0)
        throw nb::index_error("indexing gives an empty matrix; juzhen matrices cannot be empty");
    return index;
}

// A contiguous block, keeping the storage order of m (row slices of a
// sample-first matrix stay C-order).
template <class D>
Matrix<D> take_block(const Matrix<D> &m, size_t r0, size_t r1, size_t c0, size_t c1) {
    if (m.get_transpose()) {
        const Matrix<D> physical = transposed(m);
        return transposed(physical.slice(c0, c1, r0, r1));
    }
    return m.slice(r0, r1, c0, c1);
}

template <class D>
nb::object get_item(const Matrix<D> &m, nb::handle key) {
    nb::handle row_key = key, col_key;
    nb::object all = nb::steal(PySlice_New(nullptr, nullptr, nullptr));
    if (nb::isinstance<nb::tuple>(key)) {
        nb::tuple t = nb::borrow<nb::tuple>(key);
        if (t.size() != 2)
            throw nb::index_error("juzhen matrices take one or two indices");
        row_key = t[0];
        col_key = t[1];
    } else {
        col_key = all;
    }
    const Index r = parse_index(row_key, m.num_row(), "row");
    const Index c = parse_index(col_key, m.num_col(), "column");

    if (r.scalar && c.scalar) {
        // one element; reading it waits for the GPU
        if constexpr (std::is_same_v<D, float>)
            return nb::float_(m.elem(r.start, c.start));
        else
            return nb::float_(host(take_block(m, r.start, r.stop, c.start, c.stop)).elem(0, 0));
    }
    if (r.contiguous && c.contiguous)
        return nb::cast(take_block(m, r.start, r.stop, c.start, c.stop));
    // Lists of indices: gathered on the matrix's own device. Like
    // take_block, the result keeps the storage order of m, so chosen rows
    // of a sample-first matrix are copied row by row.
    if (m.get_transpose())
        return nb::cast(transposed(transposed(m).slice(c.as_list(), r.as_list())));
    return nb::cast(m.slice(r.as_list(), c.as_list()));
}

// ---------------------------------------------------------------------------
// More functions: reductions along an axis, softmax, top-k, one-hot, stacking.
// Axis 0 works down each column (one result per column, 1 x cols); axis 1
// along each row (rows x 1). The reductions run on the device on CUDA and
// ROCm; on Metal reduce() and elemwise() copy through the host, except max
// and min, which read the buffer where it is (see metal_extreme).

void check_axis(int axis) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1, got " + std::to_string(axis));
}

// The largest (or smallest) of n numbers. The comparison is the one of the
// reduce() lambdas below, done in 16 lanes so that the compiler can use SIMD;
// every lane starts from v[0], so a NaN is treated as it is there.
template <class Better>
float extreme_of(const float *v, size_t n, Better better) {
    constexpr size_t lanes = 16;
    float best[lanes];
    for (size_t k = 0; k < lanes; ++k) best[k] = v[0];
    size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        for (size_t k = 0; k < lanes; ++k) best[k] = better(v[i + k], best[k]) ? v[i + k] : best[k];
    float result = best[0];
    for (size_t k = 1; k < lanes; ++k) result = better(best[k], result) ? best[k] : result;
    for (; i < n; ++i) result = better(v[i], result) ? v[i] : result;
    return result;
}

// p holds `count` runs of `length` numbers, one after another. out gets the
// extreme of every run (per_run), or of every position across the runs.
template <class Better>
void extreme_of_runs(const float *p, size_t length, size_t count, bool per_run, float *__restrict out, Better better) {
    if (per_run) {
        for (size_t j = 0; j < count; ++j) out[j] = extreme_of(p + j * length, length, better);
        return;
    }
    std::copy(p, p + length, out);
    for (size_t j = 1; j < count; ++j) {
        const float *run = p + j * length;
        for (size_t i = 0; i < length; ++i) out[i] = better(run[i], out[i]) ? run[i] : out[i];
    }
}

// max or min along an axis on the CPU, read straight from the storage.
// reduce() hands whole stored columns to its function, so for the other
// direction it first makes a transposed copy of the matrix; here that
// direction folds the stored columns into the result one after another.
template <class Better>
Matrix<float> cpu_extreme(const Matrix<float> &m, int axis, const char *name, Better better) {
    const bool flagged = m.get_transpose();
    Matrix<float> result = axis == 0 ? Matrix<float>(name, 1, m.num_col()) : Matrix<float>(name, m.num_row(), 1);
    // the runs are the stored columns: columns of the matrix unless it is flagged
    extreme_of_runs(m.data(), flagged ? m.num_col() : m.num_row(), flagged ? m.num_row() : m.num_col(),
                    (axis == 0) != flagged, const_cast<float *>(result.data()), better);
    return result;
}

#ifdef APPLE_SILICON
// The same on Metal. A Metal buffer is memory the CPU can read once the
// pending GPU work has finished, so nothing is copied to the host: reduce()
// would copy and transpose the whole matrix first (11 ms for 4096 x 4096 on
// an M2, before the reduction itself). The buffer holds the matrix row by
// row, so the runs are rows of the matrix unless it is flagged.
template <class Better>
Matrix<MPSfloat> metal_extreme(const Matrix<MPSfloat> &m, int axis, const char *name, Better better) {
    const bool flagged = m.get_transpose();
    Matrix<float> result = axis == 0 ? Matrix<float>(name, 1, m.num_col()) : Matrix<float>(name, m.num_row(), 1);
    mpsSynchronize();
    extreme_of_runs(reinterpret_cast<const float *>(m.data()), flagged ? m.num_row() : m.num_col(),
                    flagged ? m.num_col() : m.num_row(), (axis == 1) != flagged,
                    const_cast<float *>(result.data()), better);
    return Matrix<MPSfloat>(result);
}
#endif

template <class D>
Matrix<D> max_axis(const Matrix<D> &m, int axis) {
    check_axis(axis);
    if constexpr (std::is_same_v<D, float>) {
        if (m.num_row() > 0 && m.num_col() > 0)
            return cpu_extreme(m, axis, "max", [](float a, float b) { return a > b; });
    }
#ifdef APPLE_SILICON
    if constexpr (std::is_same_v<D, MPSfloat>) {
        if (m.num_row() > 0 && m.num_col() > 0)
            return metal_extreme(m, axis, "max", [](float a, float b) { return a > b; });
    }
#endif
    return reduce([] __GPU_CPU__(float *v, float *out, int n, int) {
        float best = v[0];
        for (int i = 1; i < n; ++i) best = v[i] > best ? v[i] : best;
        out[0] = best;
    }, m, axis, 1);
}

template <class D>
Matrix<D> min_axis(const Matrix<D> &m, int axis) {
    check_axis(axis);
    if constexpr (std::is_same_v<D, float>) {
        if (m.num_row() > 0 && m.num_col() > 0)
            return cpu_extreme(m, axis, "min", [](float a, float b) { return a < b; });
    }
#ifdef APPLE_SILICON
    if constexpr (std::is_same_v<D, MPSfloat>) {
        if (m.num_row() > 0 && m.num_col() > 0)
            return metal_extreme(m, axis, "min", [](float a, float b) { return a < b; });
    }
#endif
    return reduce([] __GPU_CPU__(float *v, float *out, int n, int) {
        float best = v[0];
        for (int i = 1; i < n; ++i) best = v[i] < best ? v[i] : best;
        out[0] = best;
    }, m, axis, 1);
}

// the position of the largest value (the first one on ties), as a float
template <class D>
Matrix<D> argmax_axis(const Matrix<D> &m, int axis) {
    check_axis(axis);
    return reduce([] __GPU_CPU__(float *v, float *out, int n, int) {
        int best = 0;
        for (int i = 1; i < n; ++i) best = v[i] > v[best] ? i : best;
        out[0] = (float)best;
    }, m, axis, 1);
}

template <class D>
Matrix<D> abs_of(const Matrix<D> &m) {
    return elemwise([] __GPU_CPU__(float x) { return fabsf(x); }, m);
}

// The k largest (or smallest) values along an axis, in order, and their
// positions (as floats). NaNs are skipped.
template <class D>
std::tuple<Matrix<D>, Matrix<D>> topk_axis(const Matrix<D> &m, int k, int axis, bool largest) {
    check_axis(axis);
    const size_t n = axis == 0 ? m.num_row() : m.num_col();
    if (k < 1 || (size_t)k > n)
        throw std::invalid_argument("k must be between 1 and " + std::to_string(n) + ", got " +
                                    std::to_string(k));
    // the selection overwrites what it has taken, so it runs on a copy; with
    // the sign flipped, the smallest values are the largest
    const float sign = largest ? 1.0f : -1.0f;
    Matrix<D> work = m * sign;
    // each group gets 2k results: the values, then the positions
    Matrix<D> both = reduce([=] __GPU_CPU__(float *v, float *out, int len, int) {
        for (int i = 0; i < k; ++i) {
            int best = -1;
            float best_value = 0;  // kept in a register, not reread from v
            for (int j = 0; j < len; ++j) {
                const float x = v[j];
                if (x == x && (best < 0 || x > best_value)) {
                    best = j;
                    best_value = x;
                }
            }
            out[i] = best < 0 ? NAN : sign * best_value;
            out[k + i] = (float)best;
            if (best >= 0) v[best] = NAN;
        }
    }, work, axis, 2 * k);
    if (axis == 0)
        return {take_block(both, 0, k, 0, both.num_col()), take_block(both, k, 2 * k, 0, both.num_col())};
    return {take_block(both, 0, both.num_row(), 0, k), take_block(both, 0, both.num_row(), k, 2 * k)};
}

// m arranged so that the groups of an axis are its columns (no copy)
template <class D>
Matrix<D> groups_as_columns(const Matrix<D> &m, int axis) {
    check_axis(axis);
    return axis == 0 ? transposed(transposed(m)) : transposed(m);
}

// softmax of each group, computed as in ml/loss.hpp: the largest value is
// subtracted before exp() so it cannot overflow
template <class D>
Matrix<D> softmax_axis(const Matrix<D> &m, int axis) {
    const Matrix<D> c = groups_as_columns(m, axis);
    const auto ones = Matrix<D>::ones(c.num_row(), 1);
    auto maximum = loss_detail::column_max(c);
    auto exponents = exp(c - ones * maximum);
    auto totals = sum(exponents, 0);
    Matrix<D> p = exponents / (ones * std::move(totals));
    return axis == 0 ? std::move(p) : transposed(p);
}

template <class D>
Matrix<D> logsumexp_axis(const Matrix<D> &m, int axis) {
    const Matrix<D> c = groups_as_columns(m, axis);
    const auto ones = Matrix<D>::ones(c.num_row(), 1);
    auto maximum = loss_detail::column_max(c);
    auto totals = sum(exp(c - ones * maximum), 0);
    Matrix<D> result = log(Matrix<D>(totals)) + maximum;  // 1 x groups
    return axis == 0 ? std::move(result) : transposed(result);
}

// One-hot rows for the integer labels in an N x 1 matrix: N x classes,
// built on the device (only the labels cross from the host).
template <class D>
Matrix<D> one_hot_of(const Matrix<D> &labels, size_t classes) {
    if (labels.num_col() != 1)
        throw std::invalid_argument("one_hot: labels must have shape (N, 1), got " + shape_str(labels));
    const size_t n = labels.num_row();
    Matrix<float> arange("arange", classes, 1);
    for (size_t i = 0; i < classes; ++i) arange.elem(i, 0) = (float)i;
    // classes x N, entry (c, j) = c - label_j: zero where label_j is c
    Matrix<D> diff = to_device<D>(std::move(arange)) * Matrix<D>::ones(1, n) -
                     Matrix<D>::ones(classes, 1) * transposed(labels);
    return transposed(elemwise([] __GPU_CPU__(float x) { return fabsf(x) < 0.5f ? 1.0f : 0.0f; },
                               std::move(diff)));
}

// numpy's hstack (side by side) or vstack (one above the other). It stacks
// the transposes the other way and transposes back, which gives the same
// matrix stored the way layers want their inputs (features x batch), so
// hstack([x, features]) can go into a layer without another copy.
template <class D>
Matrix<D> stack_of(nb::list items, bool horizontal) {
    if (items.size() == 0)
        throw std::invalid_argument("nothing to stack");
    std::vector<Matrix<D>> transposes;  // views of the inputs, no copies
    for (nb::handle item : items)
        transposes.push_back(transposed(nb::cast<const Matrix<D> &>(item)));
    std::vector<MatrixView<D>> views(transposes.begin(), transposes.end());
    try {
        Matrix<D> stacked = horizontal ? Matrix<D>(vstack(views)) : Matrix<D>(hstack(views));
        return transposed(stacked);
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument(horizontal ? "hstack: all matrices must have the same number of rows"
                                               : "vstack: all matrices must have the same number of columns");
    }
}

// ---------------------------------------------------------------------------
// Neural networks

// A network owned by Python: layers are given input first and kept here in
// the output-first order that Juzhen's forward()/backward() use. The layer
// objects are kept alive by the Python Sequential.
template <class D>
class Network {
    std::list<Layer<D> *> layers_;
    std::unique_ptr<Matrix<D>> input_;  // features x batch input of the last forward
    std::unique_ptr<Matrix<int>> ids_;  // or the ids of the last forward, for a network that takes ids

    // x is batch x features; the layers want features x batch, stored
    // without the transpose flag (a numpy array already is; the result of
    // hstack or a matrix product is not)
    static Matrix<D> native(const Matrix<D> &x) { return plain(transposed(x)); }
    static Matrix<D> native_copy(const Matrix<D> &x) { return plain_copy(transposed(x)); }

   public:
    explicit Network(const std::vector<Layer<D> *> &input_first) {
        if (input_first.empty())
            throw std::invalid_argument("a network needs at least one layer");
        for (auto *layer : input_first) {
            if (!layer)
                throw std::invalid_argument("a network cannot contain None");
            for (auto *other : layers_)
                if (other == layer)
                    throw std::invalid_argument("a layer can appear in a network only once");
            layers_.push_front(layer);
        }
        require_valid_structure(layers_);
    }

    const std::list<Layer<D> *> &layers() const { return layers_; }
    bool takes_ids() const { return layers_.back()->takes_ids(); }

    // The same for a network whose first layer is an Embedding. `rows` is the
    // sequence length when the ids came as (batch, seq_len), otherwise 1.
    Matrix<D> forward_ids(IdArray ids, size_t rows) {
        input_.reset();
        ids_.reset();
        auto kept = std::make_unique<Matrix<int>>(ids_matrix(ids, rows));
        const Matrix<D> &out = Juzhen::forward(layers_, *kept);
        ids_ = std::move(kept);
        return transposed(copy_of(out));
    }

    Matrix<D> predict_ids(IdArray ids, size_t rows) {
        input_.reset();
        ids_.reset();
        const Matrix<int> kept = ids_matrix(ids, rows);
        return transposed(copy_of(Juzhen::forward(layers_, kept)));
    }

    // Runs the layers on x and keeps a copy of x for backward().
    Matrix<D> forward(const Matrix<D> &x) {
        ids_.reset();
        input_.reset();
        auto input = std::make_unique<Matrix<D>>(native_copy(x));
        const Matrix<D> &out = Juzhen::forward(layers_, *input);
        input_ = std::move(input);
        // a copy: the next forward() overwrites the layer's output
        return transposed(copy_of(out));
    }

    // forward() without keeping the input; backward() is not possible after it
    Matrix<D> predict(const Matrix<D> &x) {
        ids_.reset();
        input_.reset();
        const Matrix<D> input = native(x);
        return transposed(copy_of(Juzhen::forward(layers_, input)));
    }

    // grad is the gradient of the loss with respect to the last output
    // (batch x outputs); returns the gradient with respect to the input.
    // For a network that takes ids there is no input gradient: this returns
    // the gradient with respect to the embedded tokens (tokens x dim).
    Matrix<D> backward(const Matrix<D> &grad) {
        if (!input_ && !ids_)
            throw std::logic_error("call the network on a batch before backward()");
        // backward() may write into its argument, so it gets a copy
        Matrix<D> upstream = native_copy(grad);
        if (ids_)
            return transposed(Juzhen::backward(layers_, *ids_, std::move(upstream)));
        return transposed(Juzhen::backward(layers_, *input_, std::move(upstream)));
    }

    // J(x) * tangent, the change of the output when x moves along tangent
    Matrix<D> jvp(const Matrix<D> &x, const Matrix<D> &tangent) {
        ids_.reset();
        input_.reset();
        const Matrix<D> xn = native(x), tn = native(tangent);
        return transposed(Juzhen::jvp(layers_, xn, tn));
    }

    std::vector<ParameterRef<D>> parameters() const { return Juzhen::parameters(layers_); }
};

// The trainable parameters of a network, for the optimizer.
template <class D>
struct ParameterList {
    std::vector<ParameterRef<D>> refs;
};

template <class D>
nb::object optional_matrix(bool present, const Matrix<D> &m) {
    return present ? nb::cast(copy_of(m)) : nb::none();
}

// ---------------------------------------------------------------------------
// std::cout and std::cerr go to Python's sys.stdout and sys.stderr, so C++
// output shows up in Jupyter (like CallbackBuf in ios/JuzhenDemo).

class PythonStream : public std::streambuf {
    const char *name_;
    std::string pending_;

   public:
    explicit PythonStream(const char *name) : name_(name) {}

   protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) pending_.push_back((char)ch);
        if (ch == '\n') sync();
        return ch;
    }

    std::streamsize xsputn(const char *s, std::streamsize n) override {
        pending_.append(s, (size_t)n);
        if (std::memchr(s, '\n', (size_t)n)) sync();
        return n;
    }

    int sync() override {
        if (pending_.empty()) return 0;
        nb::gil_scoped_acquire gil;
        try {
            nb::module_::import_("sys").attr(name_).attr("write")(
                nb::str(pending_.data(), pending_.size()));
        } catch (...) {
            // not valid UTF-8, or sys.stdout is gone
        }
        pending_.clear();
        return 0;
    }
};

// Swallows output: C++ static destructors (e.g. Profiler) print after
// Python has shut down, and that output means nothing to a Python user.
class NullStream : public std::streambuf {
   protected:
    int_type overflow(int_type ch) override { return traits_type::not_eof(ch); }
    std::streamsize xsputn(const char *, std::streamsize n) override { return n; }
};

PythonStream python_stdout("stdout"), python_stderr("stderr");
NullStream null_stream;

void attach_streams() {
    std::cout.rdbuf(&python_stdout);
    std::cerr.rdbuf(&python_stderr);
}

void detach_streams() {
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(&null_stream);
    std::cerr.rdbuf(&null_stream);
}

// ---------------------------------------------------------------------------
// Random numbers and devices

#ifdef CUDA
// cuRAND's generator lives as long as this sampler; it is never destroyed.
GPUSampler &gpu_sampler() {
    static GPUSampler *sampler = new GPUSampler(1);
    return *sampler;
}
#endif

void manual_seed(int seed) {
    global_rand_gen.seed(seed);
#ifdef CUDA
    gpu_sampler().setseed(seed);
#elif defined(ROCM_HIP) || defined(APPLE_SILICON)
    GPUSampler sampler(seed);
#endif
}

void synchronize() {
#if defined(CUDA)
    CudaErrorCheck(cudaDeviceSynchronize());
#elif defined(ROCM_HIP)
    if (hipDeviceSynchronize() != hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
#elif defined(APPLE_SILICON)
    mpsSynchronize();
#endif
}

// "cpu" or "gpu"; the backend's own name ("cuda", ...) also means the GPU
bool is_gpu(const std::string &device) {
    if (device == "cpu") return false;
    if (device == "gpu" || device == kBackend) {
#ifdef JZ_HAS_GPU
        return true;
#else
        throw std::invalid_argument("this build of juzhen has no GPU backend; use device='cpu'");
#endif
    }
    throw std::invalid_argument("unknown device '" + device + "'; use 'cpu' or 'gpu'");
}

template <class D>
nb::object move_to(nb::handle self, const Matrix<D> &m, const std::string &device) {
    const bool gpu = is_gpu(device);
    if (gpu == !std::is_same_v<D, float>)
        return nb::borrow(self);
#ifdef JZ_HAS_GPU
    if constexpr (std::is_same_v<D, float>)
        return nb::cast(Matrix<GPU>(m));
    else
        return nb::cast(m.to_host());
#else
    return nb::borrow(self);
#endif
}

// ---------------------------------------------------------------------------
// Bindings for one device type

template <class D>
void bind_layers(nb::module_ &sub) {
    nb::class_<Layer<D>>(sub, "Dense", "Fully connected layer with tanh: tanh(W x + b) (C++ Layer<D>).")
        .def("__init__", [](Layer<D> *self, size_t out, size_t in, const std::string &rule, float std) {
            new (self) Layer<D>(out, in, 1, make_init(rule, std));
        }, "out_features"_a, "in_features"_a, "init"_a = "auto", "std"_a = 0.f)
        .def_prop_ro("weight", [](Layer<D> &l) { return copy_of(l.W()); }, "A copy of W (out x in).")
        .def_prop_ro("bias", [](Layer<D> &l) { return copy_of(l.b()); }, "A copy of b (out x 1).");
    nb::class_<LinearLayer<D>, Layer<D>>(sub, "Linear", "Fully connected layer: W x + b (C++ LinearLayer<D>).")
        .def("__init__", [](LinearLayer<D> *self, size_t out, size_t in, const std::string &rule, float std) {
            new (self) LinearLayer<D>(out, in, 1, make_init(rule, std));
        }, "out_features"_a, "in_features"_a, "init"_a = "auto", "std"_a = 0.f);
    nb::class_<ReluLayer<D>, Layer<D>>(sub, "ReLUDense", "Fully connected layer with ReLU (C++ ReluLayer<D>).")
        .def("__init__", [](ReluLayer<D> *self, size_t out, size_t in, const std::string &rule, float std) {
            new (self) ReluLayer<D>(out, in, 1, make_init(rule, std));
        }, "out_features"_a, "in_features"_a, "init"_a = "auto", "std"_a = 0.f);
    nb::class_<LayerNormLayer<D>, Layer<D>>(sub, "LayerNorm", "Layer normalization over the features (C++ LayerNormLayer<D>).")
        .def(nb::init<int>(), "features"_a);
    nb::class_<EmbeddingLayer<D>, Layer<D>>(sub, "Embedding", "Token ids to vectors (C++ EmbeddingLayer<D>).")
        .def("__init__", [](EmbeddingLayer<D> *self, size_t vocab, size_t dim, const std::string &rule, float std) {
            new (self) EmbeddingLayer<D>(vocab, dim, make_init(rule, std));
        }, "vocab"_a, "dim"_a, "init"_a = "auto", "std"_a = 0.f);
    nb::class_<PositionEmbeddingLayer<D>, Layer<D>>(sub, "PositionEmbedding", "Adds a learned vector per position (C++ PositionEmbeddingLayer<D>).")
        .def("__init__", [](PositionEmbeddingLayer<D> *self, size_t seq_len, size_t dim, const std::string &rule, float std) {
            new (self) PositionEmbeddingLayer<D>(seq_len, dim, make_init(rule, std));
        }, "seq_len"_a, "dim"_a, "init"_a = "auto", "std"_a = 0.f);
    // the tied output keeps its Embedding alive
    nb::class_<TiedLinearLayer<D>, Layer<D>>(sub, "TiedLinear", "Output layer that uses an Embedding's table (C++ TiedLinearLayer<D>).")
        .def(nb::init<EmbeddingLayer<D> &>(), "embedding"_a, nb::keep_alive<1, 2>());
    nb::class_<TransformerLayer<D>, Layer<D>>(sub, "Transformer", "Pre-norm transformer block (C++ TransformerLayer<D>).")
        .def(nb::init<int, int, int, int, int, int, bool>(), "d_model"_a, "d_k"_a, "d_ff"_a, "seq_len"_a,
             "batch_size"_a, "heads"_a = 1, "causal"_a = true);
#ifdef JZ_HAS_CONV
    if constexpr (std::is_same_v<D, ConvType>) {
        nb::class_<ConvLayer, Layer<D>>(sub, "Conv2d", "2-D convolution, optionally with ReLU (C++ ConvLayer).")
            .def("__init__", [](ConvLayer *self, int n, int c_in, int h, int w, int c_out, int kh, int kw, int pad,
                                int stride, bool relu, const std::string &rule, float std) {
                new (self) ConvLayer(n, c_in, h, w, c_out, kh, kw, pad, stride, relu, false, make_init(rule, std));
            }, "batch_size"_a, "in_channels"_a, "height"_a, "width"_a, "out_channels"_a, "kernel_height"_a,
                 "kernel_width"_a, "padding"_a = 0, "stride"_a = 1, "relu"_a = true, "init"_a = "auto", "std"_a = 0.f);
        nb::class_<convtransLayer, Layer<D>>(sub, "ConvTranspose2d", "Transposed 2-D convolution (C++ convtransLayer).")
            .def("__init__", [](convtransLayer *self, int n, int c_in, int h, int w, int c_out, int kh, int kw, int pad,
                                int stride, bool relu, const std::string &rule, float std) {
                new (self) convtransLayer(n, c_in, h, w, c_out, kh, kw, pad, stride, relu, false, make_init(rule, std));
            }, "batch_size"_a, "in_channels"_a, "height"_a, "width"_a, "out_channels"_a, "kernel_height"_a,
                 "kernel_width"_a, "padding"_a = 0, "stride"_a = 1, "relu"_a = true, "init"_a = "auto", "std"_a = 0.f);
    }
#endif
}

template <class D>
void bind_training(nb::module_ &sub) {
    nb::class_<ParameterRef<D>>(sub, "Parameter")
        .def_ro("name", &ParameterRef<D>::name)
        .def_prop_ro("shape", [](const ParameterRef<D> &p) { return std::make_tuple(p.value->num_row(), p.value->num_col()); })
        .def_prop_ro("value", [](const ParameterRef<D> &p) { return copy_of(*p.value); }, "A copy of the parameter.")
        .def_prop_ro("grad", [](const ParameterRef<D> &p) { return optional_matrix(p.has_grad(), p.grad()); },
                     "A copy of the accumulated gradient, or None.")
        .def_prop_ro("trainable", [](const ParameterRef<D> &p) { return p.enabled(); })
        .def("assign", [](const ParameterRef<D> &p, const Matrix<D> &value) {
            if (value.num_row() != p.value->num_row() || value.num_col() != p.value->num_col())
                throw std::invalid_argument("cannot assign shape " + shape_str(value) + " to parameter '" +
                                            p.name + "' of shape " + shape_str(*p.value));
            *p.value = plain_copy(value);
        }, "value"_a, "Set the parameter to a copy of value (same shape, e.g. for a custom initialization).")
        .def("assign", [](const ParameterRef<D> &p, nb::handle value) {
            const std::string type = Py_TYPE(value.ptr())->tp_name;
            const std::string mine = device_name<D>();
            if (type == "juzhen._core.cpu.Matrix" || type == "juzhen._core.gpu.Matrix")
                throw std::invalid_argument("parameter '" + p.name + "' is on '" + mine +
                                            "' but the value is on the other device; use value.to('" + mine + "')");
            throw nb::type_error(("assign() takes a juzhen matrix, got " + type + "; use jz.array(value, device='" +
                                  mine + "')").c_str());
        }, "value"_a)
        .def("__repr__", [](const ParameterRef<D> &p) {
            return "Parameter('" + p.name + "', shape=" + shape_str(*p.value) + ")";
        });

    nb::class_<ParameterList<D>>(sub, "ParameterList")
        .def_prop_ro("device", [](const ParameterList<D> &) { return device_name<D>(); })
        .def("__len__", [](const ParameterList<D> &l) { return l.refs.size(); })
        .def("__getitem__", [](const ParameterList<D> &l, nb::handle key) {
            const Py_ssize_t n = (Py_ssize_t)l.refs.size();
            if (PyUnicode_Check(key.ptr())) {  // by name, as state_dict() and p.name show it
                const std::string name = nb::cast<std::string>(key);
                std::string names;
                for (const auto &p : l.refs) {
                    if (p.name == name) return p;
                    names += (names.empty() ? "'" : ", '") + p.name + "'";
                }
                throw nb::key_error(("no parameter is called '" + name + "'; the names are " + names).c_str());
            }
            if (PyBool_Check(key.ptr()) || !PyIndex_Check(key.ptr()))
                throw nb::type_error("parameters()[...] takes a position (0 is the first, -1 the last) or a name "
                                     "such as 'layer0.weight'; to go through all of them write: "
                                     "for p in net.parameters()");
            const Py_ssize_t given = PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
            if (given == -1 && PyErr_Occurred()) throw nb::python_error();
            const Py_ssize_t i = given < 0 ? given + n : given;
            if (i < 0 || i >= n)
                throw nb::index_error(("parameter index " + std::to_string(given) + " is out of range: the "
                                       "network has " + std::to_string(n) + " parameters").c_str());
            return l.refs[(size_t)i];
        }, nb::arg("key").none(), nb::keep_alive<0, 1>());

    nb::class_<Network<D>>(sub, "Network")
        .def(nb::init<const std::vector<Layer<D> *> &>(), "layers"_a, nb::keep_alive<1, 2>())
        .def("forward", &Network<D>::forward, "x"_a)
        .def("predict", &Network<D>::predict, "x"_a)
        .def("forward_ids", &Network<D>::forward_ids, "ids"_a.noconvert(), "rows"_a)
        .def("predict_ids", &Network<D>::predict_ids, "ids"_a.noconvert(), "rows"_a)
        .def_prop_ro("takes_ids", &Network<D>::takes_ids)
        .def("backward", &Network<D>::backward, "grad"_a)
        .def("jvp", &Network<D>::jvp, "x"_a, "tangent"_a)
        .def("parameters", [](const Network<D> &n) { return ParameterList<D>{n.parameters()}; }, nb::keep_alive<0, 1>());

    nb::class_<AdamW<D>>(sub, "AdamW")
        .def("__init__", [](AdamW<D> *self, const ParameterList<D> &params, float lr, float beta1, float beta2,
                            float eps, float weight_decay) {
            AdamWOptions options;
            options.lr = lr;
            options.beta1 = beta1;
            options.beta2 = beta2;
            options.eps = eps;
            options.weight_decay = weight_decay;
            new (self) AdamW<D>(params.refs, options);
        }, "params"_a, "lr"_a, "beta1"_a, "beta2"_a, "eps"_a, "weight_decay"_a, nb::keep_alive<1, 2>())
        .def("zero_grad", &AdamW<D>::zero_grad)
        .def("step", &AdamW<D>::step)
        .def("set_lr", &AdamW<D>::set_lr, "lr"_a)
        .def("clip_grad_norm", &AdamW<D>::clip_grad_norm, "max_norm"_a);

    // losses take sample-first (batch x classes) matrices; value is 1 x 1
    sub.def("softmax_cross_entropy", [](const Matrix<D> &logits, const Matrix<D> &labels) {
        auto r = Juzhen::softmax_cross_entropy(transposed(logits), transposed(labels));
        return std::make_tuple(std::move(r.value), transposed(r.grad));
    }, "logits"_a, "labels"_a);
    sub.def("mse", [](const Matrix<D> &prediction, const Matrix<D> &target) {
        auto r = Juzhen::mse(transposed(prediction), transposed(target));
        return std::make_tuple(std::move(r.value), transposed(r.grad));
    }, "prediction"_a, "target"_a);
    // weights: one per sample (N x 1); already includes any scaling
    sub.def("masked_softmax_cross_entropy", [](const Matrix<D> &logits, const Matrix<D> &labels,
                                               const Matrix<D> &weights) {
        if (weights.num_row() != logits.num_row() || weights.num_col() != 1)
            throw std::invalid_argument("weights must have shape (" + std::to_string(logits.num_row()) +
                                        ", 1), got " + shape_str(weights));
        auto r = Juzhen::masked_softmax_cross_entropy(transposed(logits), transposed(labels), transposed(weights));
        return std::make_tuple(std::move(r.value), transposed(r.grad));
    }, "logits"_a, "labels"_a, "weights"_a);
    sub.def("error_rate", [](const Matrix<D> &scores, const Matrix<D> &labels) {
        return Juzhen::error_rate(transposed(scores), transposed(labels));
    }, "scores"_a, "labels"_a);

    // The same three with class numbers (one int per row of logits).
    sub.def("softmax_cross_entropy_labels", [](const Matrix<D> &logits, IdArray labels, int ignore_index) {
        const Matrix<D> z = transposed(logits);
        auto r = Juzhen::softmax_cross_entropy(z, ids_matrix(labels), ignore_index);
        return std::make_tuple(std::move(r.value), transposed(r.grad));
    }, "logits"_a, "labels"_a.noconvert(), "ignore_index"_a);
    sub.def("masked_softmax_cross_entropy_labels", [](const Matrix<D> &logits, IdArray labels,
                                                      const Matrix<D> &weights, int ignore_index) {
        const Matrix<D> z = transposed(logits);
        auto r = Juzhen::masked_softmax_cross_entropy(z, ids_matrix(labels), transposed(weights), ignore_index);
        return std::make_tuple(std::move(r.value), transposed(r.grad));
    }, "logits"_a, "labels"_a.noconvert(), "weights"_a, "ignore_index"_a);
    sub.def("error_rate_labels", [](const Matrix<D> &scores, IdArray labels, int ignore_index) {
        return Juzhen::error_rate(transposed(scores), ids_matrix(labels), ignore_index);
    }, "scores"_a, "labels"_a.noconvert(), "ignore_index"_a);

    sub.def("save_checkpoint", [](const Network<D> &net, const std::string &path, uint64_t epoch, uint64_t step,
                                  uint64_t data_position) {
        TrainingProgress progress{epoch, step, data_position};
        std::string error;
        if (!Juzhen::save_checkpoint(net.layers(), path, progress, &error))
            throw std::runtime_error("cannot save checkpoint " + path + ": " + error);
    }, "network"_a, "path"_a, "epoch"_a, "step"_a, "data_position"_a);
    sub.def("load_checkpoint", [](const Network<D> &net, const std::string &path) {
        TrainingProgress progress;
        std::string error;
        if (!Juzhen::load_checkpoint(net.layers(), path, progress, &error)) {
            // The stored type names carry the device ("LinearLayer<float>" against
            // "LinearLayer<ROCMfloat>"): say that in the package's words.
            const size_t in_file = error.find("' in the file but a '");
            if (in_file != std::string::npos) {
                const std::string file_type = error.substr(0, in_file);
                const bool file_gpu = file_type.find("CUDAfloat") != std::string::npos ||
                                      file_type.find("ROCMfloat") != std::string::npos ||
                                      file_type.find("MPSfloat") != std::string::npos;
                if (file_gpu == std::is_same_v<D, float>)
                    error = std::string("the file was saved from a network on '") + (file_gpu ? "gpu" : "cpu") +
                            "' but this network is on '" + device_name<D>() +
                            "'; to move weights between devices use other.load_state_dict(net.state_dict())";
            }
            throw std::runtime_error("cannot load checkpoint " + path + ": " + error);
        }
        return std::make_tuple(progress.epoch, progress.step, progress.data_position);
    }, "network"_a, "path"_a);
}

template <class D>
void bind(nb::module_ &top, nb::module_ &sub) {
    using Mat = Matrix<D>;
    // weak references let nn.py remember which output a loss came from
    // without keeping the matrices alive
    auto cls = nb::class_<Mat>(sub, "Matrix", nb::is_weak_referenceable(),
                               "A float32 matrix in Juzhen's memory pool.")
        .def_prop_ro("shape", [](const Mat &m) { return std::make_tuple(m.num_row(), m.num_col()); })
        .def_prop_ro("device", [](const Mat &) { return device_name<D>(); })
        .def_prop_ro("T", [](const Mat &m) { return transposed(m); }, "The transpose; shares storage, no copy.")
        .def("numpy", &to_numpy<D>, "A copy as a numpy array.")
        .def("__array__", [](const Mat &m, nb::handle dtype, nb::handle copy) {
            nb::object a = nb::cast(to_numpy(m));
            return dtype.is_none() ? a : a.attr("astype")(dtype);
        }, "dtype"_a = nb::none(), "copy"_a = nb::none())
        .def("item", &item<D>, "The value of a 1 x 1 matrix.")
        .def("__float__", &item<D>)
        .def("copy", &copy_of<D>)
        .def("to", [](nb::handle self, const std::string &device) {
            return move_to(self, nb::cast<const Mat &>(self), device);
        }, "device"_a, "This matrix on another device (a copy), or itself.")
        .def("__getitem__", &get_item<D>, nb::arg("key").none())
        .def("__repr__", &repr<D>)
        .def("__len__", [](const Mat &m) { return m.num_row(); })
        .def("__bool__", &truth<D>)
        .def("__matmul__", &matmul<D>)
        .def("__add__", [](const Mat &a, const Mat &b) { return binary(a, b, Op::add); })
        .def("__sub__", [](const Mat &a, const Mat &b) { return binary(a, b, Op::sub); })
        .def("__mul__", [](const Mat &a, const Mat &b) { return binary(a, b, Op::mul); })
        .def("__truediv__", [](const Mat &a, const Mat &b) { return binary(a, b, Op::div); })
        .def("__add__", [](const Mat &a, double s) { return a + s; })
        .def("__radd__", [](const Mat &a, double s) { return s + a; })
        .def("__sub__", [](const Mat &a, double s) { return a - s; })
        .def("__rsub__", [](const Mat &a, double s) { return s - a; })
        .def("__mul__", [](const Mat &a, double s) { return a * s; })
        .def("__rmul__", [](const Mat &a, double s) { return s * a; })
        .def("__truediv__", [](const Mat &a, double s) { return a / s; })
        .def("__rtruediv__", [](const Mat &a, double s) { return s / a; })
        .def("__neg__", [](const Mat &a) { return -a; })
        .def("__pow__", [](const Mat &a, double p) {
            if (p == 2) return square(a);
            if (p == 0.5) return Mat(sqrt(a));
            if (p == 1) return copy_of(a);
            throw std::invalid_argument("only the powers 0.5, 1 and 2 are supported; for positive entries "
                                        "jz.exp(p * jz.log(A)) is A to the power p");
        })
        .def("__iadd__", [](Mat &a, const Mat &b) -> Mat & { inplace(a, b, Op::add); return a; }, nb::rv_policy::none)
        .def("__isub__", [](Mat &a, const Mat &b) -> Mat & { inplace(a, b, Op::sub); return a; }, nb::rv_policy::none)
        .def("__iadd__", [](Mat &a, double s) -> Mat & { a.add((float)s, 1.0f); return a; }, nb::rv_policy::none)
        .def("__isub__", [](Mat &a, double s) -> Mat & { a.add((float)-s, 1.0f); return a; }, nb::rv_policy::none)
        .def("__imul__", [](Mat &a, double s) -> Mat & { a.scale((float)s); return a; }, nb::rv_policy::none)
        .def("__itruediv__", [](Mat &a, double s) -> Mat & { a.scale((float)(1.0 / s)); return a; }, nb::rv_policy::none);

    // Comparisons, element by element. Anything that is not a matrix or a
    // number is "not equal" for == and != (as for any Python object) and a
    // TypeError for the ordering operators; numpy arrays and lists get a
    // message of their own.
    {
        const std::tuple<const char *, Cmp> comparisons[] = {
            {"__eq__", Cmp::eq}, {"__ne__", Cmp::ne}, {"__lt__", Cmp::lt},
            {"__le__", Cmp::le}, {"__gt__", Cmp::gt}, {"__ge__", Cmp::ge}};
        for (const auto &[name, op] : comparisons) {
            const Cmp c = op;
            cls.def(name, [c](const Mat &a, const Mat &b) { return compare(a, b, c); });
            cls.def(name, [c](const Mat &a, nb::handle other) -> nb::object {
                const std::string type = Py_TYPE(other.ptr())->tp_name;
                // a Python or numpy number (handled here, not by a `double`
                // overload: this overload would win for an int and say
                // "not comparable")
                if (PyFloat_Check(other.ptr()) || PyLong_Check(other.ptr()) ||
                    (type.rfind("numpy.", 0) == 0 && type != "numpy.ndarray" && PyNumber_Check(other.ptr()))) {
                    const double s = PyFloat_AsDouble(other.ptr());
                    if (s == -1.0 && PyErr_Occurred()) throw nb::python_error();
                    return nb::cast(compare(a, (float)s, c));
                }
                if (type == "numpy.ndarray" || type == "list" || type == "tuple")
                    throw nb::type_error(("a juzhen matrix cannot be compared with a " + type +
                                          "; make it a matrix first: jz.array(a, device='" + device_name<D>() +
                                          "'), which turns a 1-D array of n values into n x 1. For the accuracy "
                                          "of a classifier use 1 - nn.error_rate(scores, labels)").c_str());
#ifdef JZ_HAS_GPU
                if (type == (std::is_same_v<D, float> ? "juzhen._core.gpu.Matrix" : "juzhen._core.cpu.Matrix"))
                    throw std::invalid_argument(std::string("the matrices are on different devices; move one with "
                                                            ".to('") + device_name<D>() + "')");
#endif
                return nb::borrow(Py_NotImplemented);
            }, nb::arg("other").none());
        }
    }

    // Whatever the overloads above did not take, answered in a sentence
    // instead of nanobind's list of C++ signatures. A numpy array or a list
    // gets the way to convert it; a number goes on to the overloads above
    // (they convert an int on the second pass); for any other type Python is
    // told "not implemented", so the other operand can try and Python's own
    // short message follows otherwise.
    for (const char *op : {"__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__",
                           "__rtruediv__", "__matmul__", "__rmatmul__", "__iadd__", "__isub__", "__imul__",
                           "__itruediv__", "__pow__"}) {
        const std::string name = op;
        cls.def(op, [name](const Mat &, nb::handle other) -> nb::object {
            const std::string type = Py_TYPE(other.ptr())->tp_name;
            const std::string mine = std::is_same_v<D, float> ? "cpu" : "gpu";
            if (type == "numpy.ndarray" || type == "list" || type == "tuple")
                throw nb::type_error(("a juzhen matrix cannot be combined with a " + type +
                                      "; make it a matrix first: jz.array(a, device='" + device_name<D>() + "')")
                                         .c_str());
            if (type == "juzhen._core.cpu.Matrix" || type == "juzhen._core.gpu.Matrix") {
                if (type != "juzhen._core." + mine + ".Matrix") {
                    const std::string theirs = mine == "cpu" ? "gpu" : "cpu";
                    throw std::invalid_argument("the matrices are on different devices ('" + mine + "' and '" +
                                                theirs + "'); move one with .to('" + mine + "') or .to('" +
                                                theirs + "')");
                }
                if (name == "__imul__" || name == "__itruediv__") {
                    const Mat &b = nb::cast<const Mat &>(other);
                    if (b.num_row() == 1 && b.num_col() == 1)
                        throw nb::next_overload();  // a 1 x 1 matrix is taken as its number
                    const std::string sign = name == "__imul__" ? "*" : "/";
                    throw nb::type_error(("A " + sign + "= B changes A in place for a number only; with a matrix "
                                          "write A = A " + sign + " B (a new matrix)").c_str());
                }
                if (name == "__pow__")
                    throw nb::type_error("the power must be a number: A ** 0.5, A ** 1 and A ** 2 are supported; "
                                         "for positive entries jz.exp(B * jz.log(A)) is A to the power B");
            }
            if (PyFloat_Check(other.ptr()) || PyLong_Check(other.ptr()) ||
                (type.rfind("numpy.", 0) == 0 && PyNumber_Check(other.ptr()))) {
                if (name == "__matmul__" || name == "__rmatmul__")
                    throw nb::type_error("@ is the matrix product and needs two matrices; to multiply by a "
                                         "number write A * 2");
                throw nb::next_overload();
            }
            return nb::borrow(Py_NotImplemented);
        }, nb::arg("other").none());
    }

    sub.def("from_numpy", &from_numpy<D>, "array"_a);
    // juzhen/__init__.py checks the sizes and says what to write; sized() is the backstop
    sub.def("zeros", [](size_t r, size_t c) { sized(r, c); return Mat::zeros(r, c); }, "rows"_a, "cols"_a);
    sub.def("ones", [](size_t r, size_t c) { sized(r, c); return Mat::ones(r, c); }, "rows"_a, "cols"_a);
    sub.def("randn", [](size_t r, size_t c) { sized(r, c); return Mat::randn(r, c); }, "rows"_a, "cols"_a);
    sub.def("rand", [](size_t r, size_t c) { sized(r, c); return Mat::rand(r, c); }, "rows"_a, "cols"_a);

    // The elementwise functions are overloaded on the device type, so
    // juzhen.exp(A) works for a matrix on either device.
    top.def("exp", [](const Mat &m) { return exp(m); });
    top.def("log", [](const Mat &m) { return log(m); });
    top.def("tanh", [](const Mat &m) { return tanh(m); });
    top.def("square", [](const Mat &m) { return square(m); });
    top.def("sqrt", [](const Mat &m) { return Mat(sqrt(m)); });
    top.def("sin", &sin_of<D>);
    top.def("cos", &cos_of<D>);
    top.def("relu", &relu_of<D>);
    top.def("sum", &sum_axis<D>, "m"_a, "axis"_a);
    top.def("abs", &abs_of<D>);
    top.def("max", &max_axis<D>, "m"_a, "axis"_a);
    top.def("min", &min_axis<D>, "m"_a, "axis"_a);
    top.def("argmax", &argmax_axis<D>, "m"_a, "axis"_a);
    top.def("topk", &topk_axis<D>, "m"_a, "k"_a, "axis"_a, "largest"_a);
    top.def("softmax", &softmax_axis<D>, "m"_a, "axis"_a);
    top.def("logsumexp", &logsumexp_axis<D>, "m"_a, "axis"_a);
    sub.def("one_hot", &one_hot_of<D>, "labels"_a, "classes"_a);
    sub.def("stack", &stack_of<D>, "matrices"_a, "horizontal"_a);

    bind_layers<D>(sub);
    bind_training<D>(sub);
}

}  // namespace jzpy

using namespace jzpy;

NB_MODULE(_core, m) {
    m.doc() = "C++ half of the juzhen package; see python/juzhen/__init__.py";
    nb::set_leak_warnings(false);
#ifndef LOGGING_OFF
    // The library's info messages ("GPU sampler is initialized ...") are
    // for C++ programs; Python users see only warnings and errors.
    spdlog::set_level(spdlog::level::warn);
#endif

    Juzhen::init();
#ifdef CUDA
    gpu_sampler();
#endif

    nb::module_ cpu = m.def_submodule("cpu", "Matrices and layers on the CPU (Matrix<float>).");
    bind<float>(m, cpu);
#ifdef JZ_HAS_GPU
    nb::module_ gpu = m.def_submodule("gpu", "Matrices and layers on the GPU backend of this build.");
    bind<GPU>(m, gpu);
#endif

    m.attr("backend") = kBackend;
#ifdef JZ_HAS_GPU
    m.attr("has_gpu") = true;
#else
    m.attr("has_gpu") = false;
#endif
    m.attr("project_dir") = PROJECT_DIR;
    m.def("manual_seed", &manual_seed, "seed"_a);
    m.def("synchronize", &synchronize, "Wait until the GPU has finished all queued work.");
    m.def("empty_cache", &Juzhen::release_cached, "Free the pooled memory no matrix is using.");
    m.def("attach_streams", &attach_streams);
    m.def("detach_streams", &detach_streams);
}
