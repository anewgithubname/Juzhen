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

#include <cstring>
#include <memory>
#include <streambuf>

#include "../../cpp/juzhen.hpp"
#include "../../ml/checkpoint.hpp"
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

// ---------------------------------------------------------------------------
// numpy

using InArray = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, float, nb::ndim<2>>;

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
    const Matrix<D> y = broadcast(b, a.num_row(), a.num_col());
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

// an int (or numpy integer) index into a dimension of size n
size_t position(nb::handle item, size_t n) {
    nb::object index = nb::steal(PyNumber_Index(item.ptr()));
    if (!index.is_valid()) throw nb::python_error();
    Py_ssize_t i = PyLong_AsSsize_t(index.ptr());
    if (i == -1 && PyErr_Occurred()) throw nb::python_error();
    if (i < 0) i += (Py_ssize_t)n;
    if (i < 0 || i >= (Py_ssize_t)n)
        throw nb::index_error(("index out of bounds for size " + std::to_string(n)).c_str());
    return (size_t)i;
}

Index parse_index(nb::handle key, size_t n) {
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
        index.contiguous = false;
        for (nb::handle item : key)
            index.list.push_back(position(item, n));
    } else {
        index.start = position(key, n);
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
    const Index r = parse_index(row_key, m.num_row());
    const Index c = parse_index(col_key, m.num_col());

    if (r.scalar && c.scalar) {
        // one element; reading it waits for the GPU
        if constexpr (std::is_same_v<D, float>)
            return nb::float_(m.elem(r.start, c.start));
        else
            return nb::float_(host(take_block(m, r.start, r.stop, c.start, c.stop)).elem(0, 0));
    }
    if (r.contiguous && c.contiguous)
        return nb::cast(take_block(m, r.start, r.stop, c.start, c.stop));
    // lists of indices are gathered on the host
    Matrix<float> picked = host(m).slice(r.as_list(), c.as_list());
    return nb::cast(to_device<D>(std::move(picked)));
}

// ---------------------------------------------------------------------------
// More functions: reductions along an axis, softmax, top-k, one-hot, stacking.
// Axis 0 works down each column (one result per column, 1 x cols); axis 1
// along each row (rows x 1). The reductions run on the device on CUDA and
// ROCm; on Metal reduce() and elemwise() copy through the host.

void check_axis(int axis) {
    if (axis != 0 && axis != 1)
        throw std::invalid_argument("axis must be 0 or 1, got " + std::to_string(axis));
}

template <class D>
Matrix<D> max_axis(const Matrix<D> &m, int axis) {
    check_axis(axis);
    return reduce([] __GPU_CPU__(float *v, float *out, int n, int) {
        float best = v[0];
        for (int i = 1; i < n; ++i) best = v[i] > best ? v[i] : best;
        out[0] = best;
    }, m, axis, 1);
}

template <class D>
Matrix<D> min_axis(const Matrix<D> &m, int axis) {
    check_axis(axis);
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
    }

    const std::list<Layer<D> *> &layers() const { return layers_; }

    // Runs the layers on x and keeps a copy of x for backward().
    Matrix<D> forward(const Matrix<D> &x) {
        input_.reset();
        auto input = std::make_unique<Matrix<D>>(native_copy(x));
        const Matrix<D> &out = Juzhen::forward(layers_, *input);
        input_ = std::move(input);
        // a copy: the next forward() overwrites the layer's output
        return transposed(copy_of(out));
    }

    // forward() without keeping the input; backward() is not possible after it
    Matrix<D> predict(const Matrix<D> &x) {
        input_.reset();
        const Matrix<D> input = native(x);
        return transposed(copy_of(Juzhen::forward(layers_, input)));
    }

    // grad is the gradient of the loss with respect to the last output
    // (batch x outputs); returns the gradient with respect to the input.
    Matrix<D> backward(const Matrix<D> &grad) {
        if (!input_)
            throw std::logic_error("call the network on a batch before backward()");
        // backward() may write into its argument, so it gets a copy
        Matrix<D> upstream = native_copy(grad);
        return transposed(Juzhen::backward(layers_, *input_, std::move(upstream)));
    }

    // J(x) * tangent, the change of the output when x moves along tangent
    Matrix<D> jvp(const Matrix<D> &x, const Matrix<D> &tangent) {
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
        .def(nb::init<size_t, size_t, size_t>(), "out_features"_a, "in_features"_a, "batch_size"_a = 1)
        .def_prop_ro("weight", [](Layer<D> &l) { return copy_of(l.W()); }, "A copy of W (out x in).")
        .def_prop_ro("bias", [](Layer<D> &l) { return copy_of(l.b()); }, "A copy of b (out x 1).");
    nb::class_<LinearLayer<D>, Layer<D>>(sub, "Linear", "Fully connected layer: W x + b (C++ LinearLayer<D>).")
        .def(nb::init<size_t, size_t, size_t>(), "out_features"_a, "in_features"_a, "batch_size"_a = 1);
    nb::class_<ReluLayer<D>, Layer<D>>(sub, "ReLUDense", "Fully connected layer with ReLU (C++ ReluLayer<D>).")
        .def(nb::init<size_t, size_t, size_t>(), "out_features"_a, "in_features"_a, "batch_size"_a = 1);
    nb::class_<LayerNormLayer<D>, Layer<D>>(sub, "LayerNorm", "Layer normalization over the features (C++ LayerNormLayer<D>).")
        .def(nb::init<int, int>(), "features"_a, "batch_size"_a);
    nb::class_<TransformerLayer<D>, Layer<D>>(sub, "Transformer", "Pre-norm transformer block (C++ TransformerLayer<D>).")
        .def(nb::init<int, int, int, int, int, int, bool>(), "d_model"_a, "d_k"_a, "d_ff"_a, "seq_len"_a,
             "batch_size"_a, "heads"_a = 1, "causal"_a = true);
#ifdef JZ_HAS_CONV
    if constexpr (std::is_same_v<D, ConvType>) {
        nb::class_<ConvLayer, Layer<D>>(sub, "Conv2d", "2-D convolution, optionally with ReLU (C++ ConvLayer).")
            .def(nb::init<int, int, int, int, int, int, int, int, int, bool>(), "batch_size"_a, "in_channels"_a,
                 "height"_a, "width"_a, "out_channels"_a, "kernel_height"_a, "kernel_width"_a, "padding"_a = 0,
                 "stride"_a = 1, "relu"_a = true);
        nb::class_<convtransLayer, Layer<D>>(sub, "ConvTranspose2d", "Transposed 2-D convolution (C++ convtransLayer).")
            .def(nb::init<int, int, int, int, int, int, int, int, int, bool>(), "batch_size"_a, "in_channels"_a,
                 "height"_a, "width"_a, "out_channels"_a, "kernel_height"_a, "kernel_width"_a, "padding"_a = 0,
                 "stride"_a = 1, "relu"_a = true);
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
        .def("__repr__", [](const ParameterRef<D> &p) {
            return "Parameter('" + p.name + "', shape=" + shape_str(*p.value) + ")";
        });

    nb::class_<ParameterList<D>>(sub, "ParameterList")
        .def_prop_ro("device", [](const ParameterList<D> &) { return device_name<D>(); })
        .def("__len__", [](const ParameterList<D> &l) { return l.refs.size(); })
        .def("__getitem__", [](const ParameterList<D> &l, size_t i) {
            if (i >= l.refs.size()) throw nb::index_error();
            return l.refs[i];
        }, nb::keep_alive<0, 1>());

    nb::class_<Network<D>>(sub, "Network")
        .def(nb::init<const std::vector<Layer<D> *> &>(), "layers"_a, nb::keep_alive<1, 2>())
        .def("forward", &Network<D>::forward, "x"_a)
        .def("predict", &Network<D>::predict, "x"_a)
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
        if (!Juzhen::load_checkpoint(net.layers(), path, progress, &error))
            throw std::runtime_error("cannot load checkpoint " + path + ": " + error);
        return std::make_tuple(progress.epoch, progress.step, progress.data_position);
    }, "network"_a, "path"_a);
}

template <class D>
void bind(nb::module_ &top, nb::module_ &sub) {
    using Mat = Matrix<D>;
    auto cls = nb::class_<Mat>(sub, "Matrix", "A float32 matrix in Juzhen's memory pool.")
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
        .def("__getitem__", &get_item<D>)
        .def("__repr__", &repr<D>)
        .def("__len__", [](const Mat &m) { return m.num_row(); })
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
            throw std::invalid_argument("only the powers 0.5, 1 and 2 are supported");
        })
        .def("__iadd__", [](Mat &a, const Mat &b) -> Mat & { inplace(a, b, Op::add); return a; }, nb::rv_policy::none)
        .def("__isub__", [](Mat &a, const Mat &b) -> Mat & { inplace(a, b, Op::sub); return a; }, nb::rv_policy::none)
        .def("__iadd__", [](Mat &a, double s) -> Mat & { a.add((float)s, 1.0f); return a; }, nb::rv_policy::none)
        .def("__isub__", [](Mat &a, double s) -> Mat & { a.add((float)-s, 1.0f); return a; }, nb::rv_policy::none)
        .def("__imul__", [](Mat &a, double s) -> Mat & { a.scale((float)s); return a; }, nb::rv_policy::none)
        .def("__itruediv__", [](Mat &a, double s) -> Mat & { a.scale((float)(1.0 / s)); return a; }, nb::rv_policy::none);

#ifdef JZ_HAS_GPU
    // A matrix of the other device: say so, instead of nanobind's list of
    // the overloads that did not match.
    using Other = std::conditional_t<std::is_same_v<D, float>, GPU, float>;
    for (const char *op : {"__add__", "__sub__", "__mul__", "__truediv__", "__matmul__", "__iadd__", "__isub__"})
        cls.def(op, [](const Mat &, const Matrix<Other> &) -> Mat {
            throw std::invalid_argument(std::string("the matrices are on different devices ('") +
                                        device_name<D>() + "' and '" + device_name<Other>() +
                                        "'); move one with .to('" + device_name<D>() + "') or .to('" +
                                        device_name<Other>() + "')");
        });
#endif

    sub.def("from_numpy", &from_numpy<D>, "array"_a);
    sub.def("zeros", [](size_t r, size_t c) { return Mat::zeros(r, c); }, "rows"_a, "cols"_a);
    sub.def("ones", [](size_t r, size_t c) { return Mat::ones(r, c); }, "rows"_a, "cols"_a);
    sub.def("randn", [](size_t r, size_t c) { return Mat::randn(r, c); }, "rows"_a, "cols"_a);
    sub.def("rand", [](size_t r, size_t c) { return Mat::rand(r, c); }, "rows"_a, "cols"_a);

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
