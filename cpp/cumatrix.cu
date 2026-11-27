/**
 * @file cumatrix.cu
 * @brief some non-kernel implementation of CUDA matrix class
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
    Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#include "cumatrix.cuh"

using namespace std;

GPU_handle Matrix<CUDAfloat>::global_handle = NULL;

Matrix<CUDAfloat>::Matrix(const Matrix<float>& M)
{
    static Profiler profiler("upload to CUDA memory");
    this->handle = global_handle;
    this->name = "cu_" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    CUDAfloat* p = Memory<CUDAfloat>::allocate( numcol * numrow);
    elements = shared_ptr<CUDAfloat[]>(p, [](CUDAfloat* p) {
        Memory<CUDAfloat>::free(p);
        });

    profiler.start();
    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyHostToDevice);
    profiler.end();

    if (stat != cudaSuccess) {
        LOG_ERROR("host to device memory copy failed, {}.", cudaGetErrorString(stat));
    }
}

Matrix<CUDAfloat>::Matrix(const char* name, size_t numrow, size_t numcol, int trans)
{
    this->handle = global_handle;
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    CUDAfloat* p = Memory<CUDAfloat>::allocate( numcol * numrow);
    elements.reset(p, [](CUDAfloat* p) {
        Memory<CUDAfloat>::free(p);
        });

    zeros();
}

Matrix<CUDAfloat>::Matrix(const Matrix<CUDAfloat>& M) {
    LOG_DEBUG("cuda copy constructor called");
    this->handle = M.handle;
    this->name = "copy of" + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<CUDAfloat[]>(
        Memory<CUDAfloat>::allocate(  numcol * numrow), [](CUDAfloat* p) {
            Memory<CUDAfloat>::free(p);
        });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", cudaGetErrorString(stat));
    }

}

Matrix<CUDAfloat>::Matrix(Matrix<CUDAfloat>&& M) noexcept {
    LOG_DEBUG("cuda move constructor called");
    this->handle = M.handle;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
}

Matrix<CUDAfloat>& Matrix<CUDAfloat>::operator=(const Matrix<CUDAfloat>& M) {
    if (this == &M) return *this;
    LOG_DEBUG("cuda copy assignment called");
    this->handle = M.handle;
    this->name = "copy of " + M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = std::shared_ptr<CUDAfloat[]>(
        Memory<CUDAfloat>::allocate(  numcol * numrow), [](CUDAfloat* p) {
            Memory<CUDAfloat>::free(p);
        });

    cudaError_t stat = cudaMemcpy(elements.get(), M.elements.get(), numcol * numrow * sizeof(float), cudaMemcpyDeviceToDevice);
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory copy failed, {}.", cudaGetErrorString(stat));
        ERROR_OUT;
    }

    return *this;
}

Matrix<CUDAfloat>& Matrix<CUDAfloat>::operator=(Matrix<CUDAfloat>&& M) noexcept {
    if (this == &M) return *this;
    LOG_DEBUG("cuda move assignment called");
    this->handle = M.handle;
    this->name = M.name;
    this->numrow = M.numrow;
    this->numcol = M.numcol;
    this->transpose = M.transpose;
    elements = M.elements;
    M.elements = nullptr;
    return *this;
}


void Matrix<CUDAfloat>::ones()
{
    fill(*this, 1.0f);
}

void Matrix<CUDAfloat>::zeros()
{
    LOG_DEBUG("cuda zeros called, {}, size, {}. ", (void *)elements.get(),  numcol * numrow * sizeof(CUDAfloat));
    // cudaError_t stat = cudaMemset( elements.get(), 0, numcol * numrow * sizeof(GPUfloat));
    // if (stat != cudaSuccess) {
    //     LOG_ERROR("device memory set zero failed, {}, size {}.", stat, numcol * numrow * sizeof(GPUfloat));
    // }
    fill(*this, 0.0f);
}

Matrix<float> Matrix<CUDAfloat>::to_host() const
{
    STATIC_TIC;
    Matrix<float> ret((string(name) + "->host").c_str(), numrow, numcol, transpose);
    // if (numrow * numcol == 1) {
    //     p.start();
    // }
    //GPU_status stat = cublasGetMatrix(numrow, numcol, sizeof(float), elements.get(), 
    //                numrow, ret.elements.get(), numrow);
    cudaError_t stat = cudaMemcpy(ret.elements.get(), (float *) elements.get(), sizeof(float) * numrow * numcol, ::cudaMemcpyDeviceToHost);
    // if (numrow * numcol == 1) {
    //     p.end();
    // }
    if (stat != cudaSuccess) {
        LOG_ERROR("device memory upload to host failed, {}.", cudaGetErrorString(stat));
        ERROR_OUT;
    }
    STATIC_TOC;
    return ret;
}


float Matrix<CUDAfloat>::norm() const
{
    float res;
    CuBLASErrorCheck(
        cublasSnrm2(handle, numrow * numcol, (float *) elements.get(), 1, &res)
    )
    return res;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::dot(const Matrix<CUDAfloat>& B) const
{
    STATIC_TIC;
    //check if the dimensions are compatible
    if (num_col() != B.num_row())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<CUDAfloat> C("dot", num_row(), B.num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f;
    CuBLASErrorCheck( 
        cublasSgemm(handle, transA, transB,
        num_row(), B.num_col(), num_col(), &one, (float *) elements.get(), numrow,
        (float *) B.elements.get(), B.numrow, &one, (float *) C.elements.get(), C.numrow) 
    )
    STATIC_TOC;
    return C;
}
// d_out = s1*d_in + a
__global__ void affineKernel(float *d_out, const float *d_in, float s1, float a, size_t numElements)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = s1 * d_in[i] + a;
    }
}

//s1*M + a, in one pass (it used to fill the result with a and then add s1*M)
Matrix<CUDAfloat> Matrix<CUDAfloat>::add(float a, float s1) const
{
    Matrix<CUDAfloat> C = uninitialized("add", numrow, numcol, transpose);
    size_t numElem = numrow * numcol;
    affineKernel<<<cudaConfig(numElem)>>>((float*) C.elements.get(), (const float*) elements.get(), s1, a, numElem);
    return C;
}

//rvalue s1*M + a
void Matrix<CUDAfloat>::add(float a, float s1)
{
    size_t numElem = numrow * numcol;
    addKernel<<<cudaConfig(numElem)>>>((float*) elements.get(), s1, a, numElem);
}

//M = s1*M
void Matrix<CUDAfloat>::scale(float s1){
    //original content lM is replaced to store the scaled result.
    size_t numelems = numrow * numcol;
    CuBLASErrorCheck( 
        cublasSscal(handle, numelems, (const float *) &s1, (float *)elements.get(), 1)
    )
}

//s1*M + s2*B
Matrix<CUDAfloat> Matrix<CUDAfloat>::add(const Matrix<CUDAfloat>& B, float s1, float s2) const
{
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix<CUDAfloat> C("add", num_row(), num_col());

    cublasOperation_t transA = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    CuBLASErrorCheck(
        cublasSgeam(handle, transA, transB, num_row(), num_col(), &s1, (float *) elements.get(), numrow,
        &s2, (float *) B.elements.get(), B.numrow, (float *) C.elements.get(), C.numrow)
    )

    return C;
}
//in place s1*M + s2*B
void Matrix<CUDAfloat>::add(const Matrix<CUDAfloat>& B, float s1, float s2) {
    // check if the dimensions are compatible
    if (num_row() != B.num_row() || num_col() != B.num_col()){
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = transpose != B.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    CuBLASErrorCheck(
        cublasSgeam(handle, transA, transB, numrow, numcol, &s1, (float*)elements.get(), numrow,
            &s2, (float*)B.elements.get(), B.numrow, (float*)elements.get(), numrow)
    )

}

// elementwise division
__global__ void divKernel(float *d_out, float *d_in, size_t numElements)
{

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] /= d_in[i];
    }
}


// inplace division
__global__ void divKernel(float *d_out, float d_in1, float *d_in2, size_t numElements)
{

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = d_in1 / d_in2[i];
    }
}

// M_new = M / l
Matrix<CUDAfloat> Matrix<CUDAfloat>::eleminv(double l) const{
    Matrix<CUDAfloat> result = uninitialized("elem_rec", numrow, numcol, transpose);

    size_t numElem = numrow * numcol;
    divKernel<<<cudaConfig(numElem)>>>((float*) result.elements.get(), (float)l, (float*) elements.get(), numElem);

    return result;
}

// d_out = d_a ./ d_b
__global__ void quotientKernel(float *d_out, const float *d_a, const float *d_b, size_t numElements)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = d_a[i] / d_b[i];
    }
}

// this ./ B: one pass when the two are stored alike
Matrix<CUDAfloat> Matrix<CUDAfloat>::elemdiv(const Matrix<CUDAfloat>& B) const {
    if (num_row() != B.num_row() || num_col() != B.num_col()) {
        throw std::invalid_argument("Matrix dimensions are not compatible");
    }
    if (transpose != B.transpose) {
        return hadmd(*this, B.eleminv(1.0));
    }
    Matrix<CUDAfloat> C = uninitialized("elemdiv", numrow, numcol, transpose);
    size_t numElem = numrow * numcol;
    quotientKernel<<<cudaConfig(numElem)>>>((float*) C.elements.get(), (const float*) elements.get(),
                                            (const float*) B.elements.get(), numElem);
    return C;
}

// M = M / l
void Matrix<CUDAfloat>::eleminv(double l){
    LOG_DEBUG("rval l/rM called!");

    size_t numElem = numrow * numcol;
    divKernel <<<cudaConfig(numElem)>>> ((float*) elements.get(), (float)l, (float*) elements.get(), numElem);
}

const Matrix<CUDAfloat> Matrix<CUDAfloat>::T() const
{
    string newname = name + "_T";
    Matrix<CUDAfloat> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}
// ---------------------------------------------------------------------------
// max, min and argmax along an axis. reduce() gives each stored column to one
// thread, and for the other direction copies the matrix transposed first;
// these two kernels share the work on one result among the threads of a block
// and read the storage as it lies.
//
// src holds `count` runs of `length` numbers one after another (the stored
// columns). A group is a run, or the numbers at one position of every run.
// Every thread starts from the first number of its group and takes a later
// one only if it is strictly better, and when the threads' results are merged
// the earlier of two equal numbers wins. So the result is the one a single
// pass from the start gives: for ties, for the sign of a zero, and for NaN
// (one in the first place stays, later ones are passed over).
//
// A long group is taken in parts of `chunk` numbers, one part for every block
// along the grid's second dimension. Each part leaves its result in out_value
// and where it found it in out_position, and the next pass works on those:
// `lookup` then holds the places the numbers of src came from. (Every part
// starts from the group's first number too, so the rule above still holds.)
static const unsigned extremeThreads = 256;  // the threads of a block

__device__ inline bool extremeWins(float x, int at, float best, int where, bool smallest) {
    return (smallest ? x < best : x > best) || (x == best && at < where);
}

// The groups are the runs. blockDim.x threads share a run (thread t takes the
// numbers t, t + blockDim.x, ... of its part) and a block has blockDim.y
// runs. The result of part p of run r goes to index r * parts + p.
__global__ void extremePerRunKernel(const float* src, const float* lookup, float* out_value, float* out_position,
                                    size_t length, size_t count, size_t chunk, bool smallest) {
    __shared__ float value[extremeThreads];
    __shared__ int where[extremeThreads];
    const unsigned share = blockDim.x;
    const size_t run = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const unsigned slot = threadIdx.y * share + threadIdx.x;
    float best = 0.0f;
    int at = 0;
    if (run < count) {
        const float* v = src + run * length;
        const size_t start = static_cast<size_t>(blockIdx.y) * chunk;
        const size_t stop = start + chunk < length ? start + chunk : length;
        best = v[0];
        for (size_t i = start + threadIdx.x; i < stop; i += share) {
            const float x = v[i];
            if (smallest ? x < best : x > best) {
                best = x;
                at = static_cast<int>(i);
            }
        }
    }
    value[slot] = best;
    where[slot] = at;
    __syncthreads();
    for (unsigned half = share >> 1; half > 0; half >>= 1) {
        if (threadIdx.x < half &&
            extremeWins(value[slot + half], where[slot + half], value[slot], where[slot], smallest)) {
            value[slot] = value[slot + half];
            where[slot] = where[slot + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && run < count) {
        const size_t to = run * gridDim.y + blockIdx.y;
        if (out_value) out_value[to] = value[slot];
        if (out_position)
            out_position[to] = lookup ? lookup[run * length + where[slot]] : static_cast<float>(where[slot]);
    }
}

// The groups lie across the runs, one for every position in a run. A block
// has blockDim.x neighbouring positions, and blockDim.y threads share a
// position (thread t takes the runs t, t + blockDim.y, ... of its part). The
// result of part p at position i goes to index p * length + i.
__global__ void extremeAcrossRunsKernel(const float* src, const float* lookup, float* out_value, float* out_position,
                                        size_t length, size_t count, size_t chunk, bool smallest) {
    __shared__ float value[extremeThreads];
    __shared__ int where[extremeThreads];
    const unsigned share = blockDim.y;
    const size_t pos = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned slot = threadIdx.x * share + threadIdx.y;
    float best = 0.0f;
    int at = 0;
    if (pos < length) {
        const size_t start = static_cast<size_t>(blockIdx.y) * chunk;
        const size_t stop = start + chunk < count ? start + chunk : count;
        best = src[pos];
        for (size_t j = start + threadIdx.y; j < stop; j += share) {
            const float x = src[j * length + pos];
            if (smallest ? x < best : x > best) {
                best = x;
                at = static_cast<int>(j);
            }
        }
    }
    value[slot] = best;
    where[slot] = at;
    __syncthreads();
    for (unsigned half = share >> 1; half > 0; half >>= 1) {
        if (threadIdx.y < half &&
            extremeWins(value[slot + half], where[slot + half], value[slot], where[slot], smallest)) {
            value[slot] = value[slot + half];
            where[slot] = where[slot + half];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && pos < length) {
        const size_t to = static_cast<size_t>(blockIdx.y) * length + pos;
        if (out_value) out_value[to] = value[slot];
        if (out_position)
            out_position[to] = lookup ? lookup[static_cast<size_t>(where[slot]) * length + pos]
                                      : static_cast<float>(where[slot]);
    }
}

// How many threads share one result (a power of two, at most `most`), so
// that each has at least `least` numbers to look at.
inline unsigned extremeShare(size_t each, unsigned most, unsigned least) {
    unsigned share = 1;
    while (share < most && static_cast<size_t>(share) * 2 * least <= each) share <<= 1;
    return share;
}

// The numbers of a group that one pass takes as a part. Across the runs 16
// threads share a part (they read 16 neighbouring positions together), along
// a run 256 do; a thread then has at most 64 numbers, and a few long groups
// still get many blocks. (Another pass costs little here; on ROCm, where a
// launch is dear, the parts are larger.) The parts are the grid's second
// dimension, which holds 65535.
inline size_t extremeChunk(size_t group, bool across) {
    size_t chunk = across ? 1024 : 16384;
    while ((group + chunk - 1) / chunk > 65535) chunk *= 2;
    return chunk;
}

// One pass over src (see above): the result of every part of every group.
static void extremePass(const float* src, const float* lookup, float* out_value, float* out_position,
                        size_t length, size_t count, size_t chunk, bool across, bool smallest) {
    const size_t group = across ? count : length, others = across ? length : count;
    const unsigned parts = (unsigned)((group + chunk - 1) / chunk);
    const size_t part = group < chunk ? group : chunk;
    if (across) {
        const unsigned share = extremeShare(part, 16, 16), together = extremeThreads / share;
        extremeAcrossRunsKernel<<<dim3((unsigned)((others + together - 1) / together), parts), dim3(together, share)>>>(
            src, lookup, out_value, out_position, length, count, chunk, smallest);
    } else {
        const unsigned share = extremeShare(part, extremeThreads, 8), together = extremeThreads / share;
        extremePerRunKernel<<<dim3((unsigned)((others + together - 1) / together), parts), dim3(share, together)>>>(
            src, lookup, out_value, out_position, length, count, chunk, smallest);
    }
    CudaErrorCheck(cudaGetLastError());
}

// The largest or smallest entry of every column (dim 0) or row (dim 1), or
// the position of the largest.
Matrix<CUDAfloat> axis_extreme(const Matrix<CUDAfloat>& M, int dim, AxisExtreme what) {
    if (dim != 0 && dim != 1) {
        throw std::invalid_argument("axis_extreme: dim must be 0 or 1");
    }
    // the groups are the stored columns, or else lie across them
    bool across = (dim == 0) == static_cast<bool>(M.transpose);
    const size_t results = across ? M.numrow : M.numcol;
    const size_t each = across ? M.numcol : M.numrow;
    Matrix<CUDAfloat> out = Matrix<CUDAfloat>::uninitialized("axis_extreme", dim == 0 ? 1 : results, dim == 0 ? results : 1, 0);
    if (results == 0) return out;
    if (each == 0) {
        // nothing to compare: what a comparison would start from
        const double inf = std::numeric_limits<double>::infinity();
        fill(out, what == AxisExtreme::Max ? -inf : what == AxisExtreme::Min ? inf : 0.0);
        return out;
    }
    if (M.numrow > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        M.numcol > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("axis_extreme: the matrix is too large");
    }
    const bool smallest = what == AxisExtreme::Min, position = what == AxisExtreme::ArgMax;
    size_t length = M.numrow, count = M.numcol;
    if (across && length == 1) {
        // one position: the runs are one long run
        across = false;
        length = count;
        count = 1;
    }
    const float* src = reinterpret_cast<const float*>(M.elements.get());
    const float* lookup = nullptr;
    float* const dst = reinterpret_cast<float*>(out.elements.get());
    // A long group is taken in parts: a pass leaves the result of every part
    // and where it was, and the next pass works on those.
    Matrix<CUDAfloat> values(nullptr), places(nullptr);
    size_t& group = across ? count : length;
    for (;;) {
        const size_t chunk = extremeChunk(group, across);
        const size_t parts = (group + chunk - 1) / chunk;
        if (parts == 1) {
            extremePass(src, lookup, position ? nullptr : dst, position ? dst : nullptr,
                        length, count, chunk, across, smallest);
            break;
        }
        Matrix<CUDAfloat> part_values = Matrix<CUDAfloat>::uninitialized("axis_extreme values", parts * results, 1, 0);
        Matrix<CUDAfloat> part_places = Matrix<CUDAfloat>::uninitialized("axis_extreme places", parts * results, 1, 0);
        extremePass(src, lookup, reinterpret_cast<float*>(part_values.elements.get()),
                    reinterpret_cast<float*>(part_places.elements.get()), length, count, chunk, across, smallest);
        values = std::move(part_values);
        places = std::move(part_places);
        src = reinterpret_cast<const float*>(values.elements.get());
        lookup = reinterpret_cast<const float*>(places.elements.get());
        group = parts;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sums along an axis, laid out like the two kernels above: the threads of a
// block share a group, each adds up its numbers, and the block adds those up
// in pairs; a long group is taken in parts whose sums another pass adds.
// (The sum used to be cublasSgemv with a vector of ones, which had to be
// allocated and filled for every call. The numbers are added in a different
// order now, so a sum can differ from the old one in its last digits.)

// The groups are the runs; the sum of part p of run r goes to index r * parts + p.
__global__ void sumPerRunKernel(const float* src, float* out, size_t length, size_t count, size_t chunk) {
    __shared__ float value[extremeThreads];
    const unsigned share = blockDim.x;
    const size_t run = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const unsigned slot = threadIdx.y * share + threadIdx.x;
    float total = 0.0f;
    if (run < count) {
        const float* v = src + run * length;
        const size_t start = static_cast<size_t>(blockIdx.y) * chunk;
        const size_t stop = start + chunk < length ? start + chunk : length;
        for (size_t i = start + threadIdx.x; i < stop; i += share) total += v[i];
    }
    value[slot] = total;
    __syncthreads();
    for (unsigned half = share >> 1; half > 0; half >>= 1) {
        if (threadIdx.x < half) value[slot] += value[slot + half];
        __syncthreads();
    }
    if (threadIdx.x == 0 && run < count) out[run * gridDim.y + blockIdx.y] = value[slot];
}

// The groups lie across the runs; the sum of part p at position i goes to index p * length + i.
__global__ void sumAcrossRunsKernel(const float* src, float* out, size_t length, size_t count, size_t chunk) {
    __shared__ float value[extremeThreads];
    const unsigned share = blockDim.y;
    const size_t pos = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned slot = threadIdx.x * share + threadIdx.y;
    float total = 0.0f;
    if (pos < length) {
        const size_t start = static_cast<size_t>(blockIdx.y) * chunk;
        const size_t stop = start + chunk < count ? start + chunk : count;
        for (size_t j = start + threadIdx.y; j < stop; j += share) total += src[j * length + pos];
    }
    value[slot] = total;
    __syncthreads();
    for (unsigned half = share >> 1; half > 0; half >>= 1) {
        if (threadIdx.y < half) value[slot] += value[slot + half];
        __syncthreads();
    }
    if (threadIdx.y == 0 && pos < length) out[static_cast<size_t>(blockIdx.y) * length + pos] = value[slot];
}

// One pass over src: the sum of every part of every group.
static void sumPass(const float* src, float* out, size_t length, size_t count, size_t chunk, bool across) {
    const size_t group = across ? count : length, others = across ? length : count;
    const unsigned parts = (unsigned)((group + chunk - 1) / chunk);
    const size_t part = group < chunk ? group : chunk;
    if (across) {
        const unsigned share = extremeShare(part, 16, 16), together = extremeThreads / share;
        sumAcrossRunsKernel<<<dim3((unsigned)((others + together - 1) / together), parts), dim3(together, share)>>>(
            src, out, length, count, chunk);
    } else {
        const unsigned share = extremeShare(part, extremeThreads, 8), together = extremeThreads / share;
        sumPerRunKernel<<<dim3((unsigned)((others + together - 1) / together), parts), dim3(share, together)>>>(
            src, out, length, count, chunk);
    }
    CudaErrorCheck(cudaGetLastError());
}

//similar to MATLAB's sum function, sum the matrix along the specified dimension
Matrix<CUDAfloat> sum(const Matrix<CUDAfloat>& M, int dim)
{
    // the groups are the stored columns, or else lie across them
    bool across = (dim == 0) == static_cast<bool>(M.transpose);
    const size_t results = across ? M.numrow : M.numcol;
    const size_t each = across ? M.numcol : M.numrow;
    // results x 1, seen as 1 x results for dim 0
    Matrix<CUDAfloat> sumM = Matrix<CUDAfloat>::uninitialized("sumM", results, 1, 0);
    if (dim == 0)
    {
        sumM.transpose = true;
    }
    if (results == 0) return sumM;
    if (each == 0)
    {
        fill(sumM, 0.0);
        return sumM;
    }
    size_t length = M.numrow, count = M.numcol;
    if (across && length == 1)
    {
        // one position: the runs are one long run
        across = false;
        length = count;
        count = 1;
    }
    const float* src = (const float*) M.elements.get();
    Matrix<CUDAfloat> partial(nullptr);
    size_t& group = across ? count : length;
    for (;;)
    {
        const size_t chunk = extremeChunk(group, across);
        const size_t parts = (group + chunk - 1) / chunk;
        if (parts == 1)
        {
            sumPass(src, (float*) sumM.elements.get(), length, count, chunk, across);
            break;
        }
        Matrix<CUDAfloat> sums = Matrix<CUDAfloat>::uninitialized("sum parts", parts * results, 1, 0);
        sumPass(src, (float*) sums.elements.get(), length, count, chunk, across);
        partial = std::move(sums);
        src = (const float*) partial.elements.get();
        group = parts;
    }
    return sumM;
}
ostream& operator <<(ostream& os, const Matrix<CUDAfloat>& M) {
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

curandStatus_t GPUSampler::stats = CURAND_STATUS_NOT_INITIALIZED;
curandGenerator_t GPUSampler::gen = NULL;


Matrix<CUDAfloat> Matrix<CUDAfloat>::randn(size_t m, size_t n) {
    STATIC_TIC;
    auto& gen = GPUSampler::gen;
    Matrix<CUDAfloat> M("randn", m, n);

    if (m * n % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (m * n + 1) * sizeof(float));
        curandStatus_t curand_stat = curandGenerateNormal(gen, p, m * n + 1, 0.0f, 1.0f);
        if (curand_stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("sample random nums failed, {}.", static_cast<int>(curand_stat));
            ERROR_OUT;
        }
        cudaError_t stat = cudaMemcpy((float *) M.elements.get(), p, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory copy failed, {}.", cudaGetErrorString(stat));
            ERROR_OUT;
        }
        cudaFree(p);
    }
    else {
        // generate normal 
        auto stat = curandGenerateNormal(gen, (float *) M.elements.get(), M.numcol * M.numrow, 0.0f, 1.0f);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", static_cast<int>(stat));
            ERROR_OUT;
        }
    }
    STATIC_TOC;
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::rand(size_t m, size_t n) {
    static Profiler profiler("rand gen"); profiler.start();
    auto& gen = GPUSampler::gen;
    Matrix<CUDAfloat> M("randn", m, n);

    if (m * n % 2 != 0) {
        float* p = NULL;
        cudaMalloc(&p, (m * n + 1) * sizeof(float));
        curandStatus_t curand_stat = curandGenerateUniform(gen, p, m * n + 1);
        if (curand_stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("sample random nums failed, {}.", static_cast<int>(curand_stat));
            ERROR_OUT;
        }
        cudaError_t stat = cudaMemcpy((float *) M.elements.get(), p, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
        if (stat != cudaSuccess) {
            LOG_ERROR("device memory copy failed, {}.", cudaGetErrorString(stat));
            ERROR_OUT;
        }
        cudaFree(p);
    }
    else {
        // generate normal 
        auto stat = curandGenerateUniform(gen, (float *) M.elements.get(), M.numcol * M.numrow);
        if (stat != CURAND_STATUS_SUCCESS) {
            LOG_ERROR("curand generate normal failed, {}.", static_cast<int>(stat));
            ERROR_OUT;
        }
    }
    profiler.end();
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::ones(size_t m, size_t n)
{
    STATIC_TIC;
    Matrix<CUDAfloat> M("ones", m, n); fill(M, 1.0f);
    STATIC_TOC;
    return M;
}

Matrix<CUDAfloat> Matrix<CUDAfloat>::zeros(size_t m, size_t n)
{
    Matrix<CUDAfloat> M("zeros", m, n);
    // cudaError_t stat = cudaMemset((float *) M.elements.get(), 0, M.numcol * M.numrow * sizeof(float));
    // if (stat != cudaSuccess) {
    //     LOG_ERROR("device memory set zero failed, {}.", cudaGetErrorString(stat));
    // }
    return M;
}
