#pragma once
#include "parameter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#ifdef CUDA
namespace Juzhen::optimizer_detail {
// One tile belongs to one contiguous gradient buffer. Norm/scaling do not need
// logical transpose strides: both operations visit every physical element.
struct GradientTiles {
    float* data;
    size_t count;  // number of elements in this gradient
    size_t end;    // one past the last tile index owned by this gradient
};
constexpr size_t norm_tile_size = 4096;

// binary search: which gradient owns global tile number `tile`?
__device__ inline unsigned gradient_for_tile(const GradientTiles* gradients, unsigned count, size_t tile) {
    unsigned lo = 0, hi = count;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo) / 2;
        if (tile < gradients[mid].end)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// add up shared[0..255] across a 256-thread block, leaving the total in shared[0]
__device__ inline void block_sum(double* shared) {
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
}

// one block per tile: write the tile's sum of squares to partials[blockIdx.x]
static __global__ void gradient_norm_tiles(const GradientTiles* gradients, unsigned count, double* partials) {
    const unsigned p = gradient_for_tile(gradients, count, blockIdx.x);
    const auto g = gradients[p];
    const size_t first_tile = p ? gradients[p - 1].end : 0;
    const size_t start = (blockIdx.x - first_tile) * norm_tile_size;
    const size_t stop = min(g.count, start + norm_tile_size);

    double sum = 0;
    for (size_t i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const double x = g.data[i];
        sum += x * x;
    }

    __shared__ double shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    block_sum(shared);

    if (!threadIdx.x)
        partials[blockIdx.x] = shared[0];
}

// single block: add up all partial sums and take the square root
static __global__ void finish_gradient_norm(const double* partials, size_t count, double* result) {
    double sum = 0;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
        sum += partials[i];

    __shared__ double shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    block_sum(shared);

    if (!threadIdx.x)
        *result = sqrt(shared[0]);
}

// one block per tile: multiply every element of the tile by `scale`
static __global__ void scale_gradient_tiles(const GradientTiles* gradients, unsigned count, float scale) {
    const unsigned p = gradient_for_tile(gradients, count, blockIdx.x);
    const auto g = gradients[p];
    const size_t first_tile = p ? gradients[p - 1].end : 0;
    const size_t start = (blockIdx.x - first_tile) * norm_tile_size;
    const size_t stop = min(g.count, start + norm_tile_size);

    for (size_t i = start + threadIdx.x; i < stop; i += blockDim.x)
        g.data[i] *= scale;
}

// Scratch capacity grows only when necessary. cudaMemcpy is intentionally
// synchronous: clip_grad_norm must return a checked host norm. It also protects
// descriptors still used by a previous scaling kernel on the default stream.
// Optimizer copies may share this scratch; calls follow the library's existing
// single-thread/default-stream contract. No parameter addresses are cached.
class CudaNormScratch {
    GradientTiles* gradients_ = nullptr;
    double* partials_ = nullptr;
    double* result_ = nullptr;
    size_t gradient_capacity_ = 0;
    size_t tile_capacity_ = 0;

    void reserve(size_t gradients, size_t tiles) {
        if (gradients > gradient_capacity_) {
            if (gradients_)
                CudaErrorCheck(cudaFree(gradients_));
            CudaErrorCheck(cudaMalloc(&gradients_, gradients * sizeof(GradientTiles)));
            gradient_capacity_ = gradients;
        }
        if (tiles > tile_capacity_) {
            if (partials_)
                CudaErrorCheck(cudaFree(partials_));
            CudaErrorCheck(cudaMalloc(&partials_, tiles * sizeof(double)));
            tile_capacity_ = tiles;
        }
        if (!result_)
            CudaErrorCheck(cudaMalloc(&result_, sizeof(double)));
    }

    static double sequential(const std::vector<ParameterRef<CUDAfloat>>& parameters, double max_norm) {
        // Preserve the established sequential behavior for unusual callers that
        // register the same gradient allocation under distinct parameters.
        double norm = 0;
        for (const auto& p : parameters)
            if (p.enabled() && p.has_grad())
                norm = std::hypot(norm, double(p.grad().norm()));
        if (!std::isfinite(norm))
            throw std::runtime_error("non-finite gradient norm");

        const double scale = std::min(1., max_norm / (norm + 1e-6));
        if (scale < 1)
            for (const auto& p : parameters)
                if (p.enabled() && p.has_grad())
                    p.gradient->value.scale(scale);
        return norm;
    }

public:
    CudaNormScratch() = default;
    CudaNormScratch(const CudaNormScratch&) = delete;
    CudaNormScratch& operator=(const CudaNormScratch&) = delete;
    ~CudaNormScratch() {
        // Free device storage before returning host ownership; cudaFree waits
        // for pending uses, including the asynchronous final scale operation.
        if (gradients_)
            CudaErrorCheck(cudaFree(gradients_));
        if (partials_)
            CudaErrorCheck(cudaFree(partials_));
        if (result_)
            CudaErrorCheck(cudaFree(result_));
    }

    double clip(const std::vector<ParameterRef<CUDAfloat>>& parameters, double max_norm) {
        // split every gradient into tiles of norm_tile_size elements
        std::vector<GradientTiles> gradients;
        gradients.reserve(parameters.size());
        std::unordered_set<const float*> unique;
        size_t tiles = 0;
        for (const auto& p : parameters) {
            if (!p.enabled() || !p.has_grad())
                continue;

            auto& g = p.gradient->value;
            auto* data = const_cast<float*>(reinterpret_cast<const float*>(g.data()));
            if (!unique.insert(data).second)
                return sequential(parameters, max_norm);

            const size_t count = g.num_row() * g.num_col();
            const size_t needed = (count - 1) / norm_tile_size + 1;
            if (needed > size_t(std::numeric_limits<int>::max()) - tiles)
                throw std::length_error("gradient reduction exceeds CUDA grid limit");

            tiles += needed;
            gradients.push_back({data, count, tiles});
        }
        if (gradients.empty())
            return 0.;

        // compute the global norm on the device
        reserve(gradients.size(), tiles);
        CudaErrorCheck(cudaMemcpy(gradients_, gradients.data(),
                                  gradients.size() * sizeof(GradientTiles), cudaMemcpyHostToDevice));
        gradient_norm_tiles<<<unsigned(tiles), 256>>>(gradients_, unsigned(gradients.size()), partials_);
        CudaErrorCheck(cudaGetLastError());
        finish_gradient_norm<<<1, 256>>>(partials_, tiles, result_);
        CudaErrorCheck(cudaGetLastError());

        double norm = 0;
        CudaErrorCheck(cudaMemcpy(&norm, result_, sizeof(double), cudaMemcpyDeviceToHost));
        if (!std::isfinite(norm))
            throw std::runtime_error("non-finite gradient norm");

        // rescale only if the norm is over the limit
        const double scale = std::min(1., max_norm / (norm + 1e-6));
        if (scale < 1) {
            scale_gradient_tiles<<<unsigned(tiles), 256>>>(gradients_, unsigned(gradients.size()), float(scale));
            CudaErrorCheck(cudaGetLastError());
        }
        return norm;
    }
};
} // namespace Juzhen::optimizer_detail
#endif
