#pragma once
// Token layers.
//
//   EmbeddingLayer<D>(vocab, dim)           ids -> vectors; the input layer of a network
//   PositionEmbeddingLayer<D>(seq_len, dim) adds a learned vector per position
//   TiedLinearLayer<D>(embedding)           output layer that uses the embedding table as its weight
//
// All three are ordinary layers: Sequential, parameters(), AdamW and
// save_checkpoint treat them like any other layer.
//
// Backends: CPU loops; CUDA and ROCm kernels from one source; Metal uses the
// device gather and a host loop over shared memory. A ROCm translation unit
// that is not compiled as HIP has no kernels and goes through the host.
#include "ids.hpp"
#include "layer.hpp"
#include <algorithm>
#include <limits>
#include <string>

namespace Juzhen {

namespace token_detail {
#ifdef JZ_ID_KERNELS
// out (dim x N, plain) column j = table row ids[j]. The table is vocab x dim;
// `token_major` says its storage keeps each token's vector together (the
// transpose flag is set), otherwise each feature's column is together.
static __global__ void embedding_lookup_kernel(float* out, const float* table, const int32_t* ids, size_t dim,
                                               size_t vocab, size_t count, bool token_major) {
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += size_t(blockDim.x) * gridDim.x) {
        const size_t row = k % dim, id = size_t(ids[k / dim]);
        out[k] = table[token_major ? id * dim + row : row * vocab + id];
    }
}

// Scatter-add without atomics: the host has grouped the tokens by id
// (order[starts[u] .. starts[u+1]) are the tokens of id uids[u], in token
// order), so every gradient cell has one writer and the sums are taken in a
// fixed order: bitwise repeatable, and equal to the CPU loop.
static __global__ void embedding_scatter_kernel(float* grad, const float* upstream, const int32_t* uids,
                                                const int32_t* starts, const int32_t* order, size_t dim,
                                                size_t tokens, size_t unique, bool upstream_flagged, size_t vocab,
                                                bool grad_token_major) {
    const size_t count = dim * unique;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += size_t(blockDim.x) * gridDim.x) {
        const size_t row = k % dim, u = k / dim, id = size_t(uids[u]);
        const size_t cell = grad_token_major ? id * dim + row : row * vocab + id;
        float sum = grad[cell];
        for (int32_t p = starts[u]; p < starts[u + 1]; ++p) {
            const size_t token = size_t(order[p]);
            sum += upstream[upstream_flagged ? row * tokens + token : token * dim + row];
        }
        grad[cell] = sum;
    }
}
#endif
}  // namespace token_detail

// Looks up one row of a table per integer id.
//
// The table `weight` is vocab x dim: one row per token, the shape of the
// weight of an output layer that scores tokens (which is what TiedLinear
// uses it as). eval_ids(ids) gives dim x N, column j being row ids[j].
template <class D>
class EmbeddingLayer : public Layer<D> {
    size_t vocab_, dim_;
    std::vector<int> ids_;  // of the last eval
    Matrix<D> device_ids_;  // the same on the device, as bit patterns (GPU kernels only)
    bool has_forward_ = false;

    static size_t positive(size_t n, const char* what) {
        if (!n || n > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument(std::string("Embedding: ") + what + " must be between 1 and 2147483647");
        return n;
    }

public:
    EmbeddingLayer(size_t vocab, size_t dim, Init init = Init::normal(0.02f))
        : Layer<D>(size_t(0)), vocab_(positive(vocab, "vocab")), dim_(positive(dim, "dim")), device_ids_(nullptr) {
        // a lookup has no fan-in: the automatic rule is N(0, 0.02^2)
        const Init rule = init.rule == Init::Rule::automatic ? Init::normal(0.02f) : init;
        this->weights = rule.template weights<D>(vocab_, dim_, double(dim_), false);
        this->adamW = adam_state<D>(0.0001, vocab_, dim_);
    }

    size_t vocab() const { return vocab_; }
    size_t dim() const { return dim_; }

    bool takes_ids() const override { return true; }
    bool supports_backward() const override { return true; }

    std::vector<ParameterRef<D>> parameters() override {
        return {this->parameter_gradients.parameter("weight", this->weights, this->adamW, this->need_update)};
    }
    std::vector<std::pair<std::string, Matrix<D>*>> checkpoint_parameters() override {
        return {{"weight", &this->weights}};
    }
    std::vector<std::pair<std::string, adam_state<D>*>> checkpoint_optimizers() override {
        return {{"adam_weight", &this->adamW}};
    }

    // a float matrix is not ids
    void eval(const Matrix<D>&) override {
        throw std::invalid_argument("Embedding takes integer ids (a Matrix<int>), not a matrix of numbers");
    }
    Matrix<D> backward(const Matrix<D>&, Matrix<D>&&) override {
        throw std::invalid_argument("Embedding takes integer ids (a Matrix<int>), not a matrix of numbers");
    }
    Matrix<D> grad(const Matrix<D>&) const override {
        throw std::logic_error("integer ids have no derivative");
    }
    Matrix<D> jvp(const Matrix<D>&, const Matrix<D>&) override {
        throw std::logic_error("integer ids cannot move along a tangent: jvp is not defined for an Embedding");
    }

    void eval_ids(const Matrix<int>& ids) override {
        has_forward_ = false;
        if (this->weights.num_row() != vocab_ || this->weights.num_col() != dim_)
            throw std::invalid_argument("Embedding: the table is " + std::to_string(this->weights.num_row()) + " x " +
                                        std::to_string(this->weights.num_col()) + " but should be " +
                                        std::to_string(vocab_) + " x " + std::to_string(dim_));
        auto list = token_detail::flat(ids);
        if (list.empty())
            throw std::invalid_argument("Embedding: no ids were given");
        token_detail::check_range(list, vocab_, "Embedding: token id");

        Matrix<D> output("embedding", dim_, list.size());
        if constexpr (std::is_same_v<D, float>) {
            // one feature at a time: column i of the table is read at the ids
            for (size_t i = 0; i < dim_; ++i)
                for (size_t j = 0; j < list.size(); ++j)
                    output.elem(i, j) = this->weights.elem(size_t(list[j]), i);
        } else if constexpr (token_detail::on_device<D>) {
#ifdef JZ_ID_KERNELS
            using namespace token_detail;
            device_ids_ = device_words<D>(list);
            const size_t count = dim_ * list.size();
            embedding_lookup_kernel<<<blocks(count), 256>>>(floats(output), floats(this->weights), words(device_ids_),
                                                           dim_, vocab_, count, this->weights.get_transpose());
            launched();
#endif
        } else {
#ifdef APPLE_SILICON
            if constexpr (std::is_same_v<D, MPSfloat>) {
                // the existing device gather (mpsGather): columns ids of the
                // dim x vocab view of the table, stored plain
                output = this->weights.T().columns(idxlist(list.begin(), list.end()));
                this->val = std::move(output);
                ids_ = std::move(list);
                has_forward_ = true;
                return;
            }
#endif
            // no kernels in this translation unit: a host round trip
            const Matrix<float> table = this->weights.to_host();
            Matrix<float> host("embedding", dim_, list.size());
            for (size_t i = 0; i < dim_; ++i)
                for (size_t j = 0; j < list.size(); ++j)
                    host.elem(i, j) = table.elem(size_t(list[j]), i);
            output = Matrix<D>(host);
        }
        this->val = std::move(output);
        ids_ = std::move(list);
        has_forward_ = true;
    }

    // Adds upstream column j to the gradient of table row ids[j] (repeated
    // ids add up) and returns upstream: the gradient with respect to this
    // layer's output, since ids have none.
    Matrix<D> backward_ids(const Matrix<int>& ids, Matrix<D>&& upstream) override {
        if (!has_forward_)
            throw std::logic_error("Embedding backward needs the eval of the same ids first");
        if (ids.num_row() * ids.num_col() != ids_.size())
            throw std::logic_error("Embedding backward: these are not the ids of the last eval");
        if (upstream.num_row() != dim_ || upstream.num_col() != ids_.size())
            throw std::invalid_argument("Embedding backward: the gradient is " + std::to_string(upstream.num_row()) +
                                        " x " + std::to_string(upstream.num_col()) + " but the output was " +
                                        std::to_string(dim_) + " x " + std::to_string(ids_.size()));
        if (!this->need_update)
            return std::move(upstream);

        // the accumulated gradient, zeroed if this is the first contribution
        // of the step (a tied output layer may already have added its part)
        Matrix<D>& gradient = this->parameter_gradients.buffer(this->weights);
        if constexpr (std::is_same_v<D, float>) {
            // fixed order, token after token: the reference every backend matches
            for (size_t j = 0; j < ids_.size(); ++j)
                for (size_t i = 0; i < dim_; ++i)
                    gradient.elem(size_t(ids_[j]), i) += upstream.elem(i, j);
        } else if constexpr (token_detail::on_device<D>) {
#ifdef JZ_ID_KERNELS
            using namespace token_detail;
            // group the tokens by id on the host: a stable sort keeps token order inside a group
            const size_t tokens = ids_.size();
            std::vector<int> order(tokens), uids, starts;
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ids_[a] < ids_[b]; });
            for (size_t p = 0; p < tokens; ++p)
                if (p == 0 || ids_[order[p]] != ids_[order[p - 1]]) {
                    uids.push_back(ids_[order[p]]);
                    starts.push_back(int(p));
                }
            starts.push_back(int(tokens));
            const Matrix<D> d_order = device_words<D>(order), d_uids = device_words<D>(uids),
                            d_starts = device_words<D>(starts);
            const size_t count = dim_ * uids.size();
            embedding_scatter_kernel<<<blocks(count), 256>>>(
                floats(gradient), floats(upstream), words(d_uids), words(d_starts), words(d_order), dim_, tokens,
                uids.size(), upstream.get_transpose(), vocab_, gradient.get_transpose());
            launched();
#endif
        } else {
#ifdef APPLE_SILICON
            if constexpr (std::is_same_v<D, MPSfloat>) {
                // Metal storage is shared with the host: after one
                // synchronization the host adds in the same fixed order
                mpsSynchronize();
                for (size_t j = 0; j < ids_.size(); ++j)
                    for (size_t i = 0; i < dim_; ++i) {
                        MPSfloat& cell = gradient.elem(size_t(ids_[j]), i);
                        cell = float(cell) + float(upstream.elem(i, j));
                    }
                return std::move(upstream);
            }
#endif
            // no kernels in this translation unit: a host round trip, in the same fixed order
            const Matrix<float> up = upstream.to_host();
            Matrix<float> host = Matrix<float>::zeros(vocab_, dim_);
            for (size_t j = 0; j < ids_.size(); ++j)
                for (size_t i = 0; i < dim_; ++i)
                    host.elem(size_t(ids_[j]), i) += up.elem(i, j);
            gradient += Matrix<D>(host);
        }
        return std::move(upstream);
    }

    // for TiedLinear: add to the gradient of the shared table
    void accumulate(Matrix<D>&& gradient) {
        if (this->need_update)
            this->parameter_gradients.add(this->weights, std::move(gradient));
    }
};

// Adds a learned vector to every token according to its position in its
// sequence: column n of the input is at position n % seq_len.
//
// The table `weight` is seq_len x dim. Any number of sequences is accepted
// (the number of columns must be a multiple of seq_len). Built from matrix
// products with a constant 0/1 matrix, so it needs no kernel on any backend.
template <class D>
class PositionEmbeddingLayer : public Layer<D> {
    size_t seq_len_, dim_;
    Matrix<D> place_;  // seq_len x N, 1 where column n is at position n % seq_len

    void check(const Matrix<D>& x) {
        if (x.num_row() != dim_ || !x.num_col() || x.num_col() % seq_len_)
            throw std::invalid_argument("PositionEmbedding: the input is " + std::to_string(x.num_row()) + " x " +
                                        std::to_string(x.num_col()) + " but should be " + std::to_string(dim_) +
                                        " x a multiple of seq_len = " + std::to_string(seq_len_));
        if (place_.num_col() != x.num_col()) {
            Matrix<float> host = Matrix<float>::zeros(seq_len_, x.num_col());
            for (size_t n = 0; n < x.num_col(); ++n)
                host.elem(n % seq_len_, n) = 1;
            if constexpr (std::is_same_v<D, float>)
                place_ = std::move(host);
            else
                place_ = Matrix<D>(host);
        }
    }

public:
    PositionEmbeddingLayer(size_t seq_len, size_t dim, Init init = Init::normal(0.02f))
        : Layer<D>(size_t(0)), seq_len_(seq_len), dim_(dim), place_(nullptr) {
        if (!seq_len || !dim)
            throw std::invalid_argument("PositionEmbedding: seq_len and dim must be positive");
        const Init rule = init.rule == Init::Rule::automatic ? Init::normal(0.02f) : init;
        this->weights = rule.template weights<D>(seq_len_, dim_, double(dim_), false);
        this->adamW = adam_state<D>(0.0001, seq_len_, dim_);
    }

    size_t seq_len() const { return seq_len_; }
    bool supports_backward() const override { return true; }

    std::vector<ParameterRef<D>> parameters() override {
        return {this->parameter_gradients.parameter("weight", this->weights, this->adamW, this->need_update)};
    }
    std::vector<std::pair<std::string, Matrix<D>*>> checkpoint_parameters() override {
        return {{"weight", &this->weights}};
    }
    std::vector<std::pair<std::string, adam_state<D>*>> checkpoint_optimizers() override {
        return {{"adam_weight", &this->adamW}};
    }

    void eval(const Matrix<D>& x) override {
        check(x);
        // (dim x seq_len)(seq_len x N): the table row of each column's position
        this->val = x + this->weights.T() * place_;
    }

    // d(output)/d(input) is the identity; the table row of position t gets
    // the sum of the upstream columns at that position
    Matrix<D> backward(const Matrix<D>& x, Matrix<D>&& upstream) override {
        check(x);
        if (upstream.num_row() != dim_ || upstream.num_col() != x.num_col())
            throw std::invalid_argument("PositionEmbedding backward shape mismatch");
        if (this->need_update)
            this->parameter_gradients.add(this->weights, place_ * upstream.T());  // seq_len x dim
        return std::move(upstream);
    }

    Matrix<D> grad(const Matrix<D>& x) const override { return Matrix<D>::ones(dim_, x.num_col()); }
    Matrix<D> jvp(const Matrix<D>&, const Matrix<D>& tangent) override { return Matrix<D>(tangent); }
};

// The output layer of a language model with tied weights: logits = E x,
// where E (vocab x dim) is the table of `embedding`. It has no weight and no
// bias of its own, so it registers no parameter and writes nothing to a
// checkpoint: the Embedding, which must be in the same network, does both.
template <class D>
class TiedLinearLayer : public Layer<D> {
    EmbeddingLayer<D>& embedding_;

    void check(const Matrix<D>& x, const char* what) const {
        if (x.num_row() != embedding_.dim())
            throw std::invalid_argument(std::string("TiedLinear ") + what + ": the input has " +
                                        std::to_string(x.num_row()) + " rows but the embedding has dim " +
                                        std::to_string(embedding_.dim()));
    }

public:
    explicit TiedLinearLayer(EmbeddingLayer<D>& embedding) : Layer<D>(size_t(0)), embedding_(embedding) {}

    const Layer<D>* shares_with() const override { return &embedding_; }
    bool supports_backward() const override { return true; }
    std::vector<ParameterRef<D>> parameters() override { return {}; }
    std::vector<std::pair<std::string, Matrix<D>*>> checkpoint_parameters() override { return {}; }
    std::vector<std::pair<std::string, adam_state<D>*>> checkpoint_optimizers() override { return {}; }

    void eval(const Matrix<D>& x) override {
        check(x, "eval");
        this->val = embedding_.W() * x;  // vocab x N
    }
    Matrix<D> backward(const Matrix<D>& x, Matrix<D>&& dy) override {
        check(x, "backward");
        if (dy.num_row() != embedding_.vocab() || dy.num_col() != x.num_col())
            throw std::invalid_argument("TiedLinear backward shape mismatch");
        auto dx = embedding_.W().T() * dy;
        if (this->need_update)
            embedding_.accumulate(dy * x.T());  // vocab x dim
        return dx;
    }
    Matrix<D> grad(const Matrix<D>& x) const override { return Matrix<D>::ones(embedding_.vocab(), x.num_col()); }
    Matrix<D> jvp(const Matrix<D>& x, const Matrix<D>& tangent) override {
        check(x, "jvp");
        return embedding_.W() * tangent;
    }
};

}  // namespace Juzhen
