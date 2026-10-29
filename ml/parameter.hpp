#pragma once
#include "util.cuh"
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Juzhen {
// References borrow their layer's lifetime. Matrix objects, rather than buffer
// addresses, identify parameters, so checkpoint loading can replace storage.
template<class D> struct GradientBuffer {
    Matrix<D> value;
    bool ready = false;
};

template<class D> struct ParameterRef {
    std::string name;
    Matrix<D>* value;
    GradientBuffer<D>* gradient;
    adam_state<D>* state;
    bool* trainable;
    bool decay = true;
    bool* parent_trainable = nullptr;

    bool enabled() const {
        return (!trainable || *trainable) && (!parent_trainable || *parent_trainable);
    }
    const Matrix<D>& grad() const { return gradient->value; }
    bool has_grad() const { return gradient->ready; }
    void zero_grad() const {
        if (gradient->value.num_row() && gradient->value.num_col())
            gradient->value.zeros();
        gradient->ready = false;
    }
};

// No buffers or Adam copies are allocated merely by constructing a layer.
// Copying a layer does not copy registrations bound to the old object.
template<class D> class GradientStore {
    std::map<Matrix<D>*, std::unique_ptr<GradientBuffer<D>>> buffers_;

    // the gradient buffer for `parameter`, created on first use
    GradientBuffer<D>& slot(Matrix<D>& parameter) {
        auto& entry = buffers_[&parameter];
        if (!entry)
            entry = std::make_unique<GradientBuffer<D>>();
        return *entry;
    }

    static bool same_shape(const Matrix<D>& a, const Matrix<D>& b) {
        return a.num_row() == b.num_row() && a.num_col() == b.num_col();
    }

public:
    GradientStore() = default;
    GradientStore(const GradientStore&) {}
    GradientStore& operator=(const GradientStore& other) {
        if (this != &other)
            buffers_.clear();
        return *this;
    }

    ParameterRef<D> parameter(std::string name, Matrix<D>& value, adam_state<D>& state,
                              bool& trainable, bool decay = true) {
        return {std::move(name), &value, &slot(value), &state, &trainable, decay};
    }

    // accumulate `gradient` into the parameter's buffer
    void add(Matrix<D>& parameter, Matrix<D>&& gradient) {
        if (!same_shape(parameter, gradient))
            throw std::invalid_argument("parameter gradient shape mismatch");

        auto& s = slot(parameter);
        if (!s.ready) {
            // first gradient this step: take it over directly
            s.value = std::move(gradient);
            s.ready = true;
        } else {
            if (!same_shape(s.value, parameter))
                throw std::logic_error("parameter shape changed during accumulation");
            s.value += gradient;
        }
    }

    // the parameter's gradient buffer, zeroed if nothing has been accumulated yet
    Matrix<D>& buffer(Matrix<D>& parameter) {
        auto& s = slot(parameter);
        if (!s.ready) {
            if (!same_shape(s.value, parameter))
                s.value = Matrix<D>::zeros(parameter.num_row(), parameter.num_col());
            else
                s.value.zeros();
            s.ready = true;
        }
        if (!same_shape(s.value, parameter))
            throw std::logic_error("parameter shape changed during accumulation");
        return s.value;
    }
};
} // namespace Juzhen
