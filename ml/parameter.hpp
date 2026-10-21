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
        if(gradient->value.num_row() && gradient->value.num_col()) gradient->value.zeros();
        gradient->ready = false;
    }
};

// No buffers or Adam copies are allocated merely by constructing a layer.
// Copying a legacy layer does not copy registrations bound to the old object.
template<class D> class GradientStore {
    std::map<Matrix<D>*,std::unique_ptr<GradientBuffer<D>>> buffers_;
    GradientBuffer<D>& slot(Matrix<D>& parameter) {
        auto& entry=buffers_[&parameter];
        if(!entry)entry=std::make_unique<GradientBuffer<D>>();
        return *entry;
    }
public:
    GradientStore()=default;
    GradientStore(const GradientStore&) {}
    GradientStore& operator=(const GradientStore& other) {
        if(this!=&other)buffers_.clear();return *this;
    }

    ParameterRef<D> parameter(std::string name,Matrix<D>& value,adam_state<D>& state,
                              bool& trainable,bool decay=true) {
        return {std::move(name),&value,&slot(value),&state,&trainable,decay};
    }
    void add(Matrix<D>& parameter,Matrix<D>&& gradient) {
        if(parameter.num_row()!=gradient.num_row() || parameter.num_col()!=gradient.num_col())
            throw std::invalid_argument("parameter gradient shape mismatch");
        auto& s=slot(parameter);
        if(!s.ready) { s.value=std::move(gradient);s.ready=true; }
        else {
            if(s.value.num_row()!=parameter.num_row() || s.value.num_col()!=parameter.num_col())
                throw std::logic_error("parameter shape changed during accumulation");
            s.value+=gradient;
        }
    }
    Matrix<D>& buffer(Matrix<D>& parameter) {
        auto& s=slot(parameter);
        if(!s.ready) {
            if(s.value.num_row()!=parameter.num_row() || s.value.num_col()!=parameter.num_col())
                s.value=Matrix<D>::zeros(parameter.num_row(),parameter.num_col());
            else s.value.zeros();
            s.ready=true;
        }
        if(s.value.num_row()!=parameter.num_row() || s.value.num_col()!=parameter.num_col())
            throw std::logic_error("parameter shape changed during accumulation");
        return s.value;
    }
};
} // namespace Juzhen
