#pragma once
// How a layer draws its initial weights.
#include "../cpp/juzhen.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace Juzhen {

// The starting values of a weight matrix. Biases always start at zero.
//
//   Init()            the layer's own rule: N(0, 2/fan_in) in front of a ReLU,
//                     N(0, 1/fan_in) otherwise
//   Init::normal(s)   N(0, s^2); s = 0 gives zeros and draws nothing
//
// fan_in is the number of inputs that feed one output (the columns of a dense
// weight, C_in*kH*kW for a convolution).
struct Init {
    enum class Rule { automatic, normal };
    Rule rule = Rule::automatic;
    float std = 0;

    static Init normal(float std) {
        if (!std::isfinite(std) || std < 0)
            throw std::invalid_argument("Init::normal: the standard deviation must be 0 or more, got " +
                                        std::to_string(std));
        return Init{Rule::normal, std};
    }

    // the standard deviation of a weight with this fan-in; `relu` tells the
    // automatic rule whether a ReLU follows
    float scale(double fan_in, bool relu) const {
        if (rule == Rule::normal)
            return std;
        return float(std::sqrt((relu ? 2.0 : 1.0) / fan_in));
    }

    // a rows x cols weight; one randn(rows, cols) draw unless the scale is zero
    template <class D>
    Matrix<D> weights(size_t rows, size_t cols, double fan_in, bool relu) const {
        const float s = scale(fan_in, relu);
        if (s == 0)
            return Matrix<D>::zeros(rows, cols);
        return Matrix<D>::randn(rows, cols) * s;
    }
};

}  // namespace Juzhen
