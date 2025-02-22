#ifndef JUZHEN_HPP
#define JUZHEN_HPP

int compute();

class Codeimp;

#include "core.hpp"
#include "matrix.hpp"
#include "operators.hpp"

class Code{
    std::unique_ptr<Codeimp> pimpl;
public:
    Code();
    ~Code();
    int run();
    int render();
};

typedef Matrix<float> M;

#ifndef CPU_ONLY

#include "cumatrix.cuh"
typedef Matrix<CUDAfloat> CM;
#endif

#endif