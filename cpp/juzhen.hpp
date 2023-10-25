#ifndef JUZHEN_HPP
#define JUZHEN_HPP

int compute();

#include "core.hpp"
#include "matrix.hpp"
#include "operators.hpp"

typedef Matrix<float> M;

#ifndef CPU_ONLY

#include "cumatrix.cuh"
typedef Matrix<CUDAfloat> CM;
#endif

#endif