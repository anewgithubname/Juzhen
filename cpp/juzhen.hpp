#ifndef JUZHEN_HPP
#define JUZHEN_HPP

int compute();
//class script{
//    public: 
//    virtual void oninit(){};
//    virtual void compute() = 0;
//    virtual void onexit(){};
//};

#include "core.hpp"
#include "matrix.hpp"

typedef Matrix<float> M;

#ifndef CPU_ONLY

#include "cumatrix.cuh"
typedef Matrix<CUDAfloat> CM;
#endif

#endif