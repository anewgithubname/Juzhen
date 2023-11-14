#include <iostream>
using namespace std;

#include "cpp/juzhen.hpp"

int compute(){

    Matrix<float> A = Matrix<float>::randn(500, 1000);

    {Profiler p; //start the profiler
        for (int i = 0; i < 1000; i++)
        {
            auto &&C = A * A.T();
        }
    }// profiler will automatically stop and print out elapsed time when the current scope exits. 

    return 0;
}