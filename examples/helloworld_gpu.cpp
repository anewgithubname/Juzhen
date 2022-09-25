#include <iostream> 
#include "../cpp/juzhen.hpp"
using namespace std;

int main(){ GPUMemoryDeleter md1; MemoryDeleter<float> md2;
    // cuda initialization
#ifndef CPU_ONLY
    GPUMemoryDeleter gpumd;
    GPUSampler sampler(1);
    cublasCreate(&cuMatrix::global_handle);
    LOG_INFO("CuBLAS INITIALIZED!");
#endif

    //do stuff...
    cuMatrix A(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
    cout << A << endl;
    cuMatrix B(handle, Matrix<float>("B",{{.1,.2},{.3,.4},{.5,.6}}));
    cout << B << endl << endl;

    cout << (log(exp(A*B)+1.0f)/5.0f) << endl;
    
    //free cuda
#ifndef CPU_ONLY
    cublasDestroy(cuMatrix::global_handle);
    LOG_INFO("CuBLAS FREED!");
#endif
}