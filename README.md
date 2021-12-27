# Juzhen (矩阵)
![GitHub branch checks state](https://img.shields.io/github/checks-status/anewgithubname/Juzhen/main?style=plastic)

Juzhen is a meta language for C++ matrix operations. It provides a higher level interface for lower-level numerical calculation software like [CBLAS](http://www.netlib.org/blas/), [LAPACK](http://www.netlib.org/lapack/), [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library) or [CUDA](https://en.wikipedia.org/wiki/CUDA). 

## Example
You can simply do matrix operations on CPU:
```c++
#include <iostream> 
#include "juzhen.hpp"
using namespace std;

int main(){ MemoryDeleter<float> md1; 
    Matrix<float> A = {"A", {{1,2,3},{4,5,6}}};
    cout << A << endl;
    Matrix<float> B = {"B", {{.1,.2},{.3,.4},{.5,.6}}};
    cout << B << endl << endl;

    cout << log(exp(A*B)+1.0f)/5.0f << endl;
}
```
or on GPU:
```c++
#include <iostream> 
#include "juzhen.hpp"
using namespace std;

int main(){ GPUMemoryDeleter md1; MemoryDeleter<float> md2;
    // cuBLAS initialization ...

    // suppose "handle" is cuBLAS handle.
    cuMatrix A(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
    cout << A << endl;
    cuMatrix B(handle, Matrix<float>("B",{{.1,.2},{.3,.4},{.5,.6}}));
    cout << B << endl << endl;

    cout << (log(exp(A*B)+1.0f)/5.0f) << endl;
    
    //free cuBLAS ...
}
```
They both prints out:
```
A 2 by 3
1 2 3 
4 5 6 
B 3 by 2
0.1 0.2 
0.3 0.4 
0.5 0.6 

logM 2 by 2
0.461017 0.571807 
0.981484 1.28033 
```
You can verify the result using MATLAB:
```matlab
>> A = [1,2,3;4,5,6];
>> B = [.1,.2;.3,.4;.5,.6];
>> (log(exp(A*B)+1.0)./5.0)

ans =

    0.4610    0.5718
    0.9815    1.2803

>> 
```
## Advanced Examples:
See more examples on:
1. [helloworld-cpu](examples/helloworld.cpp)
2. [helloworld-gpu](examples/helloworld_gpu.cpp)
3. [Binary Logistic Regression using a linear model](examples/logisticregression_simple.cpp).
4. Classifying MNIST digits using one hidden layer neural net (on [CPU](examples/logisticregression_MNIST.cpp)/[GPU](examples/logisticregression_MNIST_GPU.cpp)).

## Compile and Run Examples:
1. Helloworld CPU
    ```
    make helloworld
    bin/helloworld.out
    ```
2. Helloworld GPU
    ```
    make helloworld-gpu
    bin/helloworld-gpu.out
    ```
3. MNIST CPU
    ```
    make logi-cpu
    bin/logi.out
    ```
4. MNIST GPU
    ```
    make logi-gpu
    bin/logi-gpu.out
    ```
## Supported Platforms
- Linux (CPU/GPU)
- MacOS (CPU)
- Windows (CPU/GPU*), you will need to install Visual Studio 2019 to compile the code. 
## Passing by Reference
Matrices are always passed by reference. For example: 
```c++
Matrix<float> A = {"A",{{1,2},{3,4},{5,6}}};
auto B = A;  
B.zeros();
// now both A and B are zero matrices. 
```
If you want a copy of ```A``` to be stored in ```B```, do the following
```c++
Matrix<float> A = {"A",{{1,2},{3,4},{5,6}}};
auto B = A*1.0;  
B.zeros();
// A remains the old value. 
```
## Garbage Collection
The allocated memory spaces will not be automatically released, so later computations can directly claim those spaces without calling expensive memory allocation functions. To release memory, please remember to add a line at the begining of the scope: 
```
MemoryDeleter<T> md1; 
```
where ```T``` is the type of your matrix and ```GPUMemoryDeleter md1;``` if you are using GPU computation. 

## Known Issues
1. GPU computation on Windows is ~2.5 time slower than on Linux. I am not sure the cause of this. 
    - Tested on CUDA 11.5, Windows 11. 
    - Tested it on Native Windows and WSL2. Results are the same. 
2. Currently, Juzhen only supports single precision float point CBLAS/cuBLAS, although it is very easy to modify the source code and add supports for other types of data. 
## Benchmark on some CPUs/GPUs
Benchmark using MNIST example, time collected by the built-in profiling tool. 

![](benchmark.png)
