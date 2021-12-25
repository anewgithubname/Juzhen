# Juzhen (矩阵)

Juzhen is a meta language for C++ matrix operations. It provides a higher level interface for lower-level numerical calculation software like [CBLAS](http://www.netlib.org/blas/), [LAPACK](http://www.netlib.org/lapack/), [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library) or [CUDA](https://en.wikipedia.org/wiki/CUDA). 

## Example
You can simply do matrix operations on CPU:
```c++
#include <iostream> 
#include "juzhen.hpp"
using namespace std;

int main(){
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

int main(){
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
B = [.1,.2;.3,.4;.5,.6];
>> (log(exp(A*B)+1.0)./5.0)

ans =

    0.4610    0.5718
    0.9815    1.2803

>> 
```
## Advanced Examples:
See more examples on:
1. [helloworld-cpu](example/helloworld.cpp)
2. [helloworld-gpu](example/helloworld_gpu.cpp)
3. [Binary Logistic Regression using a linear model](example/logisticregression_simple.cpp).
4. Classifying MNIST digits using one hidden layer neural net (on [CPU](example/logisticregression_MNIST.cpp)/[GPU](example/logisticregression_MNIST_GPU.cpp)).

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