#include <iostream>
using namespace std;

#include "cpp/juzhen.hpp"

int main(){MemoryDeleter<float> md;

    Matrix<float> A("A", 500, 1000);
    A.randn();

    {Profiler p; //start the profiler
        for (int i = 0; i < 1000; i++)
        {
            auto &&C = A * A.T();
        }
    }

}
