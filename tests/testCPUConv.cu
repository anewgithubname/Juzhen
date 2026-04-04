/**
 * @file testCPUConv.cu
 * @brief Test CPU ConvLayer with UNetScore-like architecture
 */

#include "../ml/layer.hpp"
#include <iostream>

using namespace Juzhen;
using namespace std;

#if defined(CUDA) || defined(APPLE_SILICON)
int compute() {
    cout << "testCPUConv requires a CPU-only build." << endl;
    return 0;
}
#else

int compute() {
    cout << "Testing CPU ConvLayer..." << endl;
    
    // Create layers like UNetScore does
    int batchsize = 2;  // Small batch for testing
    
    cout << "Creating enc1..." << endl;
    ConvLayer enc1(batchsize, 3, 32, 32, 16, 3, 3, 1, 1, true);
    cout << "enc1 created successfully" << endl;
    
    cout << "Creating input matrix..." << endl;
    Matrix<float> input("input", 3 * 32 * 32, batchsize);
    input = Matrix<float>::randn(3 * 32 * 32, batchsize) * 0.1f;
    cout << "Input matrix created: " << input.num_row() << " x " << input.num_col() << endl;
    
    cout << "Calling enc1.eval()..." << endl;
    enc1.eval(input);
    cout << "enc1.eval() completed" << endl;
    
    cout << "Getting output..." << endl;
    const auto& output = enc1.value();
    cout << "Output shape: " << output.num_row() << " x " << output.num_col() << endl;
    
    cout << "Test passed!" << endl;
    return 0;
}

#endif
