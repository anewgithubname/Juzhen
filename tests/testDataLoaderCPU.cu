/**
 * @file testDataLoaderCPU.cu
 * @brief Test DataLoader with CPU backend
 */

#include "../ml/dataloader.hpp"
#include <iostream>

using namespace Juzhen;
using namespace std;

int compute() {
    cout << "Testing DataLoader on CPU..." << endl;
    
    int batchsize = 32;
    const string mnist_dir = string(PROJECT_DIR) + "/datasets/MNIST";
    
    cout << "Creating DataLoader..." << endl;
    DataLoader<float, int> loader(mnist_dir, "train", batchsize);
    cout << "DataLoader created"  << endl;
    
    cout << "Loading first batch..." << endl;
    auto [x, y] = loader.next_batch();
    cout << "Batch loaded: x=" << x.num_row() << "x" << x.num_col() << ", y=" << y.num_row() << "x" << y.num_col() << endl;
    
    cout << "Loading second batch..." << endl;
    auto [x2, y2] = loader.next_batch();
    cout << "Batch 2 loaded: x=" << x2.num_row() << "x" << x2.num_col() << ", y=" << y2.num_row() << "x" << y2.num_col() << endl;
    
    cout << "Test passed!" << endl;
    return 0;
}
