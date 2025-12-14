// main.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#define MATRIX_SIZE 10000
#define ITERATIONS 10

// CPU GEMM using Accelerate framework (cblas_sgemm) with timing and GFLOPS calculation.
void runCPUGEMM() {
    // Allocate matrices A, B, and C in row-major order.
    std::vector<float> A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> C(MATRIX_SIZE * MATRIX_SIZE, 0.0f);

    // Initialize A and B with random float values.
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Start the timer for CPU GEMM.
    auto start = std::chrono::high_resolution_clock::now();

    // Loop ITERATIONS times performing: C = A * B.
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE,
                    1.0f,
                    A.data(), MATRIX_SIZE,
                    B.data(), MATRIX_SIZE,
                    0.0f,
                    C.data(), MATRIX_SIZE);
    }
    
    // Stop the timer.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Calculate total floating-point operations: 2*MATRIX_SIZE^3 per iteration.
    double totalOps = ITERATIONS * 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
    double gflops = totalOps / elapsed.count() / 1e12;
    
    std::cout << "CPU GEMM completed in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "CPU TFLOPS: " << gflops << std::endl;
}

// GPU GEMM using Metal and MetalPerformanceShaders (MPSMatrixMultiplication) with timing and GFLOPS calculation.
void runGPUGEMM() {
    // Create the default Metal device.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"Metal is not supported on this device.");
        return;
    }
    
    // Create a command queue.
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Calculate the byte size for one matrix.
    NSUInteger matrixByteSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // Create Metal buffers for matrices A, B, and C.
    id<MTLBuffer> bufferA = [device newBufferWithLength:matrixByteSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:matrixByteSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:matrixByteSize options:MTLResourceStorageModeShared];
    
    // Fill buffers A and B with random data.
    float* ptrA = (float*)bufferA.contents;
    float* ptrB = (float*)bufferB.contents;
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        ptrA[i] = static_cast<float>(arc4random()) / UINT32_MAX;
        ptrB[i] = static_cast<float>(arc4random()) / UINT32_MAX;
    }
    
    // Create MPSMatrix descriptors for each matrix.
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:MATRIX_SIZE
                                                                           columns:MATRIX_SIZE
                                                                          rowBytes:MATRIX_SIZE * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:MATRIX_SIZE
                                                                           columns:MATRIX_SIZE
                                                                          rowBytes:MATRIX_SIZE * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:MATRIX_SIZE
                                                                           columns:MATRIX_SIZE
                                                                          rowBytes:MATRIX_SIZE * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
    
    // Create MPSMatrix objects for A, B, and C.
    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
    
    // Set up MPSMatrixMultiplication.
    // It computes: C = alpha * (A * B) + beta * C.
    // We set alpha = 1.0 and beta = 0.0.
    MPSMatrixMultiplication *gemm =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                             transposeLeft:false
                                            transposeRight:false
                                               resultRows:MATRIX_SIZE
                                            resultColumns:MATRIX_SIZE
                                          interiorColumns:MATRIX_SIZE
                                                    alpha:1.0
                                                     beta:0.0];
    
    // Start the timer for GPU GEMM.
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the GEMM operation ITERATIONS times.
    for (int iter = 0; iter < ITERATIONS; iter++) {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [gemm encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    // Stop the timer.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Calculate total floating-point operations and compute GFLOPS.
    double totalOps = ITERATIONS * 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
    double gflops = totalOps / elapsed.count() / 1e12;
    
    NSLog(@"GPU GEMM completed in %f seconds.", elapsed.count());
    NSLog(@"GPU TFLOPS: %f", gflops);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Starting CPU GEMM using Accelerate framework...");
        runCPUGEMM();
        
        NSLog(@"Starting GPU GEMM using MetalPerformanceShaders...");
        runGPUGEMM();
    }
    return 0;
}
