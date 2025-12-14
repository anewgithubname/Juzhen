#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <random>

static const int N = 10000;         // Matrix dimension
static const int NUM_ITER = 10;     // Number of multiplications

int main(int argc, const char * argv[])
{
    @autoreleasepool {
        // 1) Create the default MTLDevice
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device.\n";
            return -1;
        }

        // 2) Create a command queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Failed to create a Metal command queue.\n";
            return -1;
        }

        // We'll do 2*N^3 floating ops for each NxN multiply
        // (naive count: each result entry involves N multiplies + N-1 adds => ~2*N).
        double flopsPerMultiply = 2.0 * double(N) * double(N) * double(N);

        // 3) Allocate and initialize host memory for A, B, and C
        size_t totalCount = size_t(N) * size_t(N);
        std::vector<float> A(totalCount), B(totalCount), C(totalCount, 0.0f);

        // Fill A and B with random data in [0,1).
        std::mt19937 rng(0u);  // fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < totalCount; i++) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        // 4) Create MTLBuffers for A, B, and C
        size_t bytes = totalCount * sizeof(float);
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.data()
                                                   length:bytes
                                                  options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.data()
                                                   length:bytes
                                                  options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bufferC = [device newBufferWithBytes:C.data()
                                                   length:bytes
                                                  options:MTLResourceStorageModeManaged];

        // 5) Create MPSMatrixDescriptors
        //    rowBytes = sizeof(float) * N for a row-major NxN.
        MPSMatrixDescriptor *desc = [MPSMatrixDescriptor
                                     matrixDescriptorWithRows:N
                                     columns:N
                                     rowBytes:sizeof(float)*N
                                     dataType:MPSDataTypeFloat32];

        // Create MPSMatrix objects
        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];

        // 6) Create an MPSMatrixMultiplication kernel:
        //    C = alpha * (A x B) + beta * C
        //    We'll do alpha=1, beta=0 => C = A x B
        MPSMatrixMultiplication *gemmKernel = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:NO
            resultRows:N
            resultColumns:N
            interiorColumns:N
            alpha:1.0
            beta:0.0];

        // We'll store iteration times
        std::vector<double> times(NUM_ITER);

        // 7) Perform the matrix multiplication NUM_ITER times
        for (int iter = 0; iter < NUM_ITER; iter++) {

            // Optionally reset C on the CPU side if you'd like:
            std::fill(C.begin(), C.end(), 0.0f);
            // Copy that to GPU:
            memcpy(bufferC.contents, C.data(), bytes);
            [bufferC didModifyRange:NSMakeRange(0, bytes)];

            // Start timing
            CFAbsoluteTime startTime = CFAbsoluteTimeGetCurrent();

            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];

            // Encode the MPSMatrixMultiplication
            [gemmKernel encodeToCommandBuffer:cmdBuf
                                   leftMatrix:mA
                                  rightMatrix:mB
                                 resultMatrix:mC];

            // Commit and wait for GPU
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // End timing
            CFAbsoluteTime endTime = CFAbsoluteTimeGetCurrent();
            double elapsed = endTime - startTime;
            times[iter] = elapsed;

            // Compute approximate TFLOPs for this iteration
            double tflops = (flopsPerMultiply / elapsed) / 1.0e12;
            std::cout << "Iteration " << (iter + 1)
                      << " took " << elapsed << " s, ~"
                      << tflops << " TFLOPs\n";
        }

        // 8) Compute average time and TFLOPs
        double totalTime = 0.0;
        for (auto t : times) {
            totalTime += t;
        }
        double avgTime = totalTime / NUM_ITER;
        double avgTFLOPs = (flopsPerMultiply / avgTime) / 1.0e12;

        std::cout << "\nAverage time over " << NUM_ITER 
                  << " iterations: " << avgTime << " s\n";
        std::cout << "Approx. average performance: "
                  << avgTFLOPs << " TFLOPs\n";
    }
    return 0;
}
