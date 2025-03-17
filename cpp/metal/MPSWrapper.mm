#include <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "MPSWrapper.h"
#include <iostream>
#include <map>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLCommandBuffer> commandBuffer = nil;
static id<MTLComputePipelineState> zeroPipelineState = nil;
static std::map<float*, id<MTLBuffer>> bufferMap;

void mpsInit() {
    device = MTLCreateSystemDefaultDevice();
    
    commandQueue = [device newCommandQueue];

    NSError* error = nil;

    // Load the Metal shader source file
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    if(!defaultLibrary) {
        NSLog(@"Could not load default Metal library.");
        exit(1);
    }

    // Compile the zero kernel
    id<MTLFunction> zeroKernel = [defaultLibrary newFunctionWithName:@"zero_kernel"];
    if (!zeroKernel) {
        NSLog(@"Failed to find zeroKernel function!");
        exit(1);
    }

    // Create compute pipeline state
    zeroPipelineState = [device newComputePipelineStateWithFunction:zeroKernel error:&error];
    if (!zeroPipelineState || error) {
        NSLog(@"Failed to create zeroPipelineState: %@", error);
        exit(1);
    }

}

void mpsDestroy() {
    bufferMap.clear();
    [commandQueue release];
    [device release];
    device = nil;
    commandQueue = nil;
}

float* mpsMalloc(size_t size) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buffer==nil) throw std::bad_alloc();
    float* ptr = (float*)buffer.contents;
    bufferMap[ptr] = buffer;
    return ptr;
}

void mpsFree(float* ptr) {
    auto it = bufferMap.find(ptr);
    if(it != bufferMap.end()) {
        std::cout << "Freeing buffer " << ptr << std::endl;
        [it->second release];
        bufferMap.erase(it);
    }
}
/**
 * MPSMatrixMultiplication
 * @param A: M x K
 * @param B: K x N
 * @param C: M x N
 */
void mpsGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transposeA, bool transposeB) {
    commandBuffer = [commandQueue commandBuffer];

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) A]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];
    
    auto matB = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) B]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowB columns:colB rowBytes:colB*sizeof(float) dataType:MPSDataTypeFloat32]];

    int resultRows = transposeA ? colA : rowA;
    int resultColumns = transposeB ? rowB : colB;
    int interiorColumns = transposeA ? rowA : colA;
    auto matC = [[MPSMatrix alloc] initWithBuffer:bufferMap[C]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:resultRows columns:resultColumns rowBytes:resultColumns*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto gemmKernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
        transposeLeft:transposeA transposeRight:transposeB resultRows:resultRows resultColumns:resultColumns interiorColumns:interiorColumns alpha:1.0f beta:0.0f];
    [gemmKernel encodeToCommandBuffer:commandBuffer leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    [commandBuffer commit];

    [matA release]; [matB release]; [matC release]; [gemmKernel release];
}

void mpsRandn(float *A, int N){
    // Get GPU buffer associated with A
    id<MTLBuffer> bufferA = bufferMap[A];

    // Create a command buffer to schedule GPU commands
    commandBuffer = [commandQueue commandBuffer];

    // Random number distribution descriptor (Normal distribution)
    MPSMatrixRandomDistributionDescriptor* randDesc =
        [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:0.0f standardDeviation:1.0f];

    // GPU-based high-quality random number generator (MTGP32)
    MPSMatrixRandomMTGP32* randomKernel = nil;
    if (!randomKernel) {
        randomKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                      destinationDataType:MPSDataTypeFloat32
                                                      seed:1234
                                                      distributionDescriptor:randDesc];
    }

    // Matrix descriptor to wrap buffer A
    MPSMatrixDescriptor* matrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1 columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:matrixDesc];

    // Encode random number generation into command buffer
    [randomKernel encodeToCommandBuffer:commandBuffer destinationMatrix:matA];
    [commandBuffer commit];
    // Cleanup
    [matA release];
    [randDesc release];
    [randomKernel release];
}

void mpsFill(float* A, int N, float val) {
    id<MTLBuffer> buffer = bufferMap[A];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:zeroPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&val length:sizeof(float) atIndex:1];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = zeroPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    [commandBuffer commit];
}

/* Synchronize the command buffer */
void mpsSynchronize() {
    [commandBuffer waitUntilCompleted];
    commandBuffer = nil;
}
