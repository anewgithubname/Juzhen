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
static id<MTLComputePipelineState> ax_bPipelineState = nil;
static id<MTLComputePipelineState> matrixaddPipelineState = nil;
static id<MTLComputePipelineState> matrixproductPipelineState = nil;
static id<MTLComputePipelineState> expPipelineState = nil;
static id<MTLComputePipelineState> logPipelineState = nil;
static id<MTLComputePipelineState> elemInvPipelineState = nil;

MPSMatrixRandomMTGP32* randomKernel = nil;
MPSMatrixRandomDistributionDescriptor* randDesc = nil;

static std::map<float*, id<MTLBuffer>> bufferMap;

void mpsInit() {
    device = MTLCreateSystemDefaultDevice();
    
    commandQueue = [device newCommandQueue];

    // Random number distribution descriptor (Normal distribution)
    randDesc =
        [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:0.0f standardDeviation:1.0f];
    
    randomKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                    destinationDataType:MPSDataTypeFloat32
                                                    seed:1234
                                                    distributionDescriptor:randDesc];

    NSError* error = nil;

    // Load the Metal shader source file
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    if(!defaultLibrary) {
        NSLog(@"Could not load default Metal library.");
        exit(1);
    }

    // Compile the zero kernel
    id<MTLFunction> zeroKernel = [defaultLibrary newFunctionWithName:@"zero_kernel"];
    id<MTLFunction> ax_bkernel = [defaultLibrary newFunctionWithName:@"ax_b_kernel"];
    id<MTLFunction> matrixaddKernel = [defaultLibrary newFunctionWithName:@"matrix_add"];
    id<MTLFunction> matrixproductKernel = [defaultLibrary newFunctionWithName:@"matrix_product"];
    id<MTLFunction> expKernel = [defaultLibrary newFunctionWithName:@"inplace_exp_kernel"];
    id<MTLFunction> logKernel = [defaultLibrary newFunctionWithName:@"inplace_log_kernel"];
    id<MTLFunction> elemInvKernel = [defaultLibrary newFunctionWithName:@"outplace_transpose"];

    // Create compute pipeline state
    zeroPipelineState = [device newComputePipelineStateWithFunction:zeroKernel error:&error];
    ax_bPipelineState = [device newComputePipelineStateWithFunction:ax_bkernel error:&error];
    matrixaddPipelineState = [device newComputePipelineStateWithFunction:matrixaddKernel error:&error];
    matrixproductPipelineState = [device newComputePipelineStateWithFunction:matrixproductKernel error:&error];
    expPipelineState = [device newComputePipelineStateWithFunction:expKernel error:&error];
    logPipelineState = [device newComputePipelineStateWithFunction:logKernel error:&error];
    elemInvPipelineState = [device newComputePipelineStateWithFunction:elemInvKernel error:&error];

}

void mpsDestroy() {
    bufferMap.clear();
    [randDesc release];
    [randomKernel release];
    [zeroPipelineState release];
    [ax_bPipelineState release];
    [matrixaddPipelineState release];
    [matrixproductPipelineState release];
    [expPipelineState release];
    [logPipelineState release];
    [elemInvPipelineState release];
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

void mpsAdd(const float* A, float* B, int rowA, int colA, bool transpose, float a, float b){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    id<MTLBuffer> bufferB = bufferMap[B];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:matrixaddPipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBytes:&rowA length:sizeof(int) atIndex:2];
    [encoder setBytes:&colA length:sizeof(int) atIndex:3];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:4];
    [encoder setBytes:&a length:sizeof(float) atIndex:5];
    [encoder setBytes:&b length:sizeof(float) atIndex:6];


    MTLSize gridSize = MTLSizeMake(colA, rowA, 1); // Threads match matrix elements
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1); // Adjust for performance
        
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
}

void mpsProduct(const float* A, float* B, int rowA, int colA, bool transpose){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    id<MTLBuffer> bufferB = bufferMap[B];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:matrixproductPipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBytes:&rowA length:sizeof(int) atIndex:2];
    [encoder setBytes:&colA length:sizeof(int) atIndex:3];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:4];

    MTLSize gridSize = MTLSizeMake(colA, rowA, 1); // Threads match matrix elements
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1); // Adjust for performance

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
}

void mpsAx_b(const float* x, float a, float b, float* y, int N) {
    id<MTLBuffer> bufferX = bufferMap[(float *) x];
    id<MTLBuffer> bufferY = bufferMap[y];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:ax_bPipelineState];
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferY offset:0 atIndex:1];
    [encoder setBytes:&a length:sizeof(float) atIndex:2];
    [encoder setBytes:&b length:sizeof(float) atIndex:3];
    
    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = ax_bPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    [commandBuffer commit];
}

void mpsRandn(float *A, int N){
    // Get GPU buffer associated with A
    id<MTLBuffer> bufferA = bufferMap[A];

    // Create a command buffer to schedule GPU commands
    commandBuffer = [commandQueue commandBuffer];

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

void mpsExp(float *M, int N){
    id<MTLBuffer> buffer = bufferMap[M];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:expPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&N length:sizeof(int) atIndex:1];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = expPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    [commandBuffer commit];
}

void mpsLog(float *M, int N){
    id<MTLBuffer> buffer = bufferMap[M];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:logPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&N length:sizeof(int) atIndex:1];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = logPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    [commandBuffer commit];
}

void mpsGemv(const float* A, const float* x, float* y, int rowA, int colA, bool transposeA) {
    commandBuffer = [commandQueue commandBuffer];

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) A]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];
    
    auto vecX = [[MPSVector alloc] initWithBuffer:bufferMap[(float *) x]
        descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:transposeA ? rowA : colA dataType:MPSDataTypeFloat32]];

    auto vecY = [[MPSVector alloc] initWithBuffer:bufferMap[y]
        descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:transposeA ? colA : rowA dataType:MPSDataTypeFloat32]];
        
    auto gemvKernel = [[MPSMatrixVectorMultiplication alloc] initWithDevice:device
        transpose:transposeA rows:transposeA ? colA : rowA columns:transposeA ? rowA : colA alpha:1.0f beta:0.0f];
    [gemvKernel encodeToCommandBuffer:commandBuffer inputMatrix:matA inputVector:vecX resultVector:vecY];

    [commandBuffer commit];

    [matA release]; [vecX release]; [vecY release]; [gemvKernel release];
}

void mpsTopk(const float* A, int* B, float *C, int rowA, int colA, int k){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    // id<MTLBuffer> bufferB = bufferMap[B];
    id<MTLBuffer> bufferB  = [device newBufferWithLength:rowA*k*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = bufferMap[C];

    commandBuffer = [commandQueue commandBuffer];
    
    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferA
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];
    
    auto matB = [[MPSMatrix alloc] initWithBuffer:bufferB
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:k rowBytes:k*sizeof(int) dataType:MPSDataTypeUInt32]];
    
    auto matC = [[MPSMatrix alloc] initWithBuffer:bufferC
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto topkKernel = [[MPSMatrixFindTopK alloc] initWithDevice:device numberOfTopKValues:k];
    [topkKernel encodeToCommandBuffer:commandBuffer inputMatrix:matA resultIndexMatrix:matB resultValueMatrix:matC];

    [commandBuffer commit];

    mpsSynchronize();
    memcpy(B, bufferB.contents, rowA*k*sizeof(int));

    [matA release]; [matB release]; [matC release]; [topkKernel release]; [bufferB release];
    
}

void mpsElemInv(float *M, int N, float l){
    id<MTLBuffer> buffer = bufferMap[M];

    commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:elemInvPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&l length:sizeof(float) atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = elemInvPipelineState.maxTotalThreadsPerThreadgroup;
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
