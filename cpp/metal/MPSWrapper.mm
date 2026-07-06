#include <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "MPSWrapper.h"
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> zeroPipelineState = nil;
static id<MTLComputePipelineState> ax_bPipelineState = nil;
static id<MTLComputePipelineState> matrixaddPipelineState = nil;
static id<MTLComputePipelineState> matrixcopyblockPipelineState = nil;
static id<MTLComputePipelineState> matrixproductPipelineState = nil;
static id<MTLComputePipelineState> expPipelineState = nil;
static id<MTLComputePipelineState> logPipelineState = nil;
static id<MTLComputePipelineState> elemInvPipelineState = nil;
static id<MTLComputePipelineState> squarePipelineState = nil;
static id<MTLComputePipelineState> tanhPipelineState = nil;
static id<MTLComputePipelineState> dTanhPipelineState = nil;
static id<MTLComputePipelineState> sqrtPipelineState = nil;
static id<MTLComputePipelineState> reluPipelineState = nil;
static id<MTLComputePipelineState> dreluPipelineState = nil;
static id<MTLComputePipelineState> im2colPipelineState = nil;
static id<MTLComputePipelineState> col2imPipelineState = nil;
static id<MTLComputePipelineState> packFeatureMap2DPipelineState = nil;
static id<MTLComputePipelineState> conv2dOutputAddBiasPipelineState = nil;
static id<MTLComputePipelineState> softmaxColPipelineState = nil;
static id<MTLComputePipelineState> causalMaskPipelineState = nil;
static id<MTLComputePipelineState> attnPackPipelineState = nil;
static id<MTLComputePipelineState> attnUnpackPipelineState = nil;
static id<MTLComputePipelineState> causalMaskBatchedPipelineState = nil;
static id<MTLComputePipelineState> softmaxRowsBatchedPipelineState = nil;
static id<MTLComputePipelineState> softmaxBackwardRowsBatchedPipelineState = nil;
static id<MTLComputePipelineState> layernormForwardPipelineState = nil;
static id<MTLComputePipelineState> layernormBackwardPipelineState = nil;
static id<MTLComputePipelineState> addBiasPipelineState = nil;
static id<MTLComputePipelineState> adamUpdatePipelineState = nil;

// MPS kernel objects are expensive to create and fully reusable across
// encodes, so cache them by shape. Training loops repeat a handful of shapes,
// making the hit rate ~100%; entries live until mpsDestroy().
static std::map<std::tuple<bool, bool, int, int, int>, MPSMatrixMultiplication*> gemmKernelCache;
static std::map<std::tuple<bool, bool, int, int, int, float>, MPSMatrixMultiplication*> gemmBatchedKernelCache;
static std::map<std::tuple<bool, int, int>, MPSMatrixVectorMultiplication*> gemvKernelCache;

MPSMatrixRandomMTGP32* randomKernel = nil;
MPSMatrixRandomDistributionDescriptor* randDesc = nil;
MPSMatrixRandomMTGP32* randomUniformKernel = nil;
MPSMatrixRandomDistributionDescriptor* randUniformDesc = nil;

static std::map<float*, id<MTLBuffer>> bufferMap;

// ── batched command encoding ────────────────────────────────────────────────
// Ops encode into one shared command buffer instead of committing one buffer
// per op; the buffer is committed when it fills up (kMaxEncodedOps) or on
// mpsSynchronize(). Consecutive custom kernels share a single serial compute
// encoder (serial dispatch keeps them ordered); MPS library kernels encode
// directly into the command buffer, so the encoder is closed around them.
// mpsSynchronize() waits on every committed-but-unfinished buffer, not just
// the most recent one.
static id<MTLCommandBuffer> commandBuffer = nil;               // open, not committed
static id<MTLComputeCommandEncoder> computeEncoder = nil;      // open encoder on commandBuffer
static std::vector<id<MTLCommandBuffer>> inflightBuffers;      // committed, not waited on
static int encodedOps = 0;
static const int kMaxEncodedOps = 256;

static id<MTLCommandBuffer> currentCommandBuffer() {
    if (commandBuffer == nil) {
        commandBuffer = [[commandQueue commandBuffer] retain];
    }
    return commandBuffer;
}

static void endComputeEncoder() {
    if (computeEncoder != nil) {
        [computeEncoder endEncoding];
        [computeEncoder release];
        computeEncoder = nil;
    }
}

static id<MTLComputeCommandEncoder> currentComputeEncoder() {
    if (computeEncoder == nil) {
        computeEncoder = [[currentCommandBuffer() computeCommandEncoder] retain];
    }
    return computeEncoder;
}

// MPS library kernels encode straight into the command buffer, which requires
// no compute encoder to be open.
static id<MTLCommandBuffer> currentMPSCommandBuffer() {
    endComputeEncoder();
    return currentCommandBuffer();
}

static void commitCurrent() {
    endComputeEncoder();
    if (commandBuffer != nil) {
        [commandBuffer commit];
        inflightBuffers.push_back(commandBuffer);  // keeps the retain
        commandBuffer = nil;
        encodedOps = 0;
    }
}

// Call after encoding each op: periodically flush so long op sequences reach
// the GPU before mpsSynchronize() is called.
static void opEncoded() {
    if (++encodedOps >= kMaxEncodedOps) commitCurrent();
}

void mpsInit() {
    device = MTLCreateSystemDefaultDevice();

    commandQueue = [device newCommandQueue];

    // Random number distribution descriptor (Normal distribution)
    randDesc =
        [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:0.0f standardDeviation:1.0f];
    randUniformDesc =
        [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:0.0f maximum:1.0f];

    randomKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                    destinationDataType:MPSDataTypeFloat32
                                                    seed:3334
                                                    distributionDescriptor:randDesc];

    randomUniformKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                    destinationDataType:MPSDataTypeFloat32
                                                    seed:3334
                                                    distributionDescriptor:randUniformDesc];

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
    id<MTLFunction> matrixcopyblockKernel = [defaultLibrary newFunctionWithName:@"matrix_copy_block"];
    id<MTLFunction> matrixproductKernel = [defaultLibrary newFunctionWithName:@"matrix_product"];
    id<MTLFunction> expKernel = [defaultLibrary newFunctionWithName:@"exp_kernel"];
    id<MTLFunction> logKernel = [defaultLibrary newFunctionWithName:@"log_kernel"];
    id<MTLFunction> elemInvKernel = [defaultLibrary newFunctionWithName:@"elem_inv_kernel"];
    id<MTLFunction> squareKernel = [defaultLibrary newFunctionWithName:@"square_kernel"];
    id<MTLFunction> tanhKernel = [defaultLibrary newFunctionWithName:@"tanh_kernel"];
    id<MTLFunction> dTanhKernel = [defaultLibrary newFunctionWithName:@"dtanh_kernel"];
    id<MTLFunction> sqrtKernel = [defaultLibrary newFunctionWithName:@"sqrt_kernel"];
    id<MTLFunction> reluKernel = [defaultLibrary newFunctionWithName:@"relu_kernel"];
    id<MTLFunction> dreluKernel = [defaultLibrary newFunctionWithName:@"drelu_kernel"];
    id<MTLFunction> im2colKernel = [defaultLibrary newFunctionWithName:@"im2col_kernel"];
    id<MTLFunction> col2imKernel = [defaultLibrary newFunctionWithName:@"col2im_kernel"];
    id<MTLFunction> packFeatureMap2DKernel = [defaultLibrary newFunctionWithName:@"pack_feature_map_2d_kernel"];
    id<MTLFunction> conv2dOutputAddBiasKernel = [defaultLibrary newFunctionWithName:@"conv2d_output_add_bias_kernel"];
    id<MTLFunction> softmaxColKernel = [defaultLibrary newFunctionWithName:@"softmax_col_kernel"];
    id<MTLFunction> causalMaskKernel = [defaultLibrary newFunctionWithName:@"causal_mask_kernel"];
    id<MTLFunction> attnPackKernel = [defaultLibrary newFunctionWithName:@"attn_pack_kernel"];
    id<MTLFunction> attnUnpackKernel = [defaultLibrary newFunctionWithName:@"attn_unpack_kernel"];
    id<MTLFunction> causalMaskBatchedKernel = [defaultLibrary newFunctionWithName:@"causal_mask_batched_kernel"];
    id<MTLFunction> softmaxRowsBatchedKernel = [defaultLibrary newFunctionWithName:@"softmax_rows_batched_kernel"];
    id<MTLFunction> softmaxBackwardRowsBatchedKernel = [defaultLibrary newFunctionWithName:@"softmax_backward_rows_batched_kernel"];
    id<MTLFunction> layernormForwardKernel = [defaultLibrary newFunctionWithName:@"layernorm_forward_rm_kernel"];
    id<MTLFunction> layernormBackwardKernel = [defaultLibrary newFunctionWithName:@"layernorm_backward_rm_kernel"];
    id<MTLFunction> addBiasKernel = [defaultLibrary newFunctionWithName:@"add_bias_rm_kernel"];
    id<MTLFunction> adamUpdateKernel = [defaultLibrary newFunctionWithName:@"adam_update_kernel"];

    // Create compute pipeline state
    zeroPipelineState = [device newComputePipelineStateWithFunction:zeroKernel error:&error];
    ax_bPipelineState = [device newComputePipelineStateWithFunction:ax_bkernel error:&error];
    matrixaddPipelineState = [device newComputePipelineStateWithFunction:matrixaddKernel error:&error];
    matrixcopyblockPipelineState = [device newComputePipelineStateWithFunction:matrixcopyblockKernel error:&error];
    matrixproductPipelineState = [device newComputePipelineStateWithFunction:matrixproductKernel error:&error];
    expPipelineState = [device newComputePipelineStateWithFunction:expKernel error:&error];
    logPipelineState = [device newComputePipelineStateWithFunction:logKernel error:&error];
    elemInvPipelineState = [device newComputePipelineStateWithFunction:elemInvKernel error:&error];
    squarePipelineState = [device newComputePipelineStateWithFunction:squareKernel error:&error];
    tanhPipelineState = [device newComputePipelineStateWithFunction:tanhKernel error:&error];
    dTanhPipelineState = [device newComputePipelineStateWithFunction:dTanhKernel error:&error];
    sqrtPipelineState = [device newComputePipelineStateWithFunction:sqrtKernel error:&error];
    reluPipelineState = [device newComputePipelineStateWithFunction:reluKernel error:&error];
    dreluPipelineState = [device newComputePipelineStateWithFunction:dreluKernel error:&error];
    im2colPipelineState = [device newComputePipelineStateWithFunction:im2colKernel error:&error];
    col2imPipelineState = [device newComputePipelineStateWithFunction:col2imKernel error:&error];
    packFeatureMap2DPipelineState = [device newComputePipelineStateWithFunction:packFeatureMap2DKernel error:&error];
    conv2dOutputAddBiasPipelineState = [device newComputePipelineStateWithFunction:conv2dOutputAddBiasKernel error:&error];
    softmaxColPipelineState = [device newComputePipelineStateWithFunction:softmaxColKernel error:&error];
    causalMaskPipelineState = [device newComputePipelineStateWithFunction:causalMaskKernel error:&error];
    attnPackPipelineState = [device newComputePipelineStateWithFunction:attnPackKernel error:&error];
    attnUnpackPipelineState = [device newComputePipelineStateWithFunction:attnUnpackKernel error:&error];
    causalMaskBatchedPipelineState = [device newComputePipelineStateWithFunction:causalMaskBatchedKernel error:&error];
    softmaxRowsBatchedPipelineState = [device newComputePipelineStateWithFunction:softmaxRowsBatchedKernel error:&error];
    softmaxBackwardRowsBatchedPipelineState = [device newComputePipelineStateWithFunction:softmaxBackwardRowsBatchedKernel error:&error];
    layernormForwardPipelineState = [device newComputePipelineStateWithFunction:layernormForwardKernel error:&error];
    layernormBackwardPipelineState = [device newComputePipelineStateWithFunction:layernormBackwardKernel error:&error];
    addBiasPipelineState = [device newComputePipelineStateWithFunction:addBiasKernel error:&error];
    adamUpdatePipelineState = [device newComputePipelineStateWithFunction:adamUpdateKernel error:&error];

}

void mpsDestroy() {
    mpsSynchronize();   // drain any encoded/committed work before teardown
    bufferMap.clear();
    for (auto& [key, kernel] : gemmKernelCache) [kernel release];
    gemmKernelCache.clear();
    for (auto& [key, kernel] : gemmBatchedKernelCache) [kernel release];
    gemmBatchedKernelCache.clear();
    for (auto& [key, kernel] : gemvKernelCache) [kernel release];
    gemvKernelCache.clear();
    [randUniformDesc release];
    [randomUniformKernel release];
    [randDesc release];
    [randomKernel release];
    [zeroPipelineState release];
    [ax_bPipelineState release];
    [matrixaddPipelineState release];
    [matrixcopyblockPipelineState release];
    [matrixproductPipelineState release];
    [expPipelineState release];
    [logPipelineState release];
    [elemInvPipelineState release];
    [squarePipelineState release];
    [tanhPipelineState release];
    [dTanhPipelineState release];
    [sqrtPipelineState release];
    [reluPipelineState release];
    [dreluPipelineState release];
    [im2colPipelineState release];
    [col2imPipelineState release];
    [packFeatureMap2DPipelineState release];
    [conv2dOutputAddBiasPipelineState release];
    [softmaxColPipelineState release];
    [causalMaskPipelineState release];
    [attnPackPipelineState release];
    [attnUnpackPipelineState release];
    [causalMaskBatchedPipelineState release];
    [softmaxRowsBatchedPipelineState release];
    [softmaxBackwardRowsBatchedPipelineState release];
    [layernormForwardPipelineState release];
    [layernormBackwardPipelineState release];
    [addBiasPipelineState release];
    [adamUpdatePipelineState release];
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
    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) A]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto matB = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) B]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowB columns:colB rowBytes:colB*sizeof(float) dataType:MPSDataTypeFloat32]];

    int resultRows = transposeA ? colA : rowA;
    int resultColumns = transposeB ? rowB : colB;
    int interiorColumns = transposeA ? rowA : colA;
    auto matC = [[MPSMatrix alloc] initWithBuffer:bufferMap[C]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:resultRows columns:resultColumns rowBytes:resultColumns*sizeof(float) dataType:MPSDataTypeFloat32]];

    const auto key = std::make_tuple(transposeA, transposeB, resultRows, resultColumns, interiorColumns);
    MPSMatrixMultiplication* gemmKernel;
    auto it = gemmKernelCache.find(key);
    if (it != gemmKernelCache.end()) {
        gemmKernel = it->second;
    } else {
        gemmKernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:transposeA transposeRight:transposeB resultRows:resultRows resultColumns:resultColumns interiorColumns:interiorColumns alpha:1.0f beta:0.0f];
        gemmKernelCache[key] = gemmKernel;
    }
    [gemmKernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    opEncoded();

    [matA release]; [matB release]; [matC release];
}

void mpsAdd(const float* A, const float* B, float* C, int rowA, int colA, bool transpose, float a, float b){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    id<MTLBuffer> bufferB = bufferMap[(float *) B];
    id<MTLBuffer> bufferC = bufferMap[C];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:matrixaddPipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    [encoder setBytes:&rowA length:sizeof(int) atIndex:3];
    [encoder setBytes:&colA length:sizeof(int) atIndex:4];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:5];
    [encoder setBytes:&a length:sizeof(float) atIndex:6];
    [encoder setBytes:&b length:sizeof(float) atIndex:7];


    MTLSize gridSize = MTLSizeMake(colA, rowA, 1); // Threads match matrix elements
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1); // Adjust for performance

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    opEncoded();
}

void mpsCopyMatrixBlock(const float* src, float* dst,
                        int srcRows, int srcCols, bool srcTranspose,
                        int srcRowOffset, int srcColOffset,
                        int copyRows, int copyCols,
                        int dstRows, int dstCols, bool dstTranspose,
                        int dstRowOffset, int dstColOffset) {
    id<MTLBuffer> srcBuffer = bufferMap[(float*)src];
    id<MTLBuffer> dstBuffer = bufferMap[dst];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:matrixcopyblockPipelineState];
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer offset:0 atIndex:1];
    [encoder setBytes:&srcRows length:sizeof(int) atIndex:2];
    [encoder setBytes:&srcCols length:sizeof(int) atIndex:3];
    [encoder setBytes:&srcTranspose length:sizeof(bool) atIndex:4];
    [encoder setBytes:&srcRowOffset length:sizeof(int) atIndex:5];
    [encoder setBytes:&srcColOffset length:sizeof(int) atIndex:6];
    [encoder setBytes:&dstRows length:sizeof(int) atIndex:7];
    [encoder setBytes:&dstCols length:sizeof(int) atIndex:8];
    [encoder setBytes:&dstTranspose length:sizeof(bool) atIndex:9];
    [encoder setBytes:&dstRowOffset length:sizeof(int) atIndex:10];
    [encoder setBytes:&dstColOffset length:sizeof(int) atIndex:11];

    MTLSize gridSize = MTLSizeMake(copyCols, copyRows, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    opEncoded();
}

void mpsProduct(const float* A, const float* B, float* C, int rowA, int colA, bool transpose){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    id<MTLBuffer> bufferB = bufferMap[(float *) B];
    id<MTLBuffer> bufferC = bufferMap[C];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:matrixproductPipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    [encoder setBytes:&rowA length:sizeof(int) atIndex:3];
    [encoder setBytes:&colA length:sizeof(int) atIndex:4];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:5];

    MTLSize gridSize = MTLSizeMake(colA, rowA, 1); // Threads match matrix elements
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1); // Adjust for performance

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    opEncoded();
}

void mpsAx_b(const float* x, float a, float b, float* y, int N) {
    id<MTLBuffer> bufferX = bufferMap[(float *) x];
    id<MTLBuffer> bufferY = bufferMap[y];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

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

    opEncoded();
}

void mpsRandn(float *A, int N){
    // Get GPU buffer associated with A
    id<MTLBuffer> bufferA = bufferMap[A];

    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    // Matrix descriptor to wrap buffer A
    MPSMatrixDescriptor* matrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:1
                                                                           rowBytes:sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:matrixDesc];

    // Encode random number generation into command buffer
    [randomKernel encodeToCommandBuffer:cb destinationMatrix:matA];
    opEncoded();
    // Cleanup
    [matA release];
}

void mpsRand(float *A, int N){
    // Get GPU buffer associated with A
    id<MTLBuffer> bufferA = bufferMap[A];

    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    // Matrix descriptor to wrap buffer A
    MPSMatrixDescriptor* matrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:1
                                                                           rowBytes:sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:matrixDesc];

    // Encode random number generation into command buffer
    [randomUniformKernel encodeToCommandBuffer:cb destinationMatrix:matA];
    opEncoded();
    // Cleanup
    [matA release];
}

void mpsFill(float* A, int N, float val) {
    id<MTLBuffer> buffer = bufferMap[A];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:zeroPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&val length:sizeof(float) atIndex:1];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = zeroPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

// Shared encode path for elementwise src -> dst kernels (in-place: src == dst).
static void encodeElemwise(id<MTLComputePipelineState> pso, const float* src, float* dst, int N) {
    id<MTLBuffer> srcBuffer = bufferMap[(float *) src];
    id<MTLBuffer> dstBuffer = bufferMap[dst];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer offset:0 atIndex:1];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > (NSUInteger)N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsRelu(const float* src, float* dst, int N){
    encodeElemwise(reluPipelineState, src, dst, N);
}

void mpsdRelu(const float* src, float* dst, int N){
    encodeElemwise(dreluPipelineState, src, dst, N);
}

void mpsIm2col(const float* input, float* col,
               int N, int C, int H, int W,
               int kH, int kW,
               int padH, int padW,
               int strideH, int strideW,
               int Hout, int Wout) {
    id<MTLBuffer> inBuffer = bufferMap[(float*)input];
    id<MTLBuffer> colBuffer = bufferMap[col];

    const int K = C * kH * kW;
    const int total = N * Hout * Wout * K;

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:im2colPipelineState];
    [encoder setBuffer:inBuffer offset:0 atIndex:0];
    [encoder setBuffer:colBuffer offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];
    [encoder setBytes:&C length:sizeof(int) atIndex:3];
    [encoder setBytes:&H length:sizeof(int) atIndex:4];
    [encoder setBytes:&W length:sizeof(int) atIndex:5];
    [encoder setBytes:&kH length:sizeof(int) atIndex:6];
    [encoder setBytes:&kW length:sizeof(int) atIndex:7];
    [encoder setBytes:&padH length:sizeof(int) atIndex:8];
    [encoder setBytes:&padW length:sizeof(int) atIndex:9];
    [encoder setBytes:&strideH length:sizeof(int) atIndex:10];
    [encoder setBytes:&strideW length:sizeof(int) atIndex:11];
    [encoder setBytes:&Hout length:sizeof(int) atIndex:12];
    [encoder setBytes:&Wout length:sizeof(int) atIndex:13];

    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = im2colPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    MTLSize threadsPerGroup = MTLSizeMake(tgs, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsCol2im(const float* col, float* output,
               int N, int C, int H, int W,
               int kH, int kW,
               int padH, int padW,
               int strideH, int strideW,
               int Hout, int Wout) {
    id<MTLBuffer> colBuffer = bufferMap[(float*)col];
    id<MTLBuffer> outBuffer = bufferMap[output];

    const int total = N * C * H * W;

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:col2imPipelineState];
    [encoder setBuffer:colBuffer offset:0 atIndex:0];
    [encoder setBuffer:outBuffer offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];
    [encoder setBytes:&C length:sizeof(int) atIndex:3];
    [encoder setBytes:&H length:sizeof(int) atIndex:4];
    [encoder setBytes:&W length:sizeof(int) atIndex:5];
    [encoder setBytes:&kH length:sizeof(int) atIndex:6];
    [encoder setBytes:&kW length:sizeof(int) atIndex:7];
    [encoder setBytes:&padH length:sizeof(int) atIndex:8];
    [encoder setBytes:&padW length:sizeof(int) atIndex:9];
    [encoder setBytes:&strideH length:sizeof(int) atIndex:10];
    [encoder setBytes:&strideW length:sizeof(int) atIndex:11];
    [encoder setBytes:&Hout length:sizeof(int) atIndex:12];
    [encoder setBytes:&Wout length:sizeof(int) atIndex:13];

    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = col2imPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    MTLSize threadsPerGroup = MTLSizeMake(tgs, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsPackFeatureMap2D(const float* featureMap, float* packed,
                        int N, int C, int P) {
    id<MTLBuffer> src = bufferMap[(float*)featureMap];
    id<MTLBuffer> dst = bufferMap[packed];
    const int total = N * C * P;

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:packFeatureMap2DPipelineState];
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:dst offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];
    [encoder setBytes:&C length:sizeof(int) atIndex:3];
    [encoder setBytes:&P length:sizeof(int) atIndex:4];

    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = packFeatureMap2DPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    MTLSize threadsPerGroup = MTLSizeMake(tgs, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsConv2dOutputAddBias(const float* y2d, const float* bias, float* output,
                            int N, int Cout, int Hout, int Wout) {
    id<MTLBuffer> y = bufferMap[(float*)y2d];
    id<MTLBuffer> b = bufferMap[(float*)bias];
    id<MTLBuffer> o = bufferMap[output];
    const int total = N * Cout * Hout * Wout;

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:conv2dOutputAddBiasPipelineState];
    [encoder setBuffer:y offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:o offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(int) atIndex:3];
    [encoder setBytes:&Cout length:sizeof(int) atIndex:4];
    [encoder setBytes:&Hout length:sizeof(int) atIndex:5];
    [encoder setBytes:&Wout length:sizeof(int) atIndex:6];

    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = conv2dOutputAddBiasPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    MTLSize threadsPerGroup = MTLSizeMake(tgs, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsTanh(const float* src, float* dst, int N){
    encodeElemwise(tanhPipelineState, src, dst, N);
}

void mpsdTanh(const float* src, float* dst, int N){
    encodeElemwise(dTanhPipelineState, src, dst, N);
}

void mpsSqrt(const float* src, float* dst, int N){
    encodeElemwise(sqrtPipelineState, src, dst, N);
}

void mpsExp(const float* src, float* dst, int N){
    encodeElemwise(expPipelineState, src, dst, N);
}

void mpsLog(const float* src, float* dst, int N){
    encodeElemwise(logPipelineState, src, dst, N);
}

void mpsSquare(const float* src, float* dst, int N){
    encodeElemwise(squarePipelineState, src, dst, N);
}

void mpsSoftmaxCol(const float* src, float* dst, int rows, int cols, bool transpose) {
    id<MTLBuffer> srcBuffer = bufferMap[(float*)src];
    id<MTLBuffer> dstBuffer = bufferMap[dst];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:softmaxColPipelineState];
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(int) atIndex:2];
    [encoder setBytes:&cols length:sizeof(int) atIndex:3];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:4];

    MTLSize gridSize = MTLSizeMake(cols, 1, 1);
    NSUInteger tgs = softmaxColPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)cols) tgs = cols;
    MTLSize threadsPerGroup = MTLSizeMake(tgs, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

void mpsCausalMask(float* S, int n, bool transpose, float maskVal) {
    id<MTLBuffer> buffer = bufferMap[S];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:causalMaskPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&n length:sizeof(int) atIndex:1];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:2];
    [encoder setBytes:&maskVal length:sizeof(float) atIndex:3];

    MTLSize gridSize = MTLSizeMake(n, n, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    opEncoded();
}

void mpsGemmBatched(const float* A, const float* B, float* C, int batch,
                    int rowA, int colA, int rowB, int colB,
                    bool transposeA, bool transposeB, float alpha) {
    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) A]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA
            matrices:batch rowBytes:colA*sizeof(float)
            matrixBytes:(NSUInteger)rowA*colA*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto matB = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) B]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowB columns:colB
            matrices:batch rowBytes:colB*sizeof(float)
            matrixBytes:(NSUInteger)rowB*colB*sizeof(float) dataType:MPSDataTypeFloat32]];

    int resultRows = transposeA ? colA : rowA;
    int resultColumns = transposeB ? rowB : colB;
    int interiorColumns = transposeA ? rowA : colA;
    auto matC = [[MPSMatrix alloc] initWithBuffer:bufferMap[C]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:resultRows columns:resultColumns
            matrices:batch rowBytes:resultColumns*sizeof(float)
            matrixBytes:(NSUInteger)resultRows*resultColumns*sizeof(float) dataType:MPSDataTypeFloat32]];

    const auto key = std::make_tuple(transposeA, transposeB, resultRows, resultColumns, interiorColumns, alpha);
    MPSMatrixMultiplication* gemmKernel;
    auto it = gemmBatchedKernelCache.find(key);
    if (it != gemmBatchedKernelCache.end()) {
        gemmKernel = it->second;
    } else {
        gemmKernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:transposeA transposeRight:transposeB resultRows:resultRows resultColumns:resultColumns interiorColumns:interiorColumns alpha:alpha beta:0.0f];
        gemmBatchedKernelCache[key] = gemmKernel;
    }
    [gemmKernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    opEncoded();

    [matA release]; [matB release]; [matC release];
}

// Shared encode path for the attention pack/unpack gather kernels.
static void encodeAttnRepack(id<MTLComputePipelineState> pso, const float* src, float* dst,
                             int d_h, int heads, int seq, int batch) {
    id<MTLBuffer> srcBuffer = bufferMap[(float*)src];
    id<MTLBuffer> dstBuffer = bufferMap[dst];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer offset:0 atIndex:1];
    [encoder setBytes:&d_h length:sizeof(int) atIndex:2];
    [encoder setBytes:&heads length:sizeof(int) atIndex:3];
    [encoder setBytes:&seq length:sizeof(int) atIndex:4];
    [encoder setBytes:&batch length:sizeof(int) atIndex:5];

    const int total = heads * batch * seq * d_h;
    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = pso.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsAttnPack(const float* src, float* dst, int d_h, int heads, int seq, int batch) {
    encodeAttnRepack(attnPackPipelineState, src, dst, d_h, heads, seq, batch);
}

void mpsAttnUnpack(const float* src, float* dst, int d_h, int heads, int seq, int batch) {
    encodeAttnRepack(attnUnpackPipelineState, src, dst, d_h, heads, seq, batch);
}

void mpsCausalMaskBatched(float* S, int n, int batch, float maskVal) {
    id<MTLBuffer> buffer = bufferMap[S];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:causalMaskBatchedPipelineState];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&n length:sizeof(int) atIndex:1];
    [encoder setBytes:&batch length:sizeof(int) atIndex:2];
    [encoder setBytes:&maskVal length:sizeof(float) atIndex:3];

    const int total = batch * n * n;
    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = causalMaskBatchedPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsSoftmaxRowsBatched(const float* x, float* y, int n, int rows) {
    id<MTLBuffer> xBuffer = bufferMap[(float*)x];
    id<MTLBuffer> yBuffer = bufferMap[y];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:softmaxRowsBatchedPipelineState];
    [encoder setBuffer:xBuffer offset:0 atIndex:0];
    [encoder setBuffer:yBuffer offset:0 atIndex:1];
    [encoder setBytes:&n length:sizeof(int) atIndex:2];
    [encoder setBytes:&rows length:sizeof(int) atIndex:3];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    NSUInteger tgs = softmaxRowsBatchedPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)rows) tgs = rows;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsSoftmaxBackwardRowsBatched(const float* A, const float* dA, float* dS,
                                   int n, int rows, float scale) {
    id<MTLBuffer> aBuffer = bufferMap[(float*)A];
    id<MTLBuffer> daBuffer = bufferMap[(float*)dA];
    id<MTLBuffer> dsBuffer = bufferMap[dS];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:softmaxBackwardRowsBatchedPipelineState];
    [encoder setBuffer:aBuffer offset:0 atIndex:0];
    [encoder setBuffer:daBuffer offset:0 atIndex:1];
    [encoder setBuffer:dsBuffer offset:0 atIndex:2];
    [encoder setBytes:&n length:sizeof(int) atIndex:3];
    [encoder setBytes:&rows length:sizeof(int) atIndex:4];
    [encoder setBytes:&scale length:sizeof(float) atIndex:5];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    NSUInteger tgs = softmaxBackwardRowsBatchedPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)rows) tgs = rows;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsLayerNormForward(const float* x, const float* gamma, const float* beta,
                         float* y, float* xhat, float* invStd, int dim, int N) {
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:layernormForwardPipelineState];
    [encoder setBuffer:bufferMap[(float*)x] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)gamma] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[(float*)beta] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[y] offset:0 atIndex:3];
    [encoder setBuffer:bufferMap[xhat] offset:0 atIndex:4];
    [encoder setBuffer:bufferMap[invStd] offset:0 atIndex:5];
    [encoder setBytes:&dim length:sizeof(int) atIndex:6];
    [encoder setBytes:&N length:sizeof(int) atIndex:7];

    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger tgs = layernormForwardPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)N) tgs = N;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsLayerNormBackward(const float* dy, const float* gamma, const float* xhat,
                          const float* invStd, float* dx, int dim, int N) {
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:layernormBackwardPipelineState];
    [encoder setBuffer:bufferMap[(float*)dy] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)gamma] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[(float*)xhat] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[(float*)invStd] offset:0 atIndex:3];
    [encoder setBuffer:bufferMap[dx] offset:0 atIndex:4];
    [encoder setBytes:&dim length:sizeof(int) atIndex:5];
    [encoder setBytes:&N length:sizeof(int) atIndex:6];

    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger tgs = layernormBackwardPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)N) tgs = N;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsAddBias(float* y, const float* b, int N, int total) {
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:addBiasPipelineState];
    [encoder setBuffer:bufferMap[y] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)b] offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];
    [encoder setBytes:&total length:sizeof(int) atIndex:3];

    MTLSize gridSize = MTLSizeMake(total, 1, 1);
    NSUInteger tgs = addBiasPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)total) tgs = total;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsAdamUpdate(float* g, float* m, float* v, float alpha, float beta1,
                   float beta2, float eps, float bc1, float bc2, int n) {
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:adamUpdatePipelineState];
    [encoder setBuffer:bufferMap[g] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[m] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[v] offset:0 atIndex:2];
    [encoder setBytes:&alpha length:sizeof(float) atIndex:3];
    [encoder setBytes:&beta1 length:sizeof(float) atIndex:4];
    [encoder setBytes:&beta2 length:sizeof(float) atIndex:5];
    [encoder setBytes:&eps length:sizeof(float) atIndex:6];
    [encoder setBytes:&bc1 length:sizeof(float) atIndex:7];
    [encoder setBytes:&bc2 length:sizeof(float) atIndex:8];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger tgs = adamUpdatePipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)n) tgs = n;
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsGemv(const float* A, const float* x, float* y, int rowA, int colA, bool transposeA) {
    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferMap[(float *) A]
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto vecX = [[MPSVector alloc] initWithBuffer:bufferMap[(float *) x]
        descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:transposeA ? rowA : colA dataType:MPSDataTypeFloat32]];

    auto vecY = [[MPSVector alloc] initWithBuffer:bufferMap[y]
        descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:transposeA ? colA : rowA dataType:MPSDataTypeFloat32]];

    const auto key = std::make_tuple(transposeA, rowA, colA);
    MPSMatrixVectorMultiplication* gemvKernel;
    auto it = gemvKernelCache.find(key);
    if (it != gemvKernelCache.end()) {
        gemvKernel = it->second;
    } else {
        gemvKernel = [[MPSMatrixVectorMultiplication alloc] initWithDevice:device
            transpose:transposeA rows:transposeA ? colA : rowA columns:transposeA ? rowA : colA alpha:1.0f beta:0.0f];
        gemvKernelCache[key] = gemvKernel;
    }
    [gemvKernel encodeToCommandBuffer:cb inputMatrix:matA inputVector:vecX resultVector:vecY];

    opEncoded();

    [matA release]; [vecX release]; [vecY release];
}

void mpsTopk(const float* A, float* B, float *C, int rowA, int colA, int k){
    id<MTLBuffer> bufferA = bufferMap[(float *) A];
    // id<MTLBuffer> bufferB = bufferMap[B];
    id<MTLBuffer> bufferB  = [device newBufferWithLength:rowA*k*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = bufferMap[C];

    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();

    auto matA = [[MPSMatrix alloc] initWithBuffer:bufferA
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:colA rowBytes:colA*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto matB = [[MPSMatrix alloc] initWithBuffer:bufferB
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:k rowBytes:k*sizeof(int) dataType:MPSDataTypeUInt32]];

    auto matC = [[MPSMatrix alloc] initWithBuffer:bufferC
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rowA columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32]];

    auto topkKernel = [[MPSMatrixFindTopK alloc] initWithDevice:device numberOfTopKValues:k];
    [topkKernel encodeToCommandBuffer:cb inputMatrix:matA resultIndexMatrix:matB resultValueMatrix:matC];

    opEncoded();

    mpsSynchronize();
    // std::cout << "Copying topk results" << std::endl;
    // memcpy(B, bufferB.contents, rowA*k*sizeof(int));
    // copy the memory from row major order to column major order
    for (int i = 0; i < rowA; i++){
        for (int j = 0; j < k; j++){
            B[j*rowA + i] = ((int *)bufferB.contents)[i*k + j];
        }
    }

    [matA release]; [matB release]; [matC release]; [topkKernel release]; [bufferB release];

}

void mpsElemInv(const float* src, float* dst, int N, float l){
    id<MTLBuffer> srcBuffer = bufferMap[(float *) src];
    id<MTLBuffer> dstBuffer = bufferMap[dst];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:elemInvPipelineState];
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer offset:0 atIndex:1];
    [encoder setBytes:&l length:sizeof(float) atIndex:2];

    // Set up thread groups
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    NSUInteger threadGroupSize = elemInvPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > (NSUInteger)N) threadGroupSize = N;

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];

    opEncoded();
}

/* Commit all encoded work and wait until every in-flight command buffer has
   completed, so the host can safely read/write buffer contents. */
void mpsSynchronize() {
    commitCurrent();
    for (id<MTLCommandBuffer> cb : inflightBuffers) {
        [cb waitUntilCompleted];
        [cb release];
    }
    inflightBuffers.clear();
}
