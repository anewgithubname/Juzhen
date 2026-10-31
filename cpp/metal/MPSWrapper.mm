#include <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "MPSWrapper.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <string>
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
static id<MTLComputePipelineState> gatherPipelineState = nil;
static id<MTLComputePipelineState> boxMullerPipelineState = nil;
static id<MTLComputePipelineState> adamwPipelineState = nil;
static id<MTLComputePipelineState> layernormJvpPipelineState = nil;
static id<MTLComputePipelineState> reluBackwardPipelineState = nil;
static id<MTLComputePipelineState> columnMaxPipelineState = nil;
static id<MTLComputePipelineState> clampPipelineState = nil;
static id<MTLComputePipelineState> stepPipelineState = nil;
static id<MTLComputePipelineState> layernormForwardTiledPipelineState = nil;
static id<MTLComputePipelineState> layernormBackwardTiledPipelineState = nil;
static id<MTLComputePipelineState> layernormJvpTiledPipelineState = nil;

// MPS kernel objects are expensive to create and fully reusable across
// encodes, so cache them by shape. Training loops repeat a handful of shapes,
// making the hit rate ~100%; entries live until mpsDestroy().
static std::map<std::tuple<bool, bool, int, int, int>, MPSMatrixMultiplication*> gemmKernelCache;
static std::map<std::tuple<bool, bool, int, int, int, float, float>, MPSMatrixMultiplication*> gemmBatchedKernelCache;
static std::map<std::tuple<bool, int, int>, MPSMatrixVectorMultiplication*> gemvKernelCache;

// One uniform generator feeds both rand and randn (via Box-Muller), so a
// reseed restarts both streams, as with the single cuRAND generator.
static MPSMatrixRandomMTGP32* randomUniformKernel = nil;
static MPSMatrixRandomDistributionDescriptor* randUniformDesc = nil;
static const NSUInteger kDefaultSeed = 3334;

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

    randUniformDesc =
        [[MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:0.0f maximum:1.0f] retain];
    mpsSetSeed((int)kDefaultSeed);

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
    id<MTLFunction> gatherKernel = [defaultLibrary newFunctionWithName:@"matrix_gather_kernel"];
    id<MTLFunction> boxMullerKernel = [defaultLibrary newFunctionWithName:@"box_muller_kernel"];
    id<MTLFunction> adamwKernel = [defaultLibrary newFunctionWithName:@"adamw_kernel"];
    id<MTLFunction> layernormJvpKernel = [defaultLibrary newFunctionWithName:@"layernorm_jvp_rm_kernel"];
    id<MTLFunction> reluBackwardKernel = [defaultLibrary newFunctionWithName:@"relu_backward_inplace_kernel"];
    id<MTLFunction> columnMaxKernel = [defaultLibrary newFunctionWithName:@"column_max_kernel"];
    id<MTLFunction> clampKernel = [defaultLibrary newFunctionWithName:@"clamp_kernel"];
    id<MTLFunction> stepKernel = [defaultLibrary newFunctionWithName:@"step_kernel"];
    id<MTLFunction> layernormForwardTiledKernel = [defaultLibrary newFunctionWithName:@"layernorm_forward_tiled_kernel"];
    id<MTLFunction> layernormBackwardTiledKernel = [defaultLibrary newFunctionWithName:@"layernorm_backward_tiled_kernel"];
    id<MTLFunction> layernormJvpTiledKernel = [defaultLibrary newFunctionWithName:@"layernorm_jvp_tiled_kernel"];

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
    gatherPipelineState = [device newComputePipelineStateWithFunction:gatherKernel error:&error];
    boxMullerPipelineState = [device newComputePipelineStateWithFunction:boxMullerKernel error:&error];
    adamwPipelineState = [device newComputePipelineStateWithFunction:adamwKernel error:&error];
    layernormJvpPipelineState = [device newComputePipelineStateWithFunction:layernormJvpKernel error:&error];
    reluBackwardPipelineState = [device newComputePipelineStateWithFunction:reluBackwardKernel error:&error];
    columnMaxPipelineState = [device newComputePipelineStateWithFunction:columnMaxKernel error:&error];
    clampPipelineState = [device newComputePipelineStateWithFunction:clampKernel error:&error];
    stepPipelineState = [device newComputePipelineStateWithFunction:stepKernel error:&error];
    layernormForwardTiledPipelineState = [device newComputePipelineStateWithFunction:layernormForwardTiledKernel error:&error];
    layernormBackwardTiledPipelineState = [device newComputePipelineStateWithFunction:layernormBackwardTiledKernel error:&error];
    layernormJvpTiledPipelineState = [device newComputePipelineStateWithFunction:layernormJvpTiledKernel error:&error];

}

const char* mpsDeviceName() {
    static std::string name;
    id<MTLDevice> d = device != nil ? device : MTLCreateSystemDefaultDevice();
    name = d != nil ? [[d name] UTF8String] : "N/A";
    if (device == nil && d != nil) [d release];
    return name.c_str();
}

void mpsSetSeed(int seed) {
    // Pending work may still use the old generator's state buffers.
    mpsSynchronize();
    [randomUniformKernel release];
    randomUniformKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                    destinationDataType:MPSDataTypeFloat32
                                                    seed:(NSUInteger)(unsigned)seed
                                                    distributionDescriptor:randUniformDesc];
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
    randUniformDesc = nil;
    randomUniformKernel = nil;
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
    [gatherPipelineState release];
    [boxMullerPipelineState release];
    [adamwPipelineState release];
    [layernormJvpPipelineState release];
    [reluBackwardPipelineState release];
    [columnMaxPipelineState release];
    [clampPipelineState release];
    [stepPipelineState release];
    [layernormForwardTiledPipelineState release];
    [layernormBackwardTiledPipelineState release];
    [layernormJvpTiledPipelineState release];
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

unsigned long long mpsGpuAddress(const float* ptr) {
    auto it = bufferMap.find((float*)ptr);
    return it == bufferMap.end() ? 0 : it->second.gpuAddress;
}

float* mpsStagingAlloc(size_t size) {
    return mpsMalloc(size);
}

void mpsStagingRelease(float* ptr) {
    mpsFree(ptr);
}

void mpsFree(float* ptr) {
    auto it = bufferMap.find(ptr);
    if(it != bufferMap.end()) {
        [it->second release];
        bufferMap.erase(it);
    }
}
// TEMPORARY WORKAROUND for an Apple MPS bug. MPSMatrixMultiplication writes a
// band of NaNs into its result when the result spans a 2 GiB-aligned GPU
// virtual address, and also when it ends less than about 8 MiB below one.
// Reproduced on an M2 (macOS 27.0.1) in a standalone Metal program; the
// operands, MPS gemv, MTGP32 and our own kernels are unaffected by the same
// placement. Such results are computed into scratch buffers that keep clear of
// the boundary, in blocks of at most kGemmScratchBytes, and copied into place
// with our copy kernel.
//
// To check whether a newer macOS has fixed the bug, configure with
// -DJUZHEN_MPS_GEMM_WORKAROUND=OFF and run ctest -R mps_gemm_address_boundary.
// If it passes, remove this block, gemmScratch and the scratch paths in
// mpsGemm and mpsGemmBatched; keep the test.
#ifndef JUZHEN_MPS_GEMM_WORKAROUND
#define JUZHEN_MPS_GEMM_WORKAROUND 1
#endif
static const unsigned long long kGemmBoundary = 1ull << 31;
static const unsigned long long kGemmGuardBytes = 16ull << 20;  // twice the measured 8 MiB
static const NSUInteger kGemmScratchBytes = 256u << 20;

static bool nearGemmBoundary(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes) {
    if (!JUZHEN_MPS_GEMM_WORKAROUND || bytes == 0) return false;
    const unsigned long long lo = buffer.gpuAddress + offset, hi = lo + bytes - 1;
    return lo / kGemmBoundary != (hi + kGemmGuardBytes) / kGemmBoundary;
}

// A registered buffer of `bytes` (well below 2 GiB) clear of the boundary:
// a rejected allocation is held until the next one lands elsewhere.
static float* gemmScratch(NSUInteger bytes) {
    std::vector<float*> rejected;
    float* p = mpsMalloc(bytes);
    while (nearGemmBoundary(bufferMap[p], 0, bytes)) {
        rejected.push_back(p);
        p = mpsMalloc(bytes);
    }
    for (float* r : rejected) mpsFree(r);
    return p;
}

static MPSMatrix* matrixView(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger rows,
                             NSUInteger columns, NSUInteger rowBytes) {
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset
        descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
                                                         rowBytes:rowBytes dataType:MPSDataTypeFloat32]];
}

static MPSMatrixMultiplication* gemmKernel(bool transposeA, bool transposeB, int resultRows,
                                           int resultColumns, int interiorColumns) {
    const auto key = std::make_tuple(transposeA, transposeB, resultRows, resultColumns, interiorColumns);
    auto it = gemmKernelCache.find(key);
    if (it != gemmKernelCache.end()) return it->second;
    MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
        transposeLeft:transposeA transposeRight:transposeB resultRows:resultRows resultColumns:resultColumns
        interiorColumns:interiorColumns alpha:1.0f beta:0.0f];
    gemmKernelCache[key] = kernel;
    return kernel;
}

static void encodeGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB,
                       bool transposeA, bool transposeB) {
    const int resultRows = transposeA ? colA : rowA;
    const int resultColumns = transposeB ? rowB : colB;
    const int interiorColumns = transposeA ? rowA : colA;
    if (resultRows <= 0 || resultColumns <= 0) return;
    id<MTLBuffer> bufferA = bufferMap[(float*)A], bufferC = bufferMap[C];
    const NSUInteger rowBytesC = (NSUInteger)resultColumns * sizeof(float);

    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();
    MPSMatrix* matB = matrixView(bufferMap[(float*)B], 0, rowB, colB, colB * sizeof(float));

    if (!nearGemmBoundary(bufferC, 0, resultRows * rowBytesC)) {
        MPSMatrix* matA = matrixView(bufferA, 0, rowA, colA, colA * sizeof(float));
        MPSMatrix* matC = matrixView(bufferC, 0, resultRows, resultColumns, rowBytesC);
        [gemmKernel(transposeA, transposeB, resultRows, resultColumns, interiorColumns)
            encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        opEncoded();
        [matA release]; [matB release]; [matC release];
        return;
    }

    // Row blocks of the result: block rows of op(A) are rows of A, or columns
    // of A when it is transposed.
    const int block = (int)std::max<NSUInteger>(1, std::min<NSUInteger>(resultRows, kGemmScratchBytes / rowBytesC));
    float* scratch = gemmScratch(block * rowBytesC);
    for (int r0 = 0; r0 < resultRows; r0 += block) {
        const int rows = std::min(block, resultRows - r0);
        MPSMatrix* matA = transposeA
            ? matrixView(bufferA, (NSUInteger)r0 * sizeof(float), rowA, rows, colA * sizeof(float))
            : matrixView(bufferA, (NSUInteger)r0 * colA * sizeof(float), rows, colA, colA * sizeof(float));
        MPSMatrix* matC = matrixView(bufferMap[scratch], 0, rows, resultColumns, rowBytesC);
        [gemmKernel(transposeA, transposeB, rows, resultColumns, interiorColumns)
            encodeToCommandBuffer:currentMPSCommandBuffer() leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        opEncoded();
        [matA release]; [matC release];
        mpsCopyMatrixBlock(scratch, C, rows, resultColumns, false, 0, 0, rows, resultColumns,
                           resultRows, resultColumns, false, r0, 0);
    }
    [matB release];
    mpsFree(scratch);  // pending command buffers keep it alive until they finish
}

/**
 * MPSMatrixMultiplication
 * @param A: M x K
 * @param B: K x N
 * @param C: M x N
 */
void mpsGemm(const float* A, const float* B, float* C, int rowA, int colA, int rowB, int colB, bool transposeA, bool transposeB) {
    // Encoding leaves autoreleased MPS objects that retain the result buffer.
    // Callers have no pool to drain, so without this every scratch buffer
    // would stay allocated for the life of the process.
    @autoreleasepool {
        encodeGemm(A, B, C, rowA, colA, rowB, colB, transposeA, transposeB);
    }
}

void mpsAdd(const float* A, const float* B, float* C, int rowA, int colA, bool transpose, float a, float b){
    if (rowA <= 0 || colA <= 0) return;
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
    if (copyRows <= 0 || copyCols <= 0) return;
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
    if (rowA <= 0 || colA <= 0) return;
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
    if (N <= 0) return;
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

// Fill `buffer` with `count` uniforms in [0, 1) from the shared generator.
static void encodeUniform(id<MTLBuffer> buffer, NSUInteger count) {
    id<MTLCommandBuffer> cb = currentMPSCommandBuffer();
    MPSMatrixDescriptor* matrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:count columns:1
                                                                           rowBytes:sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
    auto mat = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:matrixDesc];
    [randomUniformKernel encodeToCommandBuffer:cb destinationMatrix:mat];
    opEncoded();
    [mat release];
}

void mpsRandn(float *A, int N){
    if (N <= 0) return;
    id<MTLBuffer> bufferA = bufferMap[A];
    // Box-Muller consumes uniforms in pairs; an odd request draws one extra
    // into a scratch buffer (retained by the command buffer until it runs).
    id<MTLBuffer> uniform = bufferA;
    if (N % 2) uniform = [device newBufferWithLength:(N + 1) * sizeof(float)
                                             options:MTLResourceStorageModePrivate];
    encodeUniform(uniform, N + N % 2);

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:boxMullerPipelineState];
    [encoder setBuffer:uniform offset:0 atIndex:0];
    [encoder setBuffer:bufferA offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(int) atIndex:2];
    const NSUInteger pairs = (N + 1) / 2;
    NSUInteger tgs = boxMullerPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > pairs) tgs = pairs;
    [encoder dispatchThreads:MTLSizeMake(pairs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
    opEncoded();
    if (uniform != bufferA) [uniform release];
}

void mpsRand(float *A, int N){
    if (N <= 0) return;
    encodeUniform(bufferMap[A], N);
}

void mpsGather(const float* src, float* dst, const unsigned* rows, int nr,
               const unsigned* cols, int nc, int srcRows, int srcCols, bool srcTranspose) {
    if (nr <= 0 || nc <= 0) return;
    // Index lists are copied into scratch buffers that the command buffer
    // retains until the gather has run.
    id<MTLBuffer> rowBuffer = [device newBufferWithBytes:rows length:nr * sizeof(unsigned)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> colBuffer = [device newBufferWithBytes:cols length:nc * sizeof(unsigned)
                                                 options:MTLResourceStorageModeShared];

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:gatherPipelineState];
    [encoder setBuffer:bufferMap[(float*)src] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[dst] offset:0 atIndex:1];
    [encoder setBuffer:rowBuffer offset:0 atIndex:2];
    [encoder setBuffer:colBuffer offset:0 atIndex:3];
    [encoder setBytes:&nr length:sizeof(int) atIndex:4];
    [encoder setBytes:&nc length:sizeof(int) atIndex:5];
    [encoder setBytes:&srcRows length:sizeof(int) atIndex:6];
    [encoder setBytes:&srcCols length:sizeof(int) atIndex:7];
    [encoder setBytes:&srcTranspose length:sizeof(bool) atIndex:8];

    const NSUInteger total = (NSUInteger)nr * nc;
    NSUInteger tgs = gatherPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > total) tgs = total;
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
    opEncoded();
    [rowBuffer release];
    [colBuffer release];
}

void mpsFill(float* A, int N, float val) {
    if (N <= 0) return;
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
    if (N <= 0) return;
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

static void encodeGemmBatched(const float* A, const float* B, float* C, int batch,
                              int rowA, int colA, int rowB, int colB,
                              bool transposeA, bool transposeB, float alpha, float beta) {
    const int resultRows = transposeA ? colA : rowA;
    const int resultColumns = transposeB ? rowB : colB;
    const int interiorColumns = transposeA ? rowA : colA;
    if (batch <= 0 || resultRows <= 0 || resultColumns <= 0) return;
    const NSUInteger bytesA = (NSUInteger)rowA * colA * sizeof(float);
    const NSUInteger bytesB = (NSUInteger)rowB * colB * sizeof(float);
    const NSUInteger bytesC = (NSUInteger)resultRows * resultColumns * sizeof(float);
    id<MTLBuffer> bufferA = bufferMap[(float*)A], bufferB = bufferMap[(float*)B], bufferC = bufferMap[C];

    const auto key = std::make_tuple(transposeA, transposeB, resultRows, resultColumns, interiorColumns, alpha, beta);
    MPSMatrixMultiplication* gemmKernel;
    auto it = gemmBatchedKernelCache.find(key);
    if (it != gemmBatchedKernelCache.end()) {
        gemmKernel = it->second;
    } else {
        gemmKernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:transposeA transposeRight:transposeB resultRows:resultRows resultColumns:resultColumns interiorColumns:interiorColumns alpha:alpha beta:beta];
        gemmBatchedKernelCache[key] = gemmKernel;
    }
    auto batchView = [](id<MTLBuffer> buffer, NSUInteger offset, int count, int rows, int columns) {
        return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset
            descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
                matrices:count rowBytes:columns*sizeof(float)
                matrixBytes:(NSUInteger)rows*columns*sizeof(float) dataType:MPSDataTypeFloat32]];
    };

    if (!nearGemmBoundary(bufferC, 0, batch * bytesC)) {
        MPSMatrix* matA = batchView(bufferA, 0, batch, rowA, colA);
        MPSMatrix* matB = batchView(bufferB, 0, batch, rowB, colB);
        MPSMatrix* matC = batchView(bufferC, 0, batch, resultRows, resultColumns);
        [gemmKernel encodeToCommandBuffer:currentMPSCommandBuffer() leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        opEncoded();
        [matA release]; [matB release]; [matC release];
        return;
    }

    // Groups of whole matrices through a boundary-free scratch buffer (see
    // mpsGemm); with beta != 0 the scratch starts from the current result.
    const int group = (int)std::max<NSUInteger>(1, std::min<NSUInteger>(batch, kGemmScratchBytes / bytesC));
    float* scratch = gemmScratch(group * bytesC);
    const int elementsC = resultRows * resultColumns;
    for (int b0 = 0; b0 < batch; b0 += group) {
        const int count = std::min(group, batch - b0);
        const int span = count * elementsC;
        if (beta != 0.0f)
            mpsCopyMatrixBlock(C, scratch, 1, batch * elementsC, false, 0, b0 * elementsC, 1, span,
                               1, span, false, 0, 0);
        MPSMatrix* matA = batchView(bufferA, b0 * bytesA, count, rowA, colA);
        MPSMatrix* matB = batchView(bufferB, b0 * bytesB, count, rowB, colB);
        MPSMatrix* matC = batchView(bufferMap[scratch], 0, count, resultRows, resultColumns);
        [gemmKernel encodeToCommandBuffer:currentMPSCommandBuffer() leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        opEncoded();
        [matA release]; [matB release]; [matC release];
        mpsCopyMatrixBlock(scratch, C, 1, span, false, 0, 0, 1, span, 1, batch * elementsC, false, 0, b0 * elementsC);
    }
    mpsFree(scratch);
}

void mpsGemmBatched(const float* A, const float* B, float* C, int batch,
                    int rowA, int colA, int rowB, int colB,
                    bool transposeA, bool transposeB, float alpha, float beta) {
    @autoreleasepool {  // see mpsGemm
        encodeGemmBatched(A, B, C, batch, rowA, colA, rowB, colB, transposeA, transposeB, alpha, beta);
    }
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

// LayerNorm kernels: one thread per column for narrow columns, otherwise a
// (32 x 8) threadgroup per 32 columns (the CUDA warp kernels' threshold).
static const int kLayerNormTiledMinDim = 32;

static void dispatchLayerNorm(id<MTLComputeCommandEncoder> encoder,
                              id<MTLComputePipelineState> serial, int dim, int N) {
    if (dim >= kLayerNormTiledMinDim) {
        [encoder dispatchThreadgroups:MTLSizeMake((N + 31) / 32, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    } else {
        NSUInteger tgs = serial.maxTotalThreadsPerThreadgroup;
        if (tgs > (NSUInteger)N) tgs = N;
        [encoder dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
    }
}

void mpsLayerNormForward(const float* x, const float* gamma, const float* beta,
                         float* y, float* xhat, float* invStd, int dim, int N) {
    if (N <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:dim >= kLayerNormTiledMinDim
        ? layernormForwardTiledPipelineState : layernormForwardPipelineState];
    [encoder setBuffer:bufferMap[(float*)x] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)gamma] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[(float*)beta] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[y] offset:0 atIndex:3];
    [encoder setBuffer:bufferMap[xhat] offset:0 atIndex:4];
    [encoder setBuffer:bufferMap[invStd] offset:0 atIndex:5];
    [encoder setBytes:&dim length:sizeof(int) atIndex:6];
    [encoder setBytes:&N length:sizeof(int) atIndex:7];

    dispatchLayerNorm(encoder, layernormForwardPipelineState, dim, N);

    opEncoded();
}

void mpsLayerNormBackward(const float* dy, const float* gamma, const float* xhat,
                          const float* invStd, float* dx, int dim, int N) {
    if (N <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:dim >= kLayerNormTiledMinDim
        ? layernormBackwardTiledPipelineState : layernormBackwardPipelineState];
    [encoder setBuffer:bufferMap[(float*)dy] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)gamma] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[(float*)xhat] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[(float*)invStd] offset:0 atIndex:3];
    [encoder setBuffer:bufferMap[dx] offset:0 atIndex:4];
    [encoder setBytes:&dim length:sizeof(int) atIndex:5];
    [encoder setBytes:&N length:sizeof(int) atIndex:6];

    dispatchLayerNorm(encoder, layernormBackwardPipelineState, dim, N);

    opEncoded();
}

void mpsLayerNormJvp(const float* dx, const float* gamma, const float* xhat,
                     const float* invStd, float* dy, int dim, int N) {
    if (N <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:dim >= kLayerNormTiledMinDim
        ? layernormJvpTiledPipelineState : layernormJvpPipelineState];
    [encoder setBuffer:bufferMap[(float*)dx] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)gamma] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[(float*)xhat] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[(float*)invStd] offset:0 atIndex:3];
    [encoder setBuffer:bufferMap[dy] offset:0 atIndex:4];
    [encoder setBytes:&dim length:sizeof(int) atIndex:5];
    [encoder setBytes:&N length:sizeof(int) atIndex:6];

    dispatchLayerNorm(encoder, layernormJvpPipelineState, dim, N);

    opEncoded();
}

void mpsReluBackwardInplace(const float* activation, float* gradient, int total) {
    encodeElemwise(reluBackwardPipelineState, activation, gradient, total);
}

void mpsColumnMax(const float* src, float* dst, int rows, int cols, bool transpose) {
    if (cols <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:columnMaxPipelineState];
    [encoder setBuffer:bufferMap[(float*)src] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[dst] offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(int) atIndex:2];
    [encoder setBytes:&cols length:sizeof(int) atIndex:3];
    [encoder setBytes:&transpose length:sizeof(bool) atIndex:4];

    NSUInteger tgs = columnMaxPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)cols) tgs = cols;
    [encoder dispatchThreads:MTLSizeMake(cols, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsClamp(const float* src, float* dst, int N, float lo, float hi) {
    if (N <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:clampPipelineState];
    [encoder setBuffer:bufferMap[(float*)src] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[dst] offset:0 atIndex:1];
    [encoder setBytes:&lo length:sizeof(float) atIndex:2];
    [encoder setBytes:&hi length:sizeof(float) atIndex:3];

    NSUInteger tgs = clampPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)N) tgs = N;
    [encoder dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

    opEncoded();
}

void mpsStep(const float* src, float* dst, int N, float threshold) {
    if (N <= 0) return;
    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();

    [encoder setComputePipelineState:stepPipelineState];
    [encoder setBuffer:bufferMap[(float*)src] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[dst] offset:0 atIndex:1];
    [encoder setBytes:&threshold length:sizeof(float) atIndex:2];

    NSUInteger tgs = stepPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > (NSUInteger)N) tgs = N;
    [encoder dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];

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
    if (N <= 0) return;
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

void mpsAdamW(float* w, const float* g, float* m, float* v, int rows, int cols,
              bool wt, bool gt, bool mt, bool vt, float lr, float beta1, float beta2,
              float eps, float bc1, float bc2, float decay) {
    const NSUInteger total = (NSUInteger)rows * cols;
    if (total == 0) return;
    const struct { float lr, beta1, beta2, eps, bc1, bc2, decay; } params =
        {lr, beta1, beta2, eps, bc1, bc2, decay};
    const unsigned layout = (wt ? 1u : 0u) | (gt ? 2u : 0u) | (mt ? 4u : 0u) | (vt ? 8u : 0u);

    id<MTLComputeCommandEncoder> encoder = currentComputeEncoder();
    [encoder setComputePipelineState:adamwPipelineState];
    [encoder setBuffer:bufferMap[w] offset:0 atIndex:0];
    [encoder setBuffer:bufferMap[(float*)g] offset:0 atIndex:1];
    [encoder setBuffer:bufferMap[m] offset:0 atIndex:2];
    [encoder setBuffer:bufferMap[v] offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(int) atIndex:4];
    [encoder setBytes:&cols length:sizeof(int) atIndex:5];
    [encoder setBytes:&layout length:sizeof(unsigned) atIndex:6];
    [encoder setBytes:&params length:sizeof(params) atIndex:7];
    NSUInteger tgs = adamwPipelineState.maxTotalThreadsPerThreadgroup;
    if (tgs > total) tgs = total;
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
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
