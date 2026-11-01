#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, JuzhenExample) {
    JuzhenExampleHelloWorld,  // examples/helloworld.cu
    JuzhenExampleGEMM,        // examples/demo_gemm.cu
    JuzhenExampleCIFARSpeed,  // CIFAR.cpp (examples/demo_cnn_rectified.cu)
};

@interface JuzhenRunner : NSObject

// Name of the Metal device the examples run on.
+ (NSString *)gpuName;

// Runs one example on the calling thread (call it off the main thread). Every
// flushed chunk of std::cout / std::cerr is passed to `output`. Waits for any
// other run to finish first.
+ (int)runExample:(JuzhenExample)example output:(void (^)(NSString *chunk))output;

// Trains the Zero-Flow velocity field and integrates the particles
// (stream_zero_flow in examples/demo_zeroflow_web.cu) on the calling thread,
// passing each NDJSON line of the demo's POST /train stream to `emit`. Stops
// early once `emit` returns NO. Returns NO for an unknown dataset name. Waits
// for any other run to finish first.
+ (BOOL)zeroFlowWithSource:(NSString *)source
                    target:(NSString *)target
                     iters:(NSInteger)iters
                      emit:(BOOL (^)(NSData *line))emit
    NS_SWIFT_NAME(zeroFlow(source:target:iters:emit:));

// Samples `count` CIFAR-10 images with `steps` Euler (or Heun, two network
// evaluations per step) steps from a checkpoint saved by
// examples/demo_cnn_rectified.cu, on the calling thread. `frame` gets
// the images as one RGBA grid every `every` steps (and at step 0 and the last
// step); `epochs` is set to how long the checkpoint was trained. Returns nil,
// or an error message if the checkpoint cannot be loaded. Waits for any other
// run to finish first.
+ (nullable NSString *)cifarGenerateFromCheckpoint:(NSString *)path
                                             count:(NSInteger)count
                                             steps:(NSInteger)steps
                                              heun:(BOOL)heun
                                             every:(NSInteger)every
                                            epochs:(NSInteger *)epochs
                                             frame:(void (^)(NSInteger step, NSData *rgba,
                                                             NSInteger width, NSInteger height))frame
    NS_SWIFT_NAME(cifarGenerate(checkpoint:count:steps:heun:every:epochs:frame:));

@end

NS_ASSUME_NONNULL_END
