// private_memory.mm - Allocates a Metal private buffer and verifies allocation.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <iostream>

int main() {
    @autoreleasepool {
        // 1️⃣ Create a Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this system.");
            return -1;
        }

        // 2️⃣ Allocate a Private Buffer (GPU-Only)
        NSUInteger bufferSize = 1024; // 1KB buffer
        id<MTLBuffer> privateBuffer = [device newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModePrivate];

        std::cout << "Buffer size: " << bufferSize << " bytes" << std::endl;
        std::cout << "Buffer contents: " << privateBuffer.contents << std::endl;
        // 3️⃣ Verify Allocation
        if (privateBuffer) {
            NSLog(@"✅ Successfully allocated private GPU buffer (%lu bytes)", bufferSize);
        } else {
            NSLog(@"❌ Failed to allocate private GPU buffer!");
            return -1;
        }

        return 0;
    }
}
