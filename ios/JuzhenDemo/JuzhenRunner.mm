#import "JuzhenRunner.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <streambuf>
#include <string>

#include "../../cpp/juzhen.hpp"

// compute() of examples/helloworld.cu and examples/demo_gemm.cu, renamed by
// per-file -Dcompute=... flags in the Xcode project so both link into one app.
int juzhen_helloworld();
int juzhen_demo_gemm();

// CIFAR.cpp
int juzhen_cifar_speed();
bool juzhen_cifar_generate(const std::string &checkpoint, int count, int steps, bool heun, int every,
                           int *epochs,
                           const std::function<void(int, const std::vector<uint8_t> &, int, int)> &frame,
                           std::string *error);

// From examples/demo_zeroflow_web.cu.
bool valid_dataset(const std::string &name);
bool stream_zero_flow(const std::string &src, const std::string &tgt, int iters,
                      const std::function<bool(const std::string &)> &emit);

namespace {

// The matrix library is not thread-safe, so runs take turns.
std::mutex juzhen_mutex;

// One run of Juzhen code. Metal is set up on the first run and kept for the
// life of the app; the memory pools are released when each run ends, as at
// the end of main() in cpp/launcher.cu.
struct Run {
    std::lock_guard<std::mutex> lock{juzhen_mutex};
    Memory<int> mdi;
    Memory<float> md32;
    Memory<double> md64;
    Memory<MPSfloat> mpsmd;

    Run() {
        static std::once_flag metal;
        std::call_once(metal, mpsInit);
    }
    // Let the GPU finish with the buffers before the pools free them.
    ~Run() { mpsSynchronize(); }
};

// Hands each flushed chunk of std::cout to the UI and echoes it to stdout,
// so the output also shows up in Xcode's console.
class CallbackBuf : public std::streambuf {
   public:
    explicit CallbackBuf(void (^output)(NSString *)) : output_(output) {}

   protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) pending_.push_back((char)ch);
        if (ch == '\n') sync();
        return ch;
    }

    std::streamsize xsputn(const char *s, std::streamsize n) override {
        pending_.append(s, n);
        if (std::memchr(s, '\n', n)) sync();
        return n;
    }

    int sync() override {
        if (pending_.empty()) return 0;
        std::fwrite(pending_.data(), 1, pending_.size(), stdout);
        std::fflush(stdout);
        NSString *chunk = [[NSString alloc] initWithBytes:pending_.data()
                                                   length:pending_.size()
                                                 encoding:NSUTF8StringEncoding];
        if (chunk) output_(chunk);
        pending_.clear();
        return 0;
    }

   private:
    void (^output_)(NSString *);
    std::string pending_;
};

}  // namespace

@implementation JuzhenRunner

+ (NSString *)gpuName {
    return [NSString stringWithUTF8String:mpsDeviceName()];
}

+ (int)runExample:(JuzhenExample)example output:(void (^)(NSString *))output {
    CallbackBuf buf(output);
    std::streambuf *cout_buf = std::cout.rdbuf(&buf);
    std::streambuf *cerr_buf = std::cerr.rdbuf(&buf);

    int ret = 1;
    {
        Run run;
        std::cout << "GPU: " << mpsDeviceName() << " (Metal)" << std::endl << std::endl;
        try {
            switch (example) {
                case JuzhenExampleHelloWorld: ret = juzhen_helloworld(); break;
                case JuzhenExampleGEMM: ret = juzhen_demo_gemm(); break;
                case JuzhenExampleCIFARSpeed: ret = juzhen_cifar_speed(); break;
            }
        } catch (const std::exception &e) {
            std::cout << "Uncaught exception: " << e.what() << std::endl;
        }
    }
    if (example == JuzhenExampleCIFARSpeed) {
        // a hot device runs slower; "serious" or "critical" means throttled
        const char *states[] = {"nominal", "fair", "serious", "critical"};
        NSInteger state = NSProcessInfo.processInfo.thermalState;
        std::cout << "thermal state: " << (state >= 0 && state < 4 ? states[state] : "unknown") << std::endl;
    }

    std::cout.flush();
    std::cout.rdbuf(cout_buf);
    std::cerr.rdbuf(cerr_buf);
    return ret;
}

+ (BOOL)zeroFlowWithSource:(NSString *)source
                    target:(NSString *)target
                     iters:(NSInteger)iters
                      emit:(BOOL (^)(NSData *))emit {
    const std::string src = source.UTF8String, tgt = target.UTF8String;
    if (!valid_dataset(src) || !valid_dataset(tgt)) return NO;
    // Same bounds as POST /train in the demo's web server.
    const int n = (int)std::clamp<NSInteger>(iters, 200, 20000);

    Run run;
    stream_zero_flow(src, tgt, n, [&](const std::string &line) {
        return (bool)emit([NSData dataWithBytes:line.data() length:line.size()]);
    });
    return YES;
}

+ (nullable NSString *)cifarGenerateFromCheckpoint:(NSString *)path
                                             count:(NSInteger)count
                                             steps:(NSInteger)steps
                                              heun:(BOOL)heun
                                             every:(NSInteger)every
                                            epochs:(NSInteger *)epochs
                                             frame:(void (^)(NSInteger, NSData *, NSInteger, NSInteger))frame {
    Run run;
    int trained = 0;
    std::string error;
    const bool ok = juzhen_cifar_generate(
        path.UTF8String, (int)count, (int)steps, heun, (int)std::max<NSInteger>(every, 1), &trained,
        [&](int step, const std::vector<uint8_t> &rgba, int width, int height) {
            frame(step, [NSData dataWithBytes:rgba.data() length:rgba.size()], width, height);
        },
        &error);
    if (epochs) *epochs = trained;
    return ok ? nil : [NSString stringWithUTF8String:error.c_str()];
}

@end
