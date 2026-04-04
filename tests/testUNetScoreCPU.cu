/**
 * @file testUNetScoreCPU.cu
 * @brief Test UNetScore with CPU backend
 */

#include "../ml/layer.hpp"
#include <iostream>

using namespace Juzhen;
using namespace std;

#if defined(CUDA) || defined(APPLE_SILICON)
int compute() {
    cout << "testUNetScoreCPU requires a CPU-only build." << endl;
    return 0;
}
#else

class UNetScore {
public:
    static constexpr int H = 32;
    static constexpr int W = 32;
    static constexpr int d = H * W;

    ConvLayer enc1;
    ConvLayer enc2;
    convtransLayer up1;
    ConvLayer dec1;
    ConvLayer head;

    Matrix<float> e1;
    Matrix<float> e2;
    Matrix<float> c_cat;

    explicit UNetScore(int bs)
                : enc1(bs,   3, 32, 32,  16, 3, 3, 1, 1, true),
                    enc2(bs,  16, 32, 32,  32, 3, 3, 1, 2, true),
                    up1 (bs,  32, 16, 16,  16, 4, 4, 1, 2, true),
                    dec1(bs,  32, 32, 32,  16, 3, 3, 1, 1, true),
                    head(bs,  16, 32, 32,   1, 3, 3, 1, 1, false),
                    e1("e1", 16 * d, bs),
                    e2("e2", 32 * 16 * 16, bs),
                    c_cat("cc", 32 * d, bs) {}

    const Matrix<float>& fwd(const Matrix<float>& inp) {
        cout << "enc1.eval..." << endl;
        enc1.eval(inp);
        e1 = enc1.value();
        cout << "enc2.eval..." << endl;
        enc2.eval(e1);
        e2 = enc2.value();
        cout << "up1.eval..." << endl;
        up1.eval(e2);
        cout << "vstack..." << endl;
        c_cat = vstack<float>({MatrixView<float>(up1.value()), MatrixView<float>(e1)});
        cout << "dec1.eval..." << endl;
        dec1.eval(c_cat);
        cout << "head.eval..." << endl;
        head.eval(dec1.value());
        return head.value();
    }
};

int compute() {
    global_rand_gen.seed(42);
    cout << "Testing UNetScore with CPU backend..." << endl;
    
    int batchsize = 2;
    cout << "Creating UNetScore..." << endl;
    UNetScore net(batchsize);
    cout << "UNetScore created" << endl;
    
    cout << "Creating input..." << endl;
    // Input should be 3-channel 32x32 + time embedding (2 channels) = 5 channels total
    Matrix<float> inp("inp", 5 * 32 * 32, batchsize);
    inp = Matrix<float>::randn(5 * 32 * 32, batchsize) * 0.1f;
    cout << "Input created: " << inp.num_row() << " x " << inp.num_col() << endl;
    
    cout << "Calling fwd..." << endl;
    const auto& out = net.fwd(inp);
    cout << "Fwd completed" << endl;
    cout << "Output shape: " << out.num_row() << " x " << out.num_col() << endl;
    
    cout << "Test passed!" << endl;
    return 0;
}

#endif
