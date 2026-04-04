#include <metal_stdlib>
using namespace metal;

kernel void zero_kernel(
    device float*       data  [[buffer(0)]],
    constant float&     value [[buffer(1)]],
    uint                id    [[thread_position_in_grid]]
)
{
    data[id] = value;
}

kernel void inplace_relu_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = v1[id] > 0.0 ? v1[id] : 0.0;
}

kernel void inplace_drelu_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = v1[id] > 0.0f ? 1.0f : 0.0f;
}

kernel void inplace_dtanh_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = 1 - tanh(v1[id]) * tanh(v1[id]);
}

kernel void inplace_tanh_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = tanh(v1[id]);
}

// Kernel to copy a matrix with optional transposition
kernel void matrix_product(device const float* A [[buffer(0)]], 
                        device float* B [[buffer(1)]], 
                        constant uint& rows [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant bool& transpose [[buffer(4)]],
                        uint2 thread_position_in_grid [[thread_position_in_grid]]) { // ✅ Fix: Use uint2

    uint rowA = thread_position_in_grid.y; // ✅ Fix: Correct access
    uint colA = thread_position_in_grid.x;

    if (rowA < rows && colA < cols) { // Prevent out-of-bounds access
        if (transpose) {
            B[colA * rows + rowA] *= A[rowA * cols + colA]; // Transposed copy
        } else {
            B[rowA * cols + colA] *= A[rowA * cols + colA]; // Direct copy
        }
    }
}

// Kernel to copy a matrix with optional transposition
kernel void matrix_add(device const float* A [[buffer(0)]], 
                        device float* B [[buffer(1)]], 
                        constant uint& rows [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant bool& transpose [[buffer(4)]],
                        constant float& a [[buffer(5)]],
                        constant float& b [[buffer(6)]],
                        uint2 thread_position_in_grid [[thread_position_in_grid]]) { // ✅ Fix: Use uint2

    uint rowA = thread_position_in_grid.y; // ✅ Fix: Correct access
    uint colA = thread_position_in_grid.x;

    if (rowA < rows && colA < cols) { // Prevent out-of-bounds access
        if (transpose) {
            B[colA * rows + rowA] = b*B[colA * rows + rowA] + a*A[rowA * cols + colA]; // Transposed copy
        } else {
            B[rowA * cols + colA] = b*B[rowA * cols + colA] + a*A[rowA * cols + colA]; // Direct copy
        }
    }
}

kernel void matrix_copy_block(device const float* src [[buffer(0)]],
                              device float* dst [[buffer(1)]],
                              constant uint& srcRows [[buffer(2)]],
                              constant uint& srcCols [[buffer(3)]],
                              constant bool& srcTranspose [[buffer(4)]],
                              constant uint& srcRowOffset [[buffer(5)]],
                              constant uint& srcColOffset [[buffer(6)]],
                              constant uint& dstRows [[buffer(7)]],
                              constant uint& dstCols [[buffer(8)]],
                              constant bool& dstTranspose [[buffer(9)]],
                              constant uint& dstRowOffset [[buffer(10)]],
                              constant uint& dstColOffset [[buffer(11)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint c = gid.x;
    const uint sr = r + srcRowOffset;
    const uint sc = c + srcColOffset;
    if (sr >= srcRows || sc >= srcCols) return;

    const uint srcPhysicalCols = srcTranspose ? srcRows : srcCols;
    const uint srcIdx = srcTranspose ? (sc * srcPhysicalCols + sr)
                                     : (sr * srcPhysicalCols + sc);

    const uint dr = r + dstRowOffset;
    const uint dc = c + dstColOffset;
    if (dr >= dstRows || dc >= dstCols) return;

    const uint dstPhysicalCols = dstTranspose ? dstRows : dstCols;
    const uint dstIdx = dstTranspose ? (dc * dstPhysicalCols + dr)
                                     : (dr * dstPhysicalCols + dc);

    dst[dstIdx] = src[srcIdx];
}

kernel void product_kernel(
    device float*       v1  [[buffer(0)]],
    device float*       v2  [[buffer(1)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v2[id] = v1[id] * v2[id];
}

kernel void ax_b_kernel(
    device float*       v1  [[buffer(0)]],
    device float*       v2  [[buffer(1)]],
    constant float&     a    [[buffer(2)]],
    constant float&     b    [[buffer(3)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v2[id] = a * v1[id] + b;
}

kernel void inplace_square_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = v1[id] * v1[id];
}

kernel void inplace_sqrt_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = sqrt(v1[id]);
}

kernel void inplace_exp_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = exp(v1[id]);
}

kernel void inplace_log_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = log(v1[id]);
}

kernel void inplace_elem_inv_kernel(
    device float*       v1  [[buffer(0)]],
    constant float&        l [[buffer(1)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = l / v1[id];
}

kernel void inplace_sigmoid_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = 1.0 / (1.0 + exp(-v1[id]));
}

kernel void sigmoid_kernel(
    device float*       v1  [[buffer(0)]],
    device float*       v2  [[buffer(1)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = 1.0 / (1.0 + exp(-v2[id]));
}

kernel void im2col_kernel(
    device const float* input [[buffer(0)]],
    device float* col [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& kH [[buffer(6)]],
    constant int& kW [[buffer(7)]],
    constant int& padH [[buffer(8)]],
    constant int& padW [[buffer(9)]],
    constant int& strideH [[buffer(10)]],
    constant int& strideW [[buffer(11)]],
    constant int& Hout [[buffer(12)]],
    constant int& Wout [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    const int K = C * kH * kW;
    const int P = Hout * Wout;
    const int PN = P * N;
    const int total = K * PN;
    if ((int)gid >= total) return;

    const int row = (int)gid / PN;
    const int patch = (int)gid % PN;

    const int n = patch / P;
    const int p = patch % P;
    const int oh = p / Wout;
    const int ow = p % Wout;

    const int ci = row / (kH * kW);
    const int rem = row % (kH * kW);
    const int kh = rem / kW;
    const int kw = rem % kW;

    const int ih = oh * strideH - padH + kh;
    const int iw = ow * strideW - padW + kw;

    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        const int feature = (ci * H + ih) * W + iw;
        const int input_idx = feature * N + n;
        v = input[input_idx];
    }
    col[row * PN + patch] = v;
}

kernel void col2im_kernel(
    device const float* col [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& kH [[buffer(6)]],
    constant int& kW [[buffer(7)]],
    constant int& padH [[buffer(8)]],
    constant int& padW [[buffer(9)]],
    constant int& strideH [[buffer(10)]],
    constant int& strideW [[buffer(11)]],
    constant int& Hout [[buffer(12)]],
    constant int& Wout [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    const int total = N * C * H * W;
    if ((int)gid >= total) return;

    const int n = (int)gid % N;
    const int t0 = (int)gid / N;
    const int iw = t0 % W;
    const int t1 = t0 / W;
    const int ih = t1 % H;
    const int ci = t1 / H;

    const int P = Hout * Wout;
    const int PN = P * N;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        const int oh_num = ih + padH - kh;
        if (oh_num < 0 || (oh_num % strideH) != 0) continue;
        const int oh = oh_num / strideH;
        if (oh < 0 || oh >= Hout) continue;

        for (int kw = 0; kw < kW; ++kw) {
            const int ow_num = iw + padW - kw;
            if (ow_num < 0 || (ow_num % strideW) != 0) continue;
            const int ow = ow_num / strideW;
            if (ow < 0 || ow >= Wout) continue;

            const int row = (ci * kH + kh) * kW + kw;
            const int patch = n * P + oh * Wout + ow;
            acc += col[row * PN + patch];
        }
    }

    output[gid] = acc;
}

kernel void pack_feature_map_2d_kernel(
    device const float* featureMap [[buffer(0)]],
    device float* packed [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& P [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const int total = N * C * P;
    if ((int)gid >= total) return;

    const int n = (int)gid % N;
    const int t0 = (int)gid / N;
    const int p = t0 % P;
    const int c = t0 / P;

    const int src_row = c * P + p;
    const int dst_col = n * P + p;
    packed[c * (P * N) + dst_col] = featureMap[src_row * N + n];
}

kernel void conv2d_output_add_bias_kernel(
    device const float* y2d [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& Cout [[buffer(4)]],
    constant int& Hout [[buffer(5)]],
    constant int& Wout [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const int P = Hout * Wout;
    const int total = N * Cout * P;
    if ((int)gid >= total) return;

    const int n = (int)gid % N;
    const int t0 = (int)gid / N;
    const int p = t0 % P;
    const int co = t0 / P;

    const int y_col = n * P + p;
    const float v = y2d[co * (P * N) + y_col] + bias[co];

    const int out_row = co * P + p;
    output[out_row * N + n] = v;
}
