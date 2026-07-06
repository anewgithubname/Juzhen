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

// Elementwise kernels read src and write dst so out-of-place ops need a
// single dispatch (no copy pass); in-place callers bind the same buffer to
// both slots.
kernel void relu_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    const float v = src[id];
    dst[id] = v > 0.0f ? v : 0.0f;
}

kernel void drelu_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = src[id] > 0.0f ? 1.0f : 0.0f;
}

kernel void dtanh_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    const float t = tanh(src[id]);
    dst[id] = 1.0f - t * t;
}

kernel void tanh_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = tanh(src[id]);
}

// C = A(optionally transposed) .* B, where B and C share the same layout.
// Grid spans A's physical (rows x cols); in-place callers bind B == C.
kernel void matrix_product(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& cols [[buffer(4)]],
                        constant bool& transpose [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {

    const uint r = gid.y;
    const uint c = gid.x;

    if (r < rows && c < cols) { // Prevent out-of-bounds access
        const uint srcIdx = r * cols + c;
        const uint dstIdx = transpose ? (c * rows + r) : srcIdx;
        C[dstIdx] = A[srcIdx] * B[dstIdx];
    }
}

// C = b*B + a*A(optionally transposed), where B and C share the same layout.
// Grid spans A's physical (rows x cols); in-place callers bind B == C.
kernel void matrix_add(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& cols [[buffer(4)]],
                        constant bool& transpose [[buffer(5)]],
                        constant float& a [[buffer(6)]],
                        constant float& b [[buffer(7)]],
                        uint2 gid [[thread_position_in_grid]]) {

    const uint r = gid.y;
    const uint c = gid.x;

    if (r < rows && c < cols) { // Prevent out-of-bounds access
        const uint srcIdx = r * cols + c;
        const uint dstIdx = transpose ? (c * rows + r) : srcIdx;
        C[dstIdx] = b * B[dstIdx] + a * A[srcIdx];
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

kernel void square_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    const float v = src[id];
    dst[id] = v * v;
}

kernel void sqrt_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = sqrt(src[id]);
}

kernel void exp_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = exp(src[id]);
}

kernel void log_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = log(src[id]);
}

kernel void elem_inv_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant float&     l   [[buffer(2)]],
    uint                id  [[thread_position_in_grid]]
)
{
    dst[id] = l / src[id];
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

// ── fused training kernels (row-major (dim, N) matrices) ───────────────────
// One thread per column c. At each loop step i the threadgroup touches the
// contiguous row segment x[i*N + c..], so accesses coalesce.

// LayerNorm forward: y[:,c] = gamma ⊙ (x[:,c]-mu_c)*inv_c + beta, storing
// xhat and inv_std for the backward pass. Replaces ~8 generic matrix ops.
kernel void layernorm_forward_rm_kernel(
    device const float* x     [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float* y           [[buffer(3)]],
    device float* xhat        [[buffer(4)]],
    device float* inv_std     [[buffer(5)]],
    constant uint& dim        [[buffer(6)]],
    constant uint& N          [[buffer(7)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= N) return;

    float mu = 0.0f;
    for (uint i = 0; i < dim; ++i) mu += x[i * N + c];
    mu /= dim;

    float var = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        const float d = x[i * N + c] - mu;
        var += d * d;
    }
    var /= dim;

    const float inv = rsqrt(var + 1e-5f);
    inv_std[c] = inv;
    for (uint i = 0; i < dim; ++i) {
        const float xh = (x[i * N + c] - mu) * inv;
        xhat[i * N + c] = xh;
        y[i * N + c] = gamma[i] * xh + beta[i];
    }
}

// LayerNorm input-gradient (dxhat = dy ⊙ gamma):
//   dx = inv_std ⊙ (dxhat - mean(dxhat) - xhat ⊙ mean(dxhat ⊙ xhat))
kernel void layernorm_backward_rm_kernel(
    device const float* dy      [[buffer(0)]],
    device const float* gamma   [[buffer(1)]],
    device const float* xhat    [[buffer(2)]],
    device const float* inv_std [[buffer(3)]],
    device float* dx            [[buffer(4)]],
    constant uint& dim          [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= N) return;

    float m1 = 0.0f, m2 = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        const float dxh = gamma[i] * dy[i * N + c];
        m1 += dxh;
        m2 += dxh * xhat[i * N + c];
    }
    m1 /= dim;
    m2 /= dim;

    const float inv = inv_std[c];
    for (uint i = 0; i < dim; ++i) {
        const float dxh = gamma[i] * dy[i * N + c];
        dx[i * N + c] = inv * (dxh - m1 - xhat[i * N + c] * m2);
    }
}

// y[i, c] += b[i]: broadcast bias add over columns, replacing the
// "+ b * ones(1, N)" outer-product GEMM and its temporary.
kernel void add_bias_rm_kernel(
    device float* y       [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& N      [[buffer(2)]],
    constant uint& total  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    y[gid] += b[gid / N];
}

// Fused Adam step (elementwise, layout-agnostic): updates m, v in place and
// rewrites g with the step alpha * m_hat / (sqrt(v_hat) + eps). bc1/bc2 are
// the host-computed bias corrections 1/(1-beta^t).
kernel void adam_update_kernel(
    device float* g       [[buffer(0)]],
    device float* m       [[buffer(1)]],
    device float* v       [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant float& beta1 [[buffer(4)]],
    constant float& beta2 [[buffer(5)]],
    constant float& eps   [[buffer(6)]],
    constant float& bc1   [[buffer(7)]],
    constant float& bc2   [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    const float gi = g[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    const float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    g[i] = alpha * (mi * bc1) / (sqrt(vi * bc2) + eps);
}

// ── batched multi-head attention helpers ───────────────────────────────────
// The packed layout stores one (seq x d_h) row-major matrix per (head, batch)
// block, concatenated: packed[((h*B + b)*seq + s)*d_h + e] = X[h*d_h + e, b*seq + s]
// where X is a (d_k x B*seq) row-major projection (Q, K or V). Rows of the
// packed matrices are X^T restricted to one head/sequence block, which is
// what the batched GEMMs below consume.

kernel void attn_pack_kernel(
    device const float* src [[buffer(0)]],   // (d_k x B*seq) row-major
    device float* dst [[buffer(1)]],         // B*H packed (seq x d_h) matrices
    constant uint& d_h  [[buffer(2)]],
    constant uint& H    [[buffer(3)]],
    constant uint& seq  [[buffer(4)]],
    constant uint& B    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = H * B * seq * d_h;
    if (gid >= total) return;
    const uint e = gid % d_h;
    uint t = gid / d_h;
    const uint s = t % seq;
    t /= seq;
    const uint b = t % B;
    const uint h = t / B;
    dst[gid] = src[(h * d_h + e) * (B * seq) + b * seq + s];
}

kernel void attn_unpack_kernel(
    device const float* src [[buffer(0)]],   // B*H packed (seq x d_h) matrices
    device float* dst [[buffer(1)]],         // (d_k x B*seq) row-major
    constant uint& d_h  [[buffer(2)]],
    constant uint& H    [[buffer(3)]],
    constant uint& seq  [[buffer(4)]],
    constant uint& B    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = H * B * seq * d_h;
    if (gid >= total) return;
    const uint e = gid % d_h;
    uint t = gid / d_h;
    const uint s = t % seq;
    t /= seq;
    const uint b = t % B;
    const uint h = t / B;
    dst[(h * d_h + e) * (B * seq) + b * seq + s] = src[gid];
}

// Causal mask over a batch of (n x n) row-major scores matrices.
kernel void causal_mask_batched_kernel(
    device float*   s        [[buffer(0)]],
    constant uint&  n        [[buffer(1)]],
    constant uint&  batch    [[buffer(2)]],
    constant float& mask_val [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = batch * n * n;
    if (gid >= total) return;
    const uint l = gid % (n * n);
    const uint q = l / n;   // query row
    const uint k = l % n;   // key col
    if (k > q) s[gid] = mask_val;
}

// Row-wise softmax over a batch of (n x n) row-major matrices: one thread per
// row; rows are contiguous, so all accesses are sequential. In-place (x == y)
// is safe since each thread owns its row.
kernel void softmax_rows_batched_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& rows [[buffer(3)]],   // batch * n
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * n;

    float m = -1e30f;
    for (uint k = 0; k < n; ++k) {
        const float v = x[base + k];
        m = m > v ? m : v;
    }

    float ssum = 0.0f;
    for (uint k = 0; k < n; ++k) {
        const float e = exp(x[base + k] - m);
        y[base + k] = e;
        ssum += e;
    }

    const float inv = 1.0f / (ssum + 1e-12f);
    for (uint k = 0; k < n; ++k) {
        y[base + k] *= inv;
    }
}

// Backward of the batched row-wise softmax:
//   dS[q,k] = A[q,k] * (dA[q,k] - sum_j A[q,j]*dA[q,j]) * scale
kernel void softmax_backward_rows_batched_kernel(
    device const float* A  [[buffer(0)]],
    device const float* dA [[buffer(1)]],
    device float* dS       [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    constant uint& rows    [[buffer(4)]],   // batch * n
    constant float& scale  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * n;

    float rs = 0.0f;
    for (uint k = 0; k < n; ++k) {
        rs += A[base + k] * dA[base + k];
    }
    for (uint k = 0; k < n; ++k) {
        dS[base + k] = A[base + k] * (dA[base + k] - rs) * scale;
    }
}

// Causal attention mask on a square (n x n) scores matrix indexed
// [query row, key col]: keys in the future (col > row) are set to mask_val.
// `transpose` is the matrix's physical-layout flag.
kernel void causal_mask_kernel(
    device float*   s         [[buffer(0)]],
    constant uint&  n         [[buffer(1)]],
    constant bool&  transpose [[buffer(2)]],
    constant float& mask_val  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint r = gid.y;
    const uint c = gid.x;
    if (r >= n || c >= n || c <= r) return;
    s[transpose ? (c * n + r) : (r * n + c)] = mask_val;
}

// Fused column-wise softmax: one thread handles one logical column of a
// (rows x cols) matrix. `transpose` describes the source's physical layout
// (Matrix<MPSfloat> stores row-major with an optional transpose flag);
// dst is always written plain row-major (rows x cols).
kernel void softmax_col_kernel(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant bool& transpose [[buffer(4)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= cols) return;

    // logical elem(i, j) lives at src[base + i*stride]
    const uint base   = transpose ? j * rows : j;
    const uint stride = transpose ? 1u : cols;

    float m = -1e30f;
    for (uint i = 0; i < rows; ++i) {
        const float v = src[base + i * stride];
        m = m > v ? m : v;
    }

    float s = 0.0f;
    for (uint i = 0; i < rows; ++i) {
        const float e = exp(src[base + i * stride] - m);
        dst[i * cols + j] = e;
        s += e;
    }

    const float inv = 1.0f / (s + 1e-12f);
    for (uint i = 0; i < rows; ++i) {
        dst[i * cols + j] *= inv;
    }
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
