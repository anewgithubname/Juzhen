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

kernel void inplace_tanh_kernel(
    device float*       v1  [[buffer(0)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = tanh(v1[id]);
}

kernel void tanh_kernel(
    device float*       v1  [[buffer(0)]],
    device float*       v2  [[buffer(1)]],
    uint                id    [[thread_position_in_grid]]
)
{
    v1[id] = tanh(v2[id]);
}