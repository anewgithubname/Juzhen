#include <iostream>
#include <vector>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// A small macro to check CUDA errors.
#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

// A small macro to check cuBLAS errors.
#define CUBLAS_CHECK(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

int main()
{
    // Matrix dimension (10,000)
    const int N = 10000;

    // Number of multiplications
    const int num_iterations = 10;

    // ~2 * N^3 floating ops for a naive NxN multiply
    // (each entry does N multiplies + N-1 additions => 2*N ops).
    double flops_per_multiply = 2.0 * double(N) * double(N) * double(N);

    // 1) Allocate host memory for A, B, C in column-major order.
    //    We'll store element (row, col) at index (row + col*N).
    //    Each matrix has N*N floats.
    size_t total_elems = size_t(N) * size_t(N);
    std::vector<float> h_A(total_elems);
    std::vector<float> h_B(total_elems);
    std::vector<float> h_C(total_elems, 0.0f);

    // 2) Fill A and B with random data in [0,1).
    std::mt19937 rng(0u);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_elems; i++) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    // 3) Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t bytes = total_elems * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // 4) Copy A and B from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // We'll create a cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Constants for SGEMM
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // We'll measure GPU time with CUDA events
    float time_ms[num_iterations];

    for (int iter = 0; iter < num_iterations; iter++) {
        // Optionally reset d_C to zero before each run
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));

        // Create/start CUDA events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        // 5) Call cublasSgemm
        //    By default, cublas expects column-major data.
        //    A, B, C are not transposed => C = alpha * A * B + beta * C
        //    leading dimension (lda, ldb, ldc) is N since each column has N elements.
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N,     // rows of C
                                 N,     // cols of C
                                 N,     // shared dimension
                                 &alpha,
                                 d_A, N, // d_A is NxN, leading dim = N
                                 d_B, N, // d_B is NxN, leading dim = N
                                 &beta,
                                 d_C, N)); // d_C is NxN, leading dim = N

        // 6) Stop timing
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        time_ms[iter] = elapsed_ms;

        // Convert time to seconds
        double elapsed_s = double(elapsed_ms) / 1000.0;

        // Compute TFLOPs
        double tflops = (flops_per_multiply / elapsed_s) / 1.0e12;

        std::cout << "Iteration " << (iter + 1)
                  << " took " << elapsed_s << " s, ~"
                  << tflops << " TFLOPs\n";
    }

    // 7) Compute average time & TFLOPs across iterations
    double total_time_s = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        total_time_s += (time_ms[i] / 1000.0);
    }
    double avg_time_s = total_time_s / num_iterations;
    double avg_tflops = (flops_per_multiply / avg_time_s) / 1.0e12;

    std::cout << "\nAverage time over " << num_iterations
              << " iterations: " << avg_time_s << " s\n";
    std::cout << "Approx. average performance: "
              << avg_tflops << " TFLOPs\n";

    // 8) Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
