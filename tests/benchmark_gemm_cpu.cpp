#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// cBLAS interface is in C, so we need to wrap it in `extern "C"`.
extern "C" {
#include <cblas.h>
}

int main()
{
    // Matrix dimension
    const int N = 10000;

    // We'll do 10 multiplications
    const int num_iterations = 10;

    // 1) Allocate memory for matrices A, B, C
    //    A, B, C are N*N in size.
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    // 2) Initialize A and B with random data in [0,1).
    std::mt19937 rng(0u);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N*N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // For an N×N multiply, the naive floating ops count is ~2*N^3
    // (each element of the result involves N multiplies + N-1 adds).
    double flops_per_multiply = 2.0 * double(N) * double(N) * double(N);

    // 3) Perform the multiplication num_iterations times
    std::vector<double> iteration_times;
    iteration_times.reserve(num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        // Re-initialize C to zero each time (optional)
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // 4) Use cBLAS sgemm to multiply A and B => C
        //    C = 1.0f * (A × B) + 0.0f * C
        cblas_sgemm(
            CblasRowMajor,   // data layout is row-major
            CblasNoTrans,    // A is not transposed
            CblasNoTrans,    // B is not transposed
            N,               // M: number of rows of A and C
            N,               // N: number of columns of B and C
            N,               // K: number of columns of A / rows of B
            1.0f,            // alpha
            A.data(), N,     // A pointer, leading dimension of A is N
            B.data(), N,     // B pointer, leading dimension of B is N
            0.0f,            // beta
            C.data(), N      // C pointer, leading dimension of C is N
        );

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        iteration_times.push_back(seconds);

        // 5) Calculate TFLOPs
        double tflops = (flops_per_multiply / seconds) / 1.0e12;

        std::cout << "Iteration " << (iter + 1) 
                  << " took " << seconds << " s, ~" 
                  << tflops << " TFLOPs\n";
    }

    // 6) Compute average time & TFLOPs across iterations
    double total_time = 0.0;
    for (auto t : iteration_times) {
        total_time += t;
    }
    double average_time = total_time / num_iterations;
    double average_tflops = (flops_per_multiply / average_time) / 1.0e12;

    std::cout << "\nAverage time over " << num_iterations 
              << " iterations: " << average_time << " s\n";
    std::cout << "Approx. average performance: " 
              << average_tflops << " TFLOPs\n";

    return 0;
}
