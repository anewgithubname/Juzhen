/**
 * @file helloworld.cu
 * @brief Hello world example
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2022 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

 */

 #include <fstream>
 #include <thread>
 #include "../cpp/juzhen.hpp"
 #include "../ml/plotting.hpp"
 
 #define HLINE std::cout << "--------------------------------" << std::endl
 
 int compute() {
    //  spdlog::set_level(spdlog::level::debug);
     global_rand_gen.seed(0);
 #ifdef CUDA
     GPUSampler sampler(1);
 #endif
 
     {

        const int DIM = 10000;
        
        HLINE;
        std::cout << "This program is for benchmarking the matrix multiplication performance." << std::endl;
        std::cout << "It will run 10 times of GEMM and print the time in milliseconds." << std::endl;
        std::cout << "The matrix size is " << DIM << " x " << DIM << std::endl;
        std::cout << "The TFLPOS is the number of Tera Floating Point Operations per Second." << std::endl;
        std::cout << "The TFLPOS is calculated as 2 * DIM^3 / duration / 1e9" << std::endl;


 #ifdef APPLE_SILICON
         {
            auto A1 = Matrix<MPSfloat>::randn(DIM, DIM);
            auto A2 = Matrix<MPSfloat>::randn(DIM, DIM);
            Matrix<MPSfloat> A3 = Matrix<MPSfloat>::zeros(DIM, DIM);
             std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
             for (int i = 0; i < 10; i++)
             {
                 A3 += A1 * A2/DIM;
                 std::cout << "."; std::cout.flush();
             }
             A3.to_host().slice(0,5,0,5);
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << std::endl << "Duration: " << duration << " ms" << std::endl;
             std::cout << "\033[34mMPS GEMM TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << "\033[0m" << std::endl;
         }
 #endif
 
 #ifdef CUDA
         {
            HLINE;
            auto A1 = Matrix<CUDAfloat>::randn(DIM, DIM);
            auto A2 = Matrix<CUDAfloat>::randn(DIM, DIM);
            Matrix<CUDAfloat> A3 = Matrix<CUDAfloat>::zeros(DIM, DIM);
             std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
             for (int i = 0; i < 10; i++)
             {
                 A3 += A1 * A2 / DIM;
                 std::cout << "."; 
             }
             A3.slice(0,5,0,5).to_host();
             
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << std::endl << "Duration: " << duration << " ms" << std::endl;
             std::cout << "\033[34mCUDA GEMM TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << "\033[0m" << std::endl;
         }
 #endif

         {
             HLINE;
             auto A1C = Matrix<float>::randn(DIM, DIM);
             auto A2C = Matrix<float>::randn(DIM, DIM);
             Matrix<float> A3 = Matrix<float>::zeros(DIM, DIM);
             std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
             for (int i = 0; i < 10; i++)
             {
                 A3 += A1C * A2C/DIM;
                 std::cout << "."; std::cout.flush();
             }
             A3.slice(0,5,0,5);
             
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << std::endl << "Duration: " << duration << " ms" << std::endl;
             std::cout << "\033[34mCPU GEMM TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << "\033[0m" << std::endl;
         }
        
     }

   
     return 0;
 }
 