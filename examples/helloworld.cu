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
         // auto A = M::rand(3, 4);
         // std::cout << A << std::endl;
     
         // auto B = M::rand(3, 3);
         // std::cout << B << std::endl;
         
         // auto AB = vstack<float>({A.columns(0, 2).T(), B.columns(0,2).T()}); 
         // std::cout << AB << std::endl;
 
         // auto C = M::randn(4, 3);
         // std::cout << C << std::endl;
         // auto D = elemwise([=](float e) {return e-1; }, C);
         // std::cout << D << std::endl;
 
         // auto E = randn_like(D);
         // std::cout << E << std::endl;
 
         // auto F = randn_like(E);
         // try
         // {
         //     std::cout << E / F.T() << std::endl; // should trigger an error
         // }
         // catch(const std::exception& e)
         // {
         //     std::cout << "Caught exception: " << e.what() << std::endl;
         //     //pause until user presses a key
         //     std::cout << "Press Enter to continue...";
         //     std::cin.get();
         // }
 
         // auto G = M::randn(3, 1000);
         // std::cout << mean(G, 1) << std::endl;
         // std::cout << stddev(G, 1) << std::endl;
         // std::cout << cov(G, 1) << std::endl;
 
         // auto H = M::randn(1000, 1);
         // plot_histogram(H.data(), H.num_row() * H.num_col(), 10);
 
         // for (int i = 0; i < 100; i++)
         // {
         //     //pause for 1 second
         //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
         //     display_progress_bar(i/100.0, "Processing...");
         // }
         // std::cout << std::endl; 
 
         // //write matrices to files
         // std::fstream fout("A.matrix");
         // fout << A;
         // fout.close();
 #define DIM 5233

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
             std::cout << A3.to_host().slice(0,5,0,5)<< std::endl;
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << "Duration: " << duration << " ms" << std::endl;
             std::cout << "TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << std::endl;
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
             std::cout << A3.slice(0,5,0,5)<< std::endl;
             
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << "Duration: " << duration << " ms" << std::endl;
             std::cout << "TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << std::endl;
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
             std::cout << A3.slice(0,5,0,5)<< std::endl;
             
             std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
             std::cout << "Duration: " << duration << " ms" << std::endl;
             std::cout << "TFLPOS: " << (2.0 * DIM * DIM * DIM) * 10 / duration / 1e9 << std::endl;
         }
         
        
        // auto A = Matrix<MPSfloat>::randn(5, 2);
        // auto B = Matrix<MPSfloat>::randn(5, 2);

        // // std::cout << A.to_host() << std::endl;
        // // std::cout << B.to_host() << std::endl;

        // // std::cout << log(exp(hadmd(A+B, B+A)) + exp(A)).to_host().slice(0,5,0,5) << std::endl;
        // std::cout << (2.0/(A)) << std::endl;
        // HLINE;
        // auto Ah = A.to_host();
        // auto Bh = B.to_host();
        // // std::cout << log(exp(hadmd(Ah+Bh, Bh+Ah)) + exp(Ah)).slice(0,5,0,5) << std::endl;
        // std::cout << (2.0/(Ah)) << std::endl;

        // auto A = Matrix<MPSfloat>::randn(3, 2);
        // std::cout << A << std::endl;
        // std::cout << "sum along row: " << std::endl;
        // std::cout << sum(A, 1).to_host() - sum(A.to_host(), 1) << std::endl;

        // // auto B = Matrix<MPSfloat>::randn(3, 2);
        // // std::cout << B << std::endl;

        // // auto C = Matrix<MPSfloat>::zeros(2, 3);
        // // C += B.T();
        // // std::cout << C.get_transpose() << std::endl;


        // auto B = Matrix<MPSfloat>::randn(60000, 10000);
        // auto C = topk(B.T(), 15, 1);

        // std::cout << "B: " << std::endl;
        // std::cout << B.T().to_host().slice(0,5,0,5) << std::endl;

        // std::cout << "topk: " << std::endl;
        // std::cout << C.slice(0,5,0,5) << std::endl;

        // auto D = Matrix<MPSfloat>::randn(3, 2) + 5;
        // std::cout << square(D.to_host()) << std::endl;
        // std::cout << square((D+0.0)) << std::endl;

        // D = Matrix<MPSfloat>::randn(3, 2) + 5;
        // D.zeros();
        // std::cout << D << std::endl;
        
     }

   
     return 0;
 }
 