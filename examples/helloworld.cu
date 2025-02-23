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
    //spdlog::set_level(spdlog::level::debug);
    global_rand_gen.seed(0);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif

    {
        auto A = M::rand(3, 4);
        std::cout << A << std::endl;
    
        auto B = M::rand(3, 3);
        std::cout << B << std::endl;
        
        auto AB = vstack<float>({A.columns(0, 2).T(), B.columns(0,2).T()}); 
        std::cout << AB << std::endl;

        auto C = M::randn(4, 3);
        std::cout << C << std::endl;
        auto D = elemwise([=](float e) {return e-1; }, C);
        std::cout << D << std::endl;

        auto E = randn_like(D);
        std::cout << E << std::endl;

        auto F = randn_like(E);
        try
        {
            std::cout << E / F.T() << std::endl; // should trigger an error
        }
        catch(const std::exception& e)
        {
            std::cout << "Caught exception: " << e.what() << std::endl;
            //pause until user presses a key
            std::cout << "Press Enter to continue...";
            std::cin.get();
        }

        auto G = M::randn(3, 1000);
        std::cout << mean(G, 1) << std::endl;
        std::cout << stddev(G, 1) << std::endl;
        std::cout << cov(G, 1) << std::endl;

        auto H = M::randn(1000, 1);
        plot_histogram(H.data(), H.num_row() * H.num_col(), 10);

        for (int i = 0; i < 100; i++)
        {
            //pause for 1 second
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            display_progress_bar(i/100.0, "Processing...");
        }
        std::cout << std::endl; 

        //write matrices to files
        std::fstream fout("A.matrix");
        fout << A;
        fout.close();
        

        // auto C = exp(A) + B;
        // std::cout << C << std::endl;

        // auto C2 = A + exp(B);
        // std::cout << C2 << std::endl;

        // HLINE;
        // auto C3 = exp(A+B) + exp(B);
        // std::cout << C3 << std::endl;

    }
  
    return 0;
}
