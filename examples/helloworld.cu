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

// overriding the default printing function for matrices
std::ostream& operator<<(std::ostream& os, const M& M) {
    using namespace std;
    // write obj to stream
    for (size_t i = 0; i < M.num_row(); i++) {
        os << endl;
        for (size_t j = 0; j < M.num_col(); j++) {
            os << M.elem(i, j) << " ";
        }
    }
    return os;
}

#define HLINE std::cout << "--------------------------------" << std::endl

int compute() {
    //  spdlog::set_level(spdlog::level::debug);
    global_rand_gen.seed(0);
#ifdef CUDA
    GPUSampler sampler(1);
#endif

    {
        std::cout << "The following code demonstrate how to use the Juzhen library." << std::endl;
        std::cout << "all compuitations run on the GPU." << std::endl;

        M A("A", {{1, 2, 3}, {4, 5, 6}});
        std::cout << "A = " << A << std::endl;

        M B("B", {{4, 5, 6}, {7, 8, 9}});
        std::cout << "B = " << B << std::endl;

        auto AB = vstack<float>({A, B});
        std::cout << "AB = [A; B] = " << AB << std::endl;

        HLINE;

        std::cout << "A + B = " << A + B << std::endl;
        std::cout << "A - B = " << A - B << std::endl;
        std::cout << "A + 1 = " << A + 1 << std::endl;
        std::cout << "A - 1 = " << A - 1 << std::endl;
        std::cout << "A / 1.5 = " << A / 1.5 << std::endl;
        std::cout << "A / B = " << A / B << std::endl;

        try {
            std::cout << "A / B^T = " << A / B.T() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Caught exception: " << e.what() << std::endl;
        }

        HLINE;

        try {
            std::cout << "A * B = " << A * B << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Caught exception: " << e.what() << std::endl;
        }

        HLINE;
        std::cout << "A * B^T = " << A * B.T() << std::endl;
    }

    return 0;
}
