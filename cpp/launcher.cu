/**
 * @file launcher.cu
 * @brief Core Components
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#include "../cpp/juzhen.hpp"
#include "../external/xpu_info/xpu_info.hpp"

void PrintLogo() {
    // print ubuntu logo

std::cout <<"    _           _                " << std::endl;
std::cout <<"   (_)_   _ ___| |__   ___ _ __  " << std::endl;
std::cout <<"   | | | | |_  / '_ \\ / _ \\ '_ \\ "<< std::endl;
std::cout <<"   | | |_| |/ /| | | |  __/ | | |"<< std::endl;
std::cout <<"  _/ |\\__,_/___|_| |_|\\___|_| |_|"<< std::endl;
std::cout <<" |__/                            "<< std::endl;
std::cout <<"                                 "<< std::endl;
}

void PrintSeparationLine() {
    std::cout << "______________________________________________" << std::endl;
}

int main()
{
    //spdlog::set_pattern("%^[%l]%$, %v");
    //spdlog::set_level(spdlog::level::debug);

    std::cout << "Welcome to Juzhen (Beta)!"<< std::endl; 
    PrintLogo();
    std::cout << "song.liu@bristol.ac.uk."<< std::endl << std::endl; 
    std::cout << "Listing Computing Devices:" << std::endl;

    PrintSeparationLine();
    DisplayCPU();
#ifndef CPU_ONLY
    DisplayGPU();
#endif
    PrintSeparationLine();

    int ret = 1;

    {
        Memory<int> mdi;
        Memory<float> md;
        Memory<CUDAfloat> gpumd;
#ifndef CPU_ONLY
        // GPUMemoryDeleter gpumd;
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
#endif
        PrintSeparationLine();
        {
           ret = compute();
        }
        PrintSeparationLine();
#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));
#endif
    }
    
    return ret;
}