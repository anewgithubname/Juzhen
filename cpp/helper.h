/**
 * @file helper.h
 * @brief declerations of some helper functions. 
 * @author Song Liu (song.liu@bristol.ac.uk)
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

#ifndef HELPER_HPP
#define HELPER_HPP

#include <chrono>
#include <array>
using namespace std;
typedef chrono::high_resolution_clock Clock;
#include <numeric> 
#include <random>
#include <algorithm>
#include <iostream>

typedef vector<int> idxlist;

int time_in_ms(Clock::time_point start, Clock::time_point end);
idxlist seq(int start, int end);
idxlist seq(int end);
idxlist shuffle(int start, int end);
idxlist shuffle(int end);
int rand_number();

#ifndef INTEL_MKL
//CBLAS declarations
extern "C"
{
    // LU decomoposition of a general matrix
    void sgetrf_(int *M, int *N, float *A, int *lda, int *IPIV, int *INFO);

    // generate inverse of a matrix given its LU decomposition
    void sgetri_(int *N, float *A, int *lda, int *IPIV, float *WORK, int *lwork, int *INFO);
}
#endif

class Profiler {
    public:
        Profiler(){
            t1 = Clock::now();
            started = true;
        };
        Profiler(string s) {
            this->s = s;
            t1 = Clock::now();
            started = true;
        }
        void start(){
            t1 = Clock::now();
            started = true;
        }
        void end(){
            if(started){
                t2 = Clock::now();
                cumulative_time += time_in_ms(t1, t2);
                // cout << s << endl << "Time: " << time_in_ms(t1, t2) << " ms" << endl << endl;
                started = false;
            }
        }
        ~Profiler() {
            end();
            cout <<s <<endl << "Time: " << cumulative_time << " ms" << endl << endl;
        }
    private:
        Clock::time_point t1;
        Clock::time_point t2;
        int cumulative_time = 0;
        string s="";
        bool started = false;
};

#endif