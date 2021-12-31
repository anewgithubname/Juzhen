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
typedef std::chrono::high_resolution_clock Clock;
#include <numeric> 
#include <random>
#include <algorithm>
#include <iostream>

#ifdef INTEL_MKL //do we use Intel mkL special funcs? doesn't seem to have much impact on perf.
#include <mkl.h>
#else
#include <cblas.h>
#endif

typedef std::vector<int> idxlist;

inline idxlist seq(int start, int end) {
    idxlist ret;
    for (int i = start; i < end; i++) {
        ret.push_back(i);
    }
    return ret;
}

inline idxlist seq(int end){
    return seq(0, end);
}

inline idxlist shuffle(int start, int end){
    using namespace std;
    idxlist ret = seq(start, end);
    random_device rd;
    mt19937 g(rd());
    shuffle(ret.begin(), ret.end(), g);
    return ret;
}

inline idxlist shuffle(int end){
    return shuffle(0, end);
}

inline int rand_number(){
    return rand() % 10;
}

inline double time_in_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0;
}

class Profiler {
    public:
        Profiler(){
            t1 = Clock::now();
            started = true;
        };
        Profiler(std::string s) {
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
                // std::cout << s << std::endl << "Time: " << time_in_ms(t1, t2) << " ms" << std::endl << std::endl;
                started = false;
            }
        }
        ~Profiler() {
            end();
            std::cout <<s <<std::endl << "Time: " << cumulative_time << " ms" << std::endl << std::endl;
        }
    private:
        Clock::time_point t1;
        Clock::time_point t2;
        double cumulative_time = 0;
        std::string s="";
        bool started = false;
};


#ifndef INTEL_MKL
//CBLAS declarations
extern "C"
{
    // LU decomoposition of a general matrix
    void sgetrf_(int *M, int *N, float *A, int *lda, int *IPIV, int *INFO);
    void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

    // generate inverse of a matrix given its LU decomposition
    void sgetri_(int *N, float *A, int *lda, int *IPIV, float *WORK, int *lwork, int *INFO);
    void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork, int *INFO);
}
#endif

//overloading CBLAS interface for different types. 
inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, float *A, int lda, float *x, int ldx, float beta, float *y, int ldy){
    cblas_sgemv(CblasColMajor, transM, m, n, alpha, A, lda, x, ldx, beta, y, ldy);
}

inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, double *A, int lda, double *x, int ldx, float beta, double *y, int ldy){
    cblas_dgemv(CblasColMajor, transM, m, n, alpha, A, lda, x, ldx, beta, y, ldy);
}

inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, int *A, int lda, int *x, int ldx, float beta, int *y, int ldy){
    std::cout << "gemv: int* not implemented" << std::endl;
}

inline void gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int nr, int nc, int nk, 
            float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc){
    cblas_sgemm(CblasColMajor, transA, transB,
                nr, nc, nk, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int nr, int nc, int nk, 
            float alpha, double *A, int lda, double *B, int ldb, float beta, double *C, int ldc){
    cblas_dgemm(CblasColMajor, transA, transB,
                nr, nc, nk, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int nr, int nc, int nk, 
            float alpha, int *A, int lda, int *B, int ldb, float beta, int *C, int ldc){
    std::cout << "CBLAS gemm not implemented for int!" << std::endl;
}

inline void getrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info){
    sgetrf_(m, n, a, lda, ipiv, info);
}

inline void getrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info){
    dgetrf_(m, n, a, lda, ipiv, info);
}

inline void getrf_(int *m, int *n, int *a, int *lda, int *ipiv, int *info){
    std::cout << "getrf not implemented for int!" << std::endl;
}

inline void getri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info){
    sgetri_(n, a, lda, ipiv, work, lwork, info);
}

inline void getri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info){
    dgetri_(n, a, lda, ipiv, work, lwork, info);
}

inline void getri_(int *n, int *a, int *lda, int *ipiv, int *work, int *lwork, int *info){
    std::cout << "getri not implemented for int!" << std::endl;
}

#ifdef INTEL_MKL
inline void omatadd(char transA, char transB, int m, int n, float alpha, float *A, int lda, float beta, float *B, int ldb, float *C, int ldc){
    mkl_somatadd('C',transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline void omatadd(char transA, char transB, int m, int n, double alpha, double *A, int lda, double beta, double *B, int ldb, double *C, int ldc){
    mkl_domatadd('C',transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline void omatadd(char transA, char transB, int m, int n, int alpha, int *A, int lda, int beta, int *B, int ldb, int *C, int ldc){
    std::cout<<"mkl_omatadd: int1 not implemented"<<std::endl;
}
#endif

#endif