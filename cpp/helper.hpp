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

#ifndef LOGGING_OFF
#ifndef SPDLOG_COMPILED_LIB
#define SPDLOG_COMPILED_LIB
#endif
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#endif
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include <string>
#include <random>
#include <list>
#include <algorithm>
#include <memory>
#include <iostream>
#ifdef INTEL_MKL //do we use Intel mkL special funcs? doesn't seem to have much impact on perf.
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <boost/serialization/strong_typedef.hpp>
#include <boost/type_index.hpp>
#define datatype(D) boost::typeindex::type_id<D>().pretty_name() 

BOOST_STRONG_TYPEDEF(float, CUDAfloat)
BOOST_STRONG_TYPEDEF(float, MPSfloat)

typedef std::vector<size_t> idxlist;

static std::mt19937 global_rand_gen; // global random number generator

inline idxlist seq(size_t start, size_t end) {
    idxlist ret;
    for (size_t i = start; i < end; i++) {
        ret.push_back(i);
    }
    return ret;
}

inline idxlist seq(size_t end){
    return seq(0, end);
}

inline idxlist shuffle(int start, int end){
    using namespace std;
    idxlist ret = seq(start, end);
    shuffle(ret.begin(), ret.end(), global_rand_gen);
    return ret;
}

inline idxlist shuffle(int end){
    return shuffle(0, end);
}

inline double time_in_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1000000.0;
}

//for loggging
#ifndef LOGGING_OFF
#define LOG_INFO(...) spdlog::info(__VA_ARGS__)
#define LOG_WARN(...) spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
#else
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_DEBUG(...)
#endif
//for force quiting the program
#define ERROR_OUT exit(1)

/**
 * @brief simple profiler class, will only print out the cumulative time. 
 **/
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
                started = false;
            }
        }
        ~Profiler() {
            end();
// #ifndef LOGGING_OFF
// 			LOG_INFO("{}, Time: {:.2f} ms.", s, cumulative_time);
// #else
            std::cout << "profiler: "<< s << ", Time: " << cumulative_time << "ms." << std::endl;
// #endif
        }
    private:
        Clock::time_point t1;
        Clock::time_point t2;
        double cumulative_time = 0;
        std::string s="";
        bool started = false;
};

#define STATIC_TIC static Profiler profiler(std::string(__FUNCTION__) + ", " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); profiler.start()
#define STATIC_TOC profiler.end()

#define TIC(profilerlogger) Profiler *profilerlogger = new Profiler(std::string(__FUNCTION__) + ", " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); profilerlogger->start()
#define TOC(profilerlogger) delete profilerlogger

#ifndef INTEL_MKL
//CBLAS declarations
extern "C"
{
    void sgesv_(int *N, int *NRHS, float *A, int *lda, int *IPIV, float *B, int *ldb, int *INFO);

    // LU decomoposition of a general matrix
    void sgetrf_(int *M, int *N, float *A, int *lda, int *IPIV, int *INFO);
    void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

    void sgetrs_(char *TRANS, int *N, int *NRHS, float *A, int *lda, int *IPIV, float *B, int *ldb, int *INFO);

    // generate inverse of a matrix given its LU decomposition
    void sgetri_(int *N, float *A, int *lda, int *IPIV, float *WORK, int *lwork, int *INFO);
    void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork, int *INFO);
}
#endif

//overloading CBLAS interface for different types. 
inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, float *A, int lda, float *x, int ldx, float beta, float *y, int ldy){
    STATIC_TIC;
    cblas_sgemv(CblasColMajor, transM, m, n, alpha, A, lda, x, ldx, beta, y, ldy);
    STATIC_TOC;
}

inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, double *A, int lda, double *x, int ldx, float beta, double *y, int ldy){
    cblas_dgemv(CblasColMajor, transM, m, n, alpha, A, lda, x, ldx, beta, y, ldy);
}

inline void gemv(CBLAS_TRANSPOSE transM, int m, int n, float alpha, int *A, int lda, int *x, int ldx, float beta, int *y, int ldy){
    std::cout << "gemv: int* not implemented" << std::endl;
}

inline void gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int nr, int nc, int nk, 
            float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc){
    STATIC_TIC;
    cblas_sgemm(CblasColMajor, transA, transB,
                nr, nc, nk, alpha, A, lda, B, ldb, beta, C, ldc);
    STATIC_TOC;
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