/**
 * @file core.hpp
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

#ifndef CORE_HPP
#define CORE_HPP

#include <stdlib.h>
#include <memory>
#include <string>
#include <iostream>
#include <random>
#include <string.h>
#include <list>
#include "helper.h"

template <class D>
struct mem_space{
    D* ptr;
    int size;
};

template <class D>
struct MemoryDeleter;

template <class D>
class Matrix
{
protected:
    int numcol;
    int numrow;
    bool transpose;
    std::string name;    

    std::shared_ptr<D[]> elements;
    static std::list<mem_space<D>> alive_mems;
    static std::list<mem_space<D>> dead_mems;

    static D* allocate(int size){
        //static Profiler p(std::string("allocator ") + typeid(D).name()); p.start();
        // search for space in the freed space.
        for(auto it = dead_mems.begin(); it != dead_mems.end(); it++){
            if(it->size == size){
                LOG_DEBUG("Found CPU space in dead memory: {}, address: {}", size, fmt::ptr(it->ptr));
                D* ptr = it->ptr;
                mem_space<D> mem; mem.ptr = it->ptr;  mem.size = it->size;
                alive_mems.push_back(mem);
                dead_mems.erase(it);
                //p.end();
                return ptr;
            }
        }
        // no space available, allocate new space
        D* ptr = new D[size];
        LOG_DEBUG("No CPU space available, allocate new space: {}, address: {}", size, fmt::ptr(ptr));
        mem_space<D> space = {ptr, size};
        alive_mems.push_back(space);
        //p.end();
        return ptr;
    }

    static void free(D* ptr){
        //static Profiler p(std::string("deleter ") + typeid(D).name()); 
        int size = -1; 
        //p.start();
        for (auto it = alive_mems.begin(); it != alive_mems.end(); it++) {
            if (it->ptr == ptr) {
                size = it->size;
                break;
            }
        }
        mem_space<D> mem = {ptr, size};
        dead_mems.push_back(mem);
        for(auto it = alive_mems.begin(); it != alive_mems.end(); it++){
            if(it->ptr == ptr){
                alive_mems.erase(it);
                break;
            }
        }
        //p.end();
    }

    Matrix(const char *name, int numrow, int numcol, int trans, std::shared_ptr<D[]> elements){
        this->name = name; 
        this->numrow = numrow;
        this->numcol = numcol;
        this->transpose = trans;
        this->elements = elements;
    }

    inline int idx(int i, int j) const{
        return transpose ? i *numrow + j : j * numrow + i;
    }
    Matrix(const char *name, int numrow, int numcol, int trans);

    //random number generator
    static std::mt19937 randomnumber_gen;
public:
    // constructors 
    Matrix() : numcol(0), numrow(0), elements(nullptr), transpose(0), name("uninit") {}
    Matrix(const char *name, int numrow, int numcol): Matrix(name, numrow, numcol, 0) {}
    Matrix(const char *name, std::vector<std::vector<double>> elems);

    Matrix(const Matrix &M);
    Matrix(Matrix &&M) noexcept;
    Matrix<D> &operator=(const Matrix<D> &M);
    Matrix<D> &operator=(Matrix<D> &&M) noexcept;

    // access matrix info
    inline D elem(int i, int j) const { return elements[idx(i, j)]; }
    inline D &elem(int i, int j) { return elements[idx(i, j)]; }
    inline int num_col() const { return transpose ? numrow : numcol; }
    inline int num_row() const { return transpose ? numcol : numrow; }
    inline int get_transpose() const { return transpose; }
    std::string get_name() const { return name; }

    // Matrix Fillers
    virtual void zeros(){
        memset(elements.get(), 0, sizeof(D) * numrow * numcol);
    }
    virtual void ones();
    void rand(){
        for (int i = 0; i < num_row() * num_col(); i++)
            elements[i] = (D)rand_number();
    }
    void randn(){ randn(0.0, 1.0);}
    virtual void randn(double mean, double std);

    // Matrix Operations / performance notes
    Matrix<D> dot(const Matrix<D> &B) const; //using cblas_dgemm
    Matrix<D> add(const Matrix<D> &B, D s1, D s2) const; // using double for loop
    Matrix<D> add(const D &b, D s1) const; //using single for loop
    Matrix<D> scale(const D &s1) const { return add(0, s1); }
    Matrix<D> inv(); // using LU decomposition 
    D norm() const; //using single for loop
    const Matrix<D> T() const;
    
    //Matrix slicers
    virtual Matrix<D> columns(int start, int end) const;
    virtual Matrix<D> columns(idxlist cols) const;
    virtual Matrix<D> rows(int start, int end) const;
    virtual Matrix<D> rows(idxlist rows) const;
    virtual Matrix<D> slice(int rstart, int rend, int cstart, int cend) const;
    virtual Matrix<D> slice(const idxlist &rowidx, const idxlist &colidx) const;

    //Matrix IOs
    void read(std::string filename);
    void write(std::string filename);

    static Matrix<D> randn(int m, int n);
    static Matrix<D> ones(int m, int n);
    static Matrix<D> zeros(int m, int n);
    //Matrix's friends
    template <class Data>
    friend Matrix<Data> sum(const Matrix<Data> &M, int dim);
    template <class Data>
    friend Matrix<Data> exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> log(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> tanh(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> tanh(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> d_tanh(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> d_tanh(Matrix<Data> && M);
    
    template <class Data>
    friend Matrix<Data> atan_exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> atan_exp(Matrix<Data> && M); 
    template <class Data>
    friend Matrix<Data> d_atan_exp(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> d_atan_exp(Matrix<Data> && M);
    template <class Data>
    friend Matrix<Data> sin(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> sin(Matrix<Data> && M);
    template <class Data>
    friend Matrix<Data> cos(const Matrix<Data> &M);
    template <class Data>
    friend Matrix<Data> cos(Matrix<Data> && M);
    
    template <class Data>
    friend Matrix<Data> square(Matrix<Data> &&M);
    template <class Data>
    friend Matrix<Data> hstack(const std::vector<Matrix<Data>> &matrices);
    template <class Data>
    friend Matrix<Data> vstack(const std::vector<Matrix<Data>> &matrices);
    template <class Data>
    friend Matrix<Data> operator*(const Data &l, Matrix<Data> &&rM);
    template <class Data>
    friend Matrix<Data> operator/(const Data &l, const Matrix<Data> &rM);
    template <class Data> 
    friend Matrix<Data> operator/(const Data &l, Matrix<Data> &&rM);
    template <class Data>
    friend Matrix<Data> operator/(Matrix<Data>&& lM, const Data& r);
    template <class Data>
    friend Matrix<Data> operator/(const Matrix<Data> &lM, const Matrix<Data> &rM);
    template <class Data> 
    friend Matrix<Data> hadmd(const Matrix<Data> &M1, const Matrix<Data> &M2);
    friend class cuMatrix;
    friend struct MemoryDeleter<D>;
};

template <class D>
std::list<mem_space<D>> Matrix<D>::alive_mems;
template <class D>
std::list<mem_space<D>> Matrix<D>::dead_mems;
template <class D>
std::mt19937 Matrix<D>::randomnumber_gen(1);

template <class D>
struct MemoryDeleter{
    ~MemoryDeleter(){
        long size = 0; 
        for(auto it = Matrix<D>::alive_mems.begin(); it != Matrix<D>::alive_mems.end(); it++){
            delete []it->ptr;
            size += it->size;
        }
        for(auto it = Matrix<D>::dead_mems.begin(); it != Matrix<D>::dead_mems.end(); it++){
            delete []it->ptr;
            size += it->size;
        }
        LOG_INFO("Total CPU memory released: {:.2f} MB.", size*sizeof(D)/1024.0/1024.0);
    }
};

template <class D>
Matrix<D>::Matrix(const char *name, int numrow, int numcol, int trans){
    this->name = name;
    this->numcol = numcol;
    this->numrow = numrow;
    transpose = trans;
    //TODO: Not nice using reset, should change
    elements.reset(Matrix<D>::allocate(numrow*numcol), [](auto p) {
           Matrix<D>::free(p);
        //    delete []p;
        });
}

template<class D>
Matrix<D>::Matrix(const char *name, std::vector<std::vector<double>> elems){
    this->name = name;
    numrow = elems.size();
    numcol = elems.front().size();
    transpose = 0;
    
    //TODO: Not nice using reset, should change
    elements.reset(Matrix<D>::allocate(numrow*numcol), [](auto p) {
           Matrix<D>::free(p);
        //    delete []p;
        });

    // double for loop may have some performance issues, 
    // but should be OK for small matrices. 
    // remember to always loop over columns first. 
    for (int j = 0; j < numcol; j++){
        for (int i = 0; i < numrow; i++){
            elem(i, j) = elems[i][j];
        }
    }
}

template <class D>
Matrix<D>::Matrix(const Matrix<D> &M){
    LOG_WARN("copy construction called, {}!", M.name);
    numcol = M.numcol;
    numrow = M.numrow;
    transpose = M.transpose;
    elements.reset(Matrix<D>::allocate(numrow*numcol), [](auto p) {
        Matrix<D>::free(p);
    });

    memcpy(elements.get(), M.elements.get(), numrow*numcol*sizeof(D));
    name = "copy of " + M.name;
}

template <class D>
Matrix<D>::Matrix(Matrix<D> &&M) noexcept{
    LOG_DEBUG("move constructor called");
    this->numcol = M.numcol;
    this->numrow = M.numrow;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
    this->name =  M.name;
}

template <class D>
Matrix<D> &Matrix<D>::operator=(const Matrix<D> &M){
    LOG_WARN("copy assignment called!");
    if(this == &M) return *this;
    numcol = M.numcol;
    numrow = M.numrow;
    transpose = M.transpose;
    elements.reset(Matrix<D>::allocate(numrow*numcol), [](auto p) {
        Matrix<D>::free(p);
    });

    memcpy(elements.get(), M.elements.get(), numrow*numcol*sizeof(D));
    name = "copy of " + M.name;
    return *this;
}

template <class D>
Matrix<D> &Matrix<D>::operator=(Matrix<D> &&M) noexcept{
    LOG_DEBUG("move assignment called");
    if(this == &M) return *this;
    this->numcol = M.numcol;
    this->numrow = M.numrow;
    this->transpose = M.transpose;
    this->elements = M.elements;
    M.elements = nullptr;
    this->name =  M.name;
    return *this;
}

template<class D>
void Matrix<D>::ones(){
    int numelems = num_row() * num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        elements[i] = 1.0;
}

//non vectorized randn
template<class D>
void Matrix<D>::randn(double mean, double std)
{
    using namespace std;
    normal_distribution<> d(mean, std);
    //cannot be vectorized, due to the implementation of std::random.
    for (int i = 0; i < num_row() * num_col(); i++)
        elements[i] = d(randomnumber_gen);
}
//vectorized norm
template<class D>
D Matrix<D>::norm() const
{
    D norm = 0;
    int numelems = num_row() * num_col();
#pragma clang loop vectorize(enable)
#pragma ivdep
    for (int i = 0; i < numelems; i++)
        norm += elements[i] * elements[i];

    return sqrt(norm);
}
//"fake" matrix transposition. It does not copy the data.
template<class D>
const Matrix<D> Matrix<D>::T() const 
{
    std::string newname = name+"_T";
    Matrix<D> MT(newname.c_str(), numrow, numcol, !transpose, elements);
    return MT;
}

template<class D>
Matrix<D> Matrix<D>::slice(int rstart, int rend, int cstart, int cend) const
{
    Matrix<D> M("submatrix", rend - rstart, cend - cstart, 0);
    for (int j = cstart; j < cend; j++)
    {
        for (int i = rstart; i < rend; i++)
        {
            M.elem(i - rstart, j - cstart) = elem(i, j);
        }
    }
    return M;
}

template <class D>
Matrix<D> Matrix<D>::slice(const idxlist &rowidx, const idxlist &colidx) const
{
    Matrix<D> M("submatrix", rowidx.size(), colidx.size(), 0);
    int i = 0;
    for (auto it = rowidx.begin(); it != rowidx.end(); it++)
    {
        int r = *it;
        int j = 0;
        for (auto it2 = colidx.begin(); it2 != colidx.end(); it2++)
        {
            int c = *it2;
            M.elem(i, j) = elem(r, c);
            j++;
        }
        i++;
    }
    return M;
}
template <class D>
Matrix<D> Matrix<D>::dot(const Matrix<D> &B) const
{
    STATIC_TIC;
    Matrix<D> C("dot", num_row(), B.num_col(),0);
    C.zeros();
#ifdef NO_CBLAS
    int n = C.num_row();
    int m = C.num_col();
    int k = num_col();

#pragma omp simd
#pragma ivdep
    for (int i = 0; i < n; i++)
    {
#pragma ivdep
        for (int j = 0; j < m; j++)
        {
#pragma ivdep
            for (int l = 0; l < k; l++)
            {
                C.elem(i, j) += elem(i, l) * B.elem(l, j);
            }
        }
    }
#else
    CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = B.transpose ? CblasTrans : CblasNoTrans;

    gemm(transA, transB, num_row(), B.num_col(), num_col(), 1.0f, elements.get(), numrow,
         B.elements.get(), B.numrow, 1.0f, C.elements.get(), C.numrow);
#endif
    STATIC_TOC;
    return C;
}

/*
Matrix inversion using Lapack, found and modified from: 
https://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c/41474839
*/
template <class D>
Matrix<D> Matrix<D>::inv()
{
    Matrix<D> inv("inv_M", numrow, numcol, 0);
    for (int i = 0; i < numrow * numcol; i++)
    {
        inv.elements[i] = elements[i];
    }

    int N = num_row();
    int error = 0;
    int *pivot = (int *)malloc(N * sizeof(int)); // LAPACK requires MIN(M,N), here M==N, so N will do fine.
    int Nwork = 2 * N * N;
    D *workspace = (D *)malloc(Nwork * sizeof(D));

    /*  LU factorisation */
    getrf_(&N, &N, inv.elements.get(), &N, pivot, &error);

    if (error != 0)
    {
        // NSLog(@"Error 1");
        std::cout << "Error 1" << std::endl;
        ::free(pivot);
        ::free(workspace);
        exit(1);
    }

    /*  matrix inversion */
    getri_(&N, inv.elements.get(), &N, pivot, workspace, &Nwork, &error);

    if (error != 0)
    {
        // NSLog(@"Error 2");
        std::cout << "Error 2" << std::endl;
        ::free(pivot);
        ::free(workspace);
        exit(1);
    }

    ::free(pivot);
    ::free(workspace);

    return inv;
}

// s1*A + s2*B
template <class D>
Matrix<D> Matrix<D>::add(const Matrix<D> &B, D s1, D s2) const
{
    Matrix<D> C("add", num_row(), num_col(), 0); 

#ifdef INTEL_MKL
    C.zeros();
    char transA = transpose ? 'T': 'N';
    char transB = B.transpose ?  'T': 'N';

    omatadd(transA, transB, num_row(), num_col(), s1, elements.get(), numrow,
                 s2, B.elements.get(), B.numrow, C.elements.get(), C.numrow);
#else
    for (int j = 0; j < num_col(); j++)
    {
        for (int i = 0; i < num_row(); i++)
        {
            //NOTE: Perhaps there is better way of doing this
            // considering the cache hit rate. 
            C.elements[idx(i,j)] = s1 * elements[idx(i,j)] + s2 * B.elements[idx(i,j)];
        }
    }
#endif
    return C;
}
//s1*A + b
template <class D>
Matrix<D> Matrix<D>::add(const D &b, D s1) const
{
    Matrix<D> C("add", numrow, numcol, transpose);
    int numelems = num_row() * num_col();
#pragma ivdep
    for (int i = 0; i < numelems; i++)
    {
        C.elements[i] = s1 * elements[i] + b;
    }
    return C;
}

template <class D>
/*
    Read a matrix from file.
    filename: the file that contains the matrix.
    */
void Matrix<D>::read(std::string filename)
{
    FILE *f = fopen(filename.c_str(), "rb");
    // read int variables to the file.
    numrow = getw(f);
    numcol = getw(f);
    transpose = getw(f);

    fread(elements.get(), sizeof(D), numcol * numrow, f);

    fclose(f);
}
template <class D>
/*
    Write matrix M to file
    M: the matrix to be written
    filename: name of the file to be created.
    */
void Matrix<D>::write(std::string filename)
{
    FILE *f = fopen(filename.c_str(), "wb");
    // write int variables to the file.
    putw(numrow, f);
    putw(numcol, f);
    putw(transpose, f);

    fwrite(elements.get(), sizeof(D), numcol * numrow, f);

    fclose(f);
}

#endif