/**
 * @file cukernels.cu
 * @brief cuda kernels and some kenernel dependent functions
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
#include "cuda.h"
#include "cublas_v2.h"
#include "cudam.h"
using namespace std;

__global__ void fillKernel(float *d_out, float val, int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = val;
    }
}

__global__ void productKernel(float *d_out, float *d_in, int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] *= d_in[i];
    }
}

__global__ void divKernel(float *d_out, float *d_in, int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] /= d_in[i];
    }
}

__global__ void divKernel(float *d_out, float d_in1, float *d_in2, int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = d_in1 / d_in2[i];
    }
}

__global__ void copyKernel(float *d_out, float *d_in, int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_out[i] = d_in[i];
    }
}

__global__ void inplaceExpKernel(float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vec[i] = exp(vec[i]);
    }
}

__global__ void expKernel(float *vecdes, float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = exp(vec[i]);
    }
}

__global__ void logKernel(float *vecdes, float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = log(vec[i]);
    }
}

__global__ void tanhKernel(float *vecdes, float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = tanh(vec[i]);
    }
}

__global__ void inPlaceTanhKernel(float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vec[i] = tanh(vec[i]);
    }
}

__global__ void d_tanhKernel(float *vecdes, float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vecdes[i] = 1.0 - tanh(vec[i]) * tanh(vec[i]);
    }
}

__global__ void inplaceD_tanhKernel(float *vec, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        vec[i] = 1.0 - tanh(vec[i]) * tanh(vec[i]);
    }
}

cuMatrix& fill(cuMatrix &M, float a){
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    fillKernel<<<blocksPerGrid, threadsPerBlock>>>(M.elements.get(), a, numElem);
    return M;
}

// in place exponential
cuMatrix exp(cuMatrix &&M)
{
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    inplaceExpKernel<<<blocksPerGrid, threadsPerBlock>>>(M.elements.get(), numElem);
    return std::move(M);
}

cuMatrix exp(const cuMatrix &M)
{
    cuMatrix result(M.handle, "expM", M.numrow, M.numcol, M.transpose);
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    expKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M.elements.get(), numElem);
    return result;
}

cuMatrix log(const cuMatrix &M)
{
    cuMatrix result(M.handle, "logM", M.numrow, M.numcol, M.transpose);

    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    logKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M.elements.get(), numElem);
    return result;
}

cuMatrix tanh(const cuMatrix &M)
{
    cuMatrix result(M.handle, "tanhM", M.numrow, M.numcol, M.transpose);
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M.elements.get(), numElem);
    return result;
}

//in place tanh
cuMatrix tanh(cuMatrix &&M)
{
    // cout << "rval tanh" << endl;
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    inPlaceTanhKernel<<<blocksPerGrid, threadsPerBlock>>>(M.elements.get(), numElem);
    return std::move(M);
}

cuMatrix d_tanh(const cuMatrix &M)
{
    cuMatrix result(M.handle, "d_tanhM", M.numrow, M.numcol, M.transpose);

    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    d_tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M.elements.get(), numElem);
    return result;
}

//in place d_tanh
cuMatrix d_tanh(cuMatrix &&M)
{
    // cout << "rval d_tanh" << endl;
    int numElem = M.num_row() * M.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    inplaceD_tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(M.elements.get(), numElem);
    return std::move(M);
}

void copy(cuMatrix &dest, const cuMatrix &src)
{
    int numElem = src.num_row() * src.num_col();
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;

    dest.numrow = src.numrow;
    dest.numcol = src.numcol;
    dest.transpose = src.transpose;

    dest.elements.reset();
    float *p = cuMatrix::allocate(numElem);
    //cudaError_t stats = cudaMalloc(&p, numElem * sizeof(float));
    //if (stats != cudaSuccess)
    //{
    //    printf("Error allocating memory on GPU\n");
    //    exit(1);
    //}
    dest.elements = shared_ptr<float[]>(p, [](auto p) {
        cuMatrix::free(p);
        });

    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(dest.elements.get(), src.elements.get(), numElem);
}

cuMatrix hstack(vector<cuMatrix> &matrices)
{
    int num_row = matrices[0].num_row();
    int num_col = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        num_col += matrices[i].num_col();
    }
    cuMatrix result(matrices[0].handle, "hstack", num_row, num_col, 0);

    int col_index = 0;
    for (int i = 0; i < matrices.size(); i++)
    {
        if (matrices[i].get_transpose() == 0)
        {
            int numElem = matrices[i].num_row() * matrices[i].num_col();
            int threadsPerBlock = 1024;
            int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
            copyKernel<<<blocksPerGrid, threadsPerBlock>>>(&result.elements.get()[col_index * num_row], matrices[i].elements.get(), numElem);
        }
        else
        {
            float alpha = 1.0, beta = 0.0;
            cublasSgeam(matrices[0].handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        num_row, matrices[i].num_col(),
                        &alpha, matrices[i].elements.get(), matrices[i].numrow,
                        &beta, &result.elements.get()[col_index * num_row], num_row,
                        &result.elements.get()[col_index * num_row], num_row);
        }
        col_index += matrices[i].num_col();
    }

    return result;
}

cuMatrix hstack(vector<cuMatrix> &&matrices){
    return hstack(matrices);
}

const cuMatrix vstack(vector<cuMatrix>& matrices)
{

   for (int i = 0; i < matrices.size(); i++)
   {
       matrices[i].transpose = !matrices[i].transpose;
   }

   return hstack(matrices).T();
}

const cuMatrix vstack(vector<cuMatrix>&& matrices)
{
    return vstack(matrices);
}

cuMatrix hadmd(const cuMatrix &M1, const cuMatrix &M2)
{   
    cuMatrix result(M1.handle, "hadmd", M1.numrow, M1.numcol, M1.transpose);

    //if M2 has a different transposition flag, transpose M2 and store it in result. 
    cublasOperation_t transM2 = (M2.transpose != M1.transpose) ? CUBLAS_OP_T : CUBLAS_OP_N;
    float s1 = 1.0, s2 = 0.0;
    GPU_status stat = cublasSgeam(result.handle, transM2, CUBLAS_OP_N, M1.numrow, M1.numcol,
                                  &s1, M2.elements.get(), M2.numrow,
                                  &s2, result.elements.get(), result.numrow, result.elements.get(), result.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("tranpose failed");
    }

    int numElem = M1.numrow * M1.numcol;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    productKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M1.elements.get(), numElem);
    return result;
}
// //rvalue hadmd
// cuMatrix hadmd(const cuMatrix &M1, cuMatrix &&M2)
// {
//     cout << "rvalue hadmd" << endl;
//     //if M2 has a different transposition flag, transpose M2 and store it in result. 
//     cublasOperation_t transM2 = (M2.transpose != M1.transpose) ? CUBLAS_OP_T : CUBLAS_OP_N;
//     float s1 = 1.0, s2 = 0.0;
//     GPU_status stat = cublasSgeam(M1.handle, transM2, CUBLAS_OP_N, M1.numrow, M1.numcol,
//                                   &s1, M2.elements.get(), M2.numrow,
//                                   &s2, M2.elements.get(), M2.numrow, M2.elements.get(), M2.numrow);
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//         printf("tranpose failed");
//     }

//     int numElem = M1.numrow * M1.numcol;
//     int threadsPerBlock = 1024;
//     int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
//     productKernel<<<blocksPerGrid, threadsPerBlock>>>(M2.elements.get(), M1.elements.get(), numElem);
//     return std::move(M2);
// }
// //rvalue hadmd
// cuMatrix hadmd(cuMatrix &&M1, const cuMatrix &M2)
// {
//     cout << "rvalue hadmd" << endl;
//     //if M2 has a different transposition flag, transpose M1 and store it in result. 
//     cublasOperation_t transM1 = (M2.transpose != M1.transpose) ? CUBLAS_OP_T : CUBLAS_OP_N;
//     float s1 = 1.0, s2 = 0.0;
//     GPU_status stat = cublasSgeam(M2.handle, transM1, CUBLAS_OP_N, M2.numrow, M2.numcol,
//                                   &s1, M1.elements.get(), M1.numrow,
//                                   &s2, M1.elements.get(), M1.numrow, M1.elements.get(), M1.numrow);
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//         printf("tranpose failed");
//     }

//     int numElem = M1.numrow * M1.numcol;
//     int threadsPerBlock = 1024;
//     int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
//     productKernel<<<blocksPerGrid, threadsPerBlock>>>(M1.elements.get(), M2.elements.get(), numElem);
//     return std::move(M1);
// }

cuMatrix operator/(const float &l, const cuMatrix &rM)
{
    cuMatrix result(rM.handle, "elem_div", rM.numrow, rM.numcol, rM.transpose);

    int numElem = rM.numrow * rM.numcol;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    divKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), l, rM.elements.get(), numElem);

    return result;
}

cuMatrix operator/(const cuMatrix &M1, const cuMatrix &M2)
{
    cuMatrix result(M1.handle, "elem_div", M2.numrow, M2.numcol, M2.transpose);

    cublasOperation_t transM1 = (M2.transpose != M1.transpose) ? CUBLAS_OP_T : CUBLAS_OP_N;
    float s1 = 1.0, s2 = 1.0;
    GPU_status stat = cublasSgeam(result.handle, transM1, CUBLAS_OP_N, M2.numrow, M2.numcol,
                                  &s1, M1.elements.get(), M1.numrow,
                                  &s2, result.elements.get(), result.numrow, result.elements.get(), result.numrow);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("/ failed");
    }

    int numElem = M1.numrow * M1.numcol;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    divKernel<<<blocksPerGrid, threadsPerBlock>>>(result.elements.get(), M2.elements.get(), numElem);
    return result;
}