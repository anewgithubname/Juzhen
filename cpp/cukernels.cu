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
#include "cumatrix.cuh"
using namespace std;

__global__ void addKernel(float *d_out, float s1, float a, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        d_out[i] = s1 * d_out[i] + a;
    }
}

__global__ void fillKernel(float *d_out, float val, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        d_out[i] = val;
    }
}

__global__ void squareKernel(float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vec[i] *= vec[i];
    }
}

__global__ void squareKernel(float *res, float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        res[i] = vec[i] * vec[i];
    }
}

__global__ void productKernel(float *d_out, float *d_in, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        d_out[i] *= d_in[i];
    }
}

__global__ void copyKernel(float *d_out, float *d_in, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        d_out[i] = d_in[i];
    }
}

__global__ void copyKernel(float *d_out, float *d_in, size_t numElements,
                           size_t numrow, size_t rowstart, size_t rowend,
                           size_t colstart, size_t colend, bool direction) {
    size_t k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < numElements) {
        // d_out[i] = d_in[i];
        int numrow_out = rowend - rowstart;
        int j = k / numrow_out;
        int i = k % numrow_out;

        int idx = (i + rowstart) + (j + colstart) * numrow;

        if (direction)
            d_out[k] = d_in[idx];  // copy out
        else
            d_in[idx] = d_out[k];  // assign
    }
}

__global__ void inplaceExpKernel(float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vec[i] = exp(vec[i]);
    }
}

__global__ void expKernel(float *vecdes, float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = exp(vec[i]);
    }
}

__global__ void logKernel(float *vecdes, float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = log(vec[i]);
    }
}

__global__ void tanhKernel(float *vecdes, float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = tanh(vec[i]);
    }
}

__global__ void inPlaceTanhKernel(float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vec[i] = tanh(vec[i]);
    }
}

__global__ void d_tanhKernel(float *vecdes, float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vecdes[i] = 1.0 - tanh(vec[i]) * tanh(vec[i]);
    }
}

__global__ void inplaceD_tanhKernel(float *vec, size_t numElements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        vec[i] = 1.0 - tanh(vec[i]) * tanh(vec[i]);
    }
}

Matrix<CUDAfloat> &fill(Matrix<CUDAfloat> &M, double a) {
    size_t numElem = M.num_row() * M.num_col();
    fillKernel<<<cudaConfig(numElem)>>>((float *)M.elements.get(), (float)a,
                                        numElem);
    return M;
}

// in place exponential
Matrix<CUDAfloat> exp(Matrix<CUDAfloat> &&M) {
    size_t numElem = M.num_row() * M.num_col();
    inplaceExpKernel<<<cudaConfig(numElem)>>>((float *)M.elements.get(),
                                              numElem);
    return std::move(M);
}

Matrix<CUDAfloat> exp(const Matrix<CUDAfloat> &M) {
    STATIC_TIC;
    Matrix<CUDAfloat> result("expM", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col();
    expKernel<<<cudaConfig(numElem)>>>((float *)result.elements.get(),
                                       (float *)M.elements.get(), numElem);
    STATIC_TOC;
    return result;
}

Matrix<CUDAfloat> log(const Matrix<CUDAfloat> &M) {
    Matrix<CUDAfloat> result("logM", M.numrow, M.numcol, M.transpose);

    size_t numElem = M.num_row() * M.num_col();
    logKernel<<<cudaConfig(numElem)>>>((float *)result.elements.get(),
                                       (float *)M.elements.get(), numElem);
    return result;
}

Matrix<CUDAfloat> tanh(const Matrix<CUDAfloat> &M) {
    Matrix<CUDAfloat> result("tanhM", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col();
    tanhKernel<<<cudaConfig(numElem)>>>((float *)result.elements.get(),
                                        (float *)M.elements.get(), numElem);
    return result;
}

// in place tanh
Matrix<CUDAfloat> tanh(Matrix<CUDAfloat> &&M) {
    STATIC_TIC;
    // cout << "rval tanh" << endl;
    size_t numElem = M.num_row() * M.num_col();
    inPlaceTanhKernel<<<cudaConfig(numElem)>>>((float *)M.elements.get(),
                                               numElem);
    STATIC_TOC;
    return std::move(M);
}

Matrix<CUDAfloat> d_tanh(const Matrix<CUDAfloat> &M) {
    STATIC_TIC;
    Matrix<CUDAfloat> result("d_tanhM", M.numrow, M.numcol, M.transpose);

    size_t numElem = M.num_row() * M.num_col();
    d_tanhKernel<<<cudaConfig(numElem)>>>((float *)result.elements.get(),
                                          (float *)M.elements.get(), numElem);
    STATIC_TOC;
    return result;
}

// in place d_tanh
Matrix<CUDAfloat> d_tanh(Matrix<CUDAfloat> &&M) {
    STATIC_TIC;
    // cout << "rval d_tanh" << endl;
    size_t numElem = M.num_row() * M.num_col();
    inplaceD_tanhKernel<<<cudaConfig(numElem)>>>((float *)M.elements.get(),
                                                 numElem);
    STATIC_TOC;
    return std::move(M);
}

// in place square
Matrix<CUDAfloat> square(Matrix<CUDAfloat> &&M) {
    // cout << "rval d_tanh" << endl;
    size_t numElem = M.num_row() * M.num_col();
    squareKernel<<<cudaConfig(numElem)>>>((float *)M.elements.get(), numElem);
    return std::move(M);
}

// in place square
Matrix<CUDAfloat> square(const Matrix<CUDAfloat> &M) {
    // cout << "lval d_tanh" << endl;
    Matrix<CUDAfloat> res("square", M.numrow, M.numcol, M.transpose);
    size_t numElem = M.num_row() * M.num_col();
    squareKernel<<<cudaConfig(numElem)>>>((float *)res.elements.get(),
                                          (float *)M.elements.get(), numElem);
    return res;
}

void copy(Matrix<CUDAfloat> &dest, const Matrix<CUDAfloat> &src) {
    size_t numElem = src.num_row() * src.num_col();

    dest.numrow = src.numrow;
    dest.numcol = src.numcol;
    dest.transpose = src.transpose;

    dest.elements.reset();
    CUDAfloat *p = Memory<CUDAfloat>::allocate(numElem);

    dest.elements = shared_ptr<CUDAfloat[]>(
        p, [](CUDAfloat *p) { Memory<CUDAfloat>::free(p); });

    copyKernel<<<cudaConfig(numElem)>>>((float *)dest.elements.get(),
                                        (float *)src.elements.get(), numElem);
}

Matrix<CUDAfloat> hstack(vector<MatrixView<CUDAfloat>> matrices) {
    // remove matrix size of zero, otherwise, it will cause CUBLAS error
    auto t = std::remove_if(matrices.begin(), matrices.end(),
                            [](const MatrixView<CUDAfloat> &m) {
                                return m.num_row() == 0 || m.num_col() == 0;
                            });
    matrices.erase(t, matrices.end());

    size_t num_row = matrices[0].num_row();
    size_t num_col = 0;
    for (size_t i = 0; i < matrices.size(); i++) {
        num_col += matrices[i].num_col();
    }
    Matrix<CUDAfloat> result("hstack", num_row, num_col, 0);

    size_t col_index = 0;
    for (size_t i = 0; i < matrices.size(); i++) {
        if (matrices[i].get_transpose() == 0) {
            size_t numElem = matrices[i].num_row() * matrices[i].num_col();
            copyKernel<<<cudaConfig(numElem)>>>(
                (float *)&result.elements.get()[col_index * num_row],
                (float *)matrices[i].data(), numElem);
        } else {
            float alpha = 1.0, beta = 0.0;
            CuBLASErrorCheck(cublasSgeam(
                Matrix<CUDAfloat>::global_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                num_row, matrices[i].num_col(), &alpha,
                (float *)matrices[i].data(), matrices[i].num_col(), &beta,
                (float *)&result.elements.get()[col_index * num_row], num_row,
                (float *)&result.elements.get()[col_index * num_row], num_row));
        }
        col_index += matrices[i].num_col();
    }

    return result;
}

const Matrix<CUDAfloat> vstack(vector<MatrixView<CUDAfloat>> matrices) {
    for (size_t i = 0; i < matrices.size(); i++) {        
        matrices[i].transpose = !matrices[i].transpose;
    }

    return hstack(matrices).T();
}

Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat> &M1,
                        const Matrix<CUDAfloat> &M2) {
    static Profiler p("GPU hadmd");
    p.start();
    Matrix<CUDAfloat> result("hadmd", M1.numrow, M1.numcol, M1.transpose);

    // if M2 has a different transposition flag, transpose M2 and store it in
    // result.
    cublasOperation_t transM2 =
        (M2.transpose != M1.transpose) ? CUBLAS_OP_T : CUBLAS_OP_N;
    float s1 = 1.0, s2 = 0.0;

    CuBLASErrorCheck(cublasSgeam(result.handle, transM2, CUBLAS_OP_N, M1.numrow,
                                 M1.numcol, &s1, (float *)M2.elements.get(),
                                 M2.numrow, &s2, (float *)result.elements.get(),
                                 result.numrow, (float *)result.elements.get(),
                                 result.numrow));

    size_t numElem = M1.numrow * M1.numcol;
    productKernel<<<cudaConfig(numElem)>>>((float *)result.elements.get(),
                                           (float *)M1.elements.get(), numElem);
    p.end();
    return result;
}

// Matrix<GPUfloat> hadmd(const Matrix<GPUfloat>& M1, Matrix<GPUfloat>&& M2) {
//     return hadmd(M1, M2);
// }

// Matrix<GPUfloat> hadmd(Matrix<GPUfloat>&& M1, const Matrix<GPUfloat> & M2) {
//     return hadmd(M1, M2);
// }

// rvalue hadmd
Matrix<CUDAfloat> hadmd(const Matrix<CUDAfloat> &M1, Matrix<CUDAfloat> &&M2) {
    // cout << "rvalue hadmd" << endl;
    // if M2 has a different transposition flag, use the lval version
    if (M2.transpose == M1.transpose) {
        size_t numElem = M1.numrow * M1.numcol;
        productKernel<<<cudaConfig(numElem)>>>(
            (float *)M2.elements.get(), (float *)M1.elements.get(), numElem);
        return std::move(M2);
    } else {
        return hadmd(M1, M2);
    }
}
// rvalue hadmd
Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat> &&M1, const Matrix<CUDAfloat> &M2) {
    // cout << "rvalue hadmd" << endl;
    // if M2 has a different transposition flag, use the lval version
    if (M2.transpose == M1.transpose) {
        size_t numElem = M1.numrow * M1.numcol;
        productKernel<<<cudaConfig(numElem)>>>(
            (float *)M1.elements.get(), (float *)M2.elements.get(), numElem);
        return std::move(M1);
    } else {
        return hadmd(M1, M2);
    }
}

Matrix<CUDAfloat> hadmd(Matrix<CUDAfloat> &&M1, Matrix<CUDAfloat> &&M2) {
    return hadmd(M1, std::move(M2));
}
