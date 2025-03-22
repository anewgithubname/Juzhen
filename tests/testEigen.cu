/**
 * @file testEigen.cu
 * @brief a side by side comparison between juzhen and Eigen
 */

#define EIGEN_USE_BLAS

#include "../cpp/juzhen.hpp"
#include "../external/Eigen3/include/eigen3/Eigen/Dense"

void dumpJuzhentoarray(float *array, const M &m)
{
    for (int i = 0; i < m.num_col(); i++)
    {
        for (int j = 0; j < m.num_row(); j++)
        {
            array[i * m.num_row() + j] = m.data()[j * m.num_col() + i];
        }
    }
}

void dumpEigentoarray(float *array, const Eigen::MatrixXf &m)
{
    for (int i = 0; i < m.cols(); i++)
    {
        for (int j = 0; j < m.rows(); j++)
        {
            array[i * m.rows() + j] = m.data()[j * m.cols() + i];
        }
    }
}

void copyJuzhentoEigen(Eigen::MatrixXf &m, const M &M)
{
    for (int i = 0; i < M.num_row(); i++)
    {
        for (int j = 0; j < M.num_col(); j++)
        {
            m(i, j) = M(i, j);
        }
    }
}

int comparearraies(float *array1, float *array2, int size)
{
    int ret = 0;
    for (int i = 0; i < size; i++)
    {
        if (fabs(array1[i] - array2[i]) > 1e-3)
        {
            std::cout << "array1[" << i << "] = " << array1[i] << " array2[" << i << "] = " << array2[i] << std::endl;
            ret = 1;
            break;
        }
    }
    return ret;
}

int test1()
{
    std::string base = PROJECT_DIR;
    // unit test started
    M A = {"A", {{1, 2, 3}, {3, 4, 5}}};
    M B = {"B", {{6, 7, 8}, {9, 10, 11}}};

    // compute log(exp(A)+exp(B)) + AT*B(0:2, :) on CPU
    auto C = log (exp (A) + exp(B));
    std::cout << "C = " << C << std::endl;

    // export result to array
    float C_array[6] = {0};
    dumpJuzhentoarray(C_array, C);

    //compute log(exp(A)+exp(B)) + AT*B(0:2, :) using Eigen on CPU
    Eigen::MatrixXf A1(2, 3);
    A1 << 1, 2, 3, 3, 4, 5;
    Eigen::MatrixXf B1(2, 3);
    B1 << 6, 7, 8, 9, 10, 11;

    //print A1, B1
    std::cout << "A1 = " << A1 << std::endl;
    std::cout << "B1 = " << B1 << std::endl;

    //compute log(exp(A)+exp(B))
    Eigen::MatrixXf C1 = (A1.array().exp() + B1.array().exp()).log();
    std::cout << "C1 = " << C1 << std::endl;

    // export C1 to array
    float C1_array[6] = {0};
    dumpEigentoarray(C1_array, C1);

    //compare two results in array
    return comparearraies(C_array, C1_array, 6);
}


int test2(){
    // test matrix multiplication
    M A = {"A", {{1, 2, 3}, {3, 4, 5}}};
    M B = {"B", {{6, 7, 8}, {9, 10, 11}}};

    // compute A*B on CPU
    auto C = A*B.T();
    std::cout << "C = " << C << std::endl;
    float C_array[4] = {0};
    dumpJuzhentoarray(C_array, C);

    //compute A*B using Eigen on CPU
    Eigen::MatrixXf A1(2, 3);
    A1 << 1, 2, 3, 3, 4, 5;
    Eigen::MatrixXf B1(2, 3);
    B1 << 6, 7, 8, 9, 10, 11;

    auto C1 = A1*B1.transpose();
    std::cout << "C1 = " << std::endl << C1 << std::endl;
    float C1_array[4] = {0};
    dumpEigentoarray(C1_array, C1);

    //compare two results in array
    return comparearraies(C_array, C1_array, 4);
}


int test3() {
    // test random large matrices multiplication

    int n = 10001;
#ifdef CUDA
    auto A = CM::randn(n, n);
    auto B = CM::randn(n, n);
    auto C = CM::zeros(n, n);
#elif defined(APPLE_SILICON)
    auto A = Matrix<MPSfloat>::randn(n, n);
    auto B = Matrix<MPSfloat>::randn(n, n);
    auto C = Matrix<MPSfloat>::zeros(n, n);
#else
    auto A = M::randn(n, n);
    auto B = M::randn(n, n);
    auto C = M::zeros(n, n);
#endif

    {Profiler p("Juzhen");
        // compute A*B on CPU
        for (int i = 0; i < 50; i++) {
            C += hadmd(A * B.T() / n, A) + B - A;
        }
    }

    auto *C_array = new float[n*n];
#if defined(CUDA) || defined(APPLE_SILICON)
    dumpJuzhentoarray(C_array, C.to_host());
#else
    dumpJuzhentoarray(C_array, C);
#endif

    // do the same thing using eigen
    Eigen::MatrixXf A1(n, n);
    // copy A to A1
#if defined(CUDA) || defined(APPLE_SILICON)
    copyJuzhentoEigen(A1, A.to_host());
#else
    copyJuzhentoEigen(A1, A);
#endif

    Eigen::MatrixXf B1(n, n);
#if defined(CUDA) || defined(APPLE_SILICON)
    copyJuzhentoEigen(B1, B.to_host());
#else
    copyJuzhentoEigen(B1, B);
#endif

    Eigen::MatrixXf C1(n, n);
    {Profiler p("Eigen");
        for (int i = 0; i < 50; i++) {
            C1 += (A1 * B1.transpose() / n).cwiseProduct(A1) + B1 - A1;
        }
    }
    auto *C1_array = new float[n*n];
    dumpEigentoarray(C1_array, C1);

    //compare two results in array
    int ret = comparearraies(C_array, C1_array, n*n);

    delete[] C_array;
    delete[] C1_array;

    return ret;

}

int compute()
{
#ifdef CUDA
    GPUSampler sampler(1);
#endif
//    spdlog::set_level(spdlog::level::debug);
    std::cout << __cplusplus << " " << HAS_CONCEPTS << std::endl;

    int ret = 0;
    ret += test1();
    std::cout << std::endl;

    ret += test2();
    std::cout << std::endl;

    ret += test3();
    std::cout << std::endl;

    if (ret == 0)
    {
        LOG_INFO("unit test passed!");
    }
    else
    {
        LOG_ERROR("unit test failed!");
    }

    return ret;
}