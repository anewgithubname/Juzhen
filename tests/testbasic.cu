#include "../cpp/juzhen.hpp"

int test1()
{
    std::string base = PROJECT_DIR;
    // unit test started
    M A = {"A", {{1, 2, 3}, {3, 4, 5}}};
    M B = {"B", {{6, 7, 8}, {9, 10, 11}}};

    // compute log(exp(A)+exp(B)) + AT*B(0:2, :) on CPU
    auto C = log(exp(-A / B) + exp(hadmd(B, A))) - (A.T() * B).rows(0, 2);
    std::cout << "C = " << C << std::endl;

    M bench = read<float>(base + "/tests/basic.testdata");

    if ((C - bench).norm() < 1e-5)
    {
        LOG_INFO("unit test passed!");
        return 0;
    }
    else
    {
        LOG_ERROR("unit test failed!");
        return 1;
    }
}

#ifdef CUDA
int test2()
{
    std::string base = PROJECT_DIR;
    // unit test started
    M AA = {"A", {{1, 2, 3}, {3, 4, 5}}};
    M BB = {"B", {{6, 7, 8}, {9, 10, 11}}};

    CM A(AA);
    CM B(BB);

    M bench = read<float>(base + "/tests/basic.testdata");

    // compute log(exp(A)+exp(B)) + AT*B(0:2, :) on GPU
    auto C = log(exp(-A / B) + exp(hadmd(B, A))) - (A.T() * B).rows(0, 2);
    std::cout << "C = " << C << std::endl;

    if ((C - (CM)bench).norm() < 1e-5)
    {
        LOG_INFO("unit test passed!");
        return 0;
    }
    else
    {
        LOG_ERROR("unit test failed!");
        return 1;
    }
}
#endif

int test3()
{
    // unit test started
    M A = {"A", {{1, 2, 3}, {3, 4, 5}}};
    A.columns(0, 2, M("T", {{1,1},{1,1}}));
    M B = {"B", {{6, 7, 8}, {9, 10, 11}}};
    B.rows(0, 1, M("T", {{-1,-1,-1}}));

    int ret = 0;
    if ((vstack<float>({A, B}) - M("C", {{1, 1, 3}, {1, 1, 5}, {-1, -1, -1}, {9, 10, 11}})).norm() > 1e-5)
    {
        LOG_ERROR("vstack failed!");
        ret += 1;
    }

    if ((hstack<float>({A, B}) - M("C", {{1, 1, 3, -1, -1, -1}, {1, 1, 5, 9, 10, 11}})).norm() > 1e-5)
    {
        LOG_ERROR("hstack failed!");
        ret += 1;
    }

    LOG_INFO("unit test passed!");
    return ret;
}

#ifdef CUDA
int test4()
{
    // unit test started
    M AA = {"A", {{1, 2, 3}, {3, 4, 5}}};
    M BB = {"B", {{6, 7, 8}, {9, 10, 11}}};

    CM A(AA);
    A.columns(0, 2, CM(M("T", {{1,1},{1,1}})));
    CM B(BB);
    B.rows(0, 1, CM(M("T", {{-1,-1,-1}})));

    int ret = 0;
    if ((vstack({A, B}).to_host() - M("C", {{1, 1, 3}, {1, 1, 5}, {-1, -1, -1}, {9, 10, 11}})).norm() > 1e-5)
    {
        LOG_ERROR("vstack failed!");
        ret += 1;
    }

    if ((hstack({A, B}).to_host() - M("C", {{1, 1, 3, -1, -1, -1}, {1, 1, 5, 9, 10, 11}})).norm() > 1e-5)
    {
        LOG_ERROR("hstack failed!");
        std::cout << hstack({A, B}).to_host() << std::endl;
        ret += 1;
    }

    LOG_INFO("unit test passed!");
    return ret;
}
#endif

int test5(){
#ifdef CUDA
    CM A = CM::ones(3, 4);
    CM B = CM::ones(3, 3);
#elif defined(APPLE_SILICON)
    Matrix<MPSfloat> A = Matrix<MPSfloat>::ones(3, 4);
    Matrix<MPSfloat> B = Matrix<MPSfloat>::ones(3, 3);
#else
    M A = M::ones(3, 4);
    M B = M::ones(3, 3);
#endif

    int count = 0; 

    // add with wrong dimensions
    try
    {
        std::cout << A + B << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // subtract with wrong dimensions
    try
    {
        std::cout << A - B << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // multiply with wrong dimensions
    try
    {
        std::cout << A * B << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // divide with wrong dimensions
    try
    {
        std::cout << A / B << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // hadamard with wrong dimensions
    try
    {
        std::cout << hadmd(A, B) << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // add with wrong dimensions
    try
    {
        std::cout << A + B.T() << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // subtract with wrong dimensions
    try
    {
        std::cout << A - B.T() << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // multiply with wrong dimensions
    try
    {
        std::cout << A * B.T() << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // divide with wrong dimensions
    try
    {
        std::cout << A / B.T() << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    // hadamard with wrong dimensions
    try
    {
        std::cout << hadmd(A, B.T()) << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        count++;
    }

    if (count == 10)
    {
        LOG_INFO("unit test passed!");
        return 0;
    }
    else
    {
        LOG_ERROR("unit test failed!");
        return 1;
    }

}

int compute()
{
    // spdlog::set_level(spdlog::level::debug);
    std::cout << __cplusplus << " " << HAS_CONCEPTS << std::endl;

    int ret = 0;
    ret += test1();
    std::cout << std::endl;

#ifdef CUDA
    ret += test2();
    std::cout << std::endl;
#endif

    ret += test3();
    std::cout << std::endl;

#ifdef CUDA
    ret += test4();
#endif

    ret += test5();
    std::cout << std::endl;

    if (ret == 0)
    {
        LOG_INFO("--------------------");
        LOG_INFO("|      ALL OK!     |");
        LOG_INFO("--------------------");
    }
    else
    {
        LOG_ERROR("--------------------");
        LOG_ERROR("|    NOT ALL OK!   |");
        LOG_ERROR("--------------------");
    }
        
    return ret;
}