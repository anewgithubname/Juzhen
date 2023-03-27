#include "../cpp/juzhen.hpp"
#define HLINE std::cout << "--------------------------------" << std::endl

int compute() {
    global_rand_gen.seed(0);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif

    std::cout << "Source Dir Folder:" << PROJECT_DIR << std::endl;

    // //spdlog::set_level(spdlog::level::debug);
    // {
    //     auto d = CM::randn(1000000, 1).to_host();
    //     std::vector<float> data(d.data(), d.data() + d.num_row());
    //     int num_bins = 50;
    //     print_histogram(data, num_bins);
    // }

    // HLINE;

    {
        auto A = M::rand(2, 3);
        std::cout << A << std::endl;
    
        auto B = M::rand(3, 2);
        std::cout << B << std::endl;

        auto C = sin(A) / log(exp(A) + exp(B));
        std::cout << C << std::endl;
    }
    
    HLINE;

    {
#ifndef CPU_ONLY
        auto C = CM::zeros(5000, 5000);

        auto A = CM::randn(5000, 5001);
        auto B = CM::randn(5001, 5000);
#else
        auto C = M::zeros(5000, 5000);

        auto A = M::randn(5000, 5001);
        auto B = M::randn(5001, 5000);
#endif
        TIC(bench); LOG_INFO("benchmark started...");
        for (int i = 0; i < 10; i++)
        {
            C += A * B / A.num_col();
        }
        std::cout << "C's norm is " << C.norm() << "." << std::endl;
        TOC(bench); LOG_INFO("benchmark finished.");
    }

    HLINE;
  
    return 0;
}
