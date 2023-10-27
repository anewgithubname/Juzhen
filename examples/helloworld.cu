
#include "../cpp/juzhen.hpp"

#define HLINE std::cout << "--------------------------------" << std::endl

int compute() {
    //spdlog::set_level(spdlog::level::debug);
    global_rand_gen.seed(0);
#ifndef CPU_ONLY
    GPUSampler sampler(1);
#endif

    {
        auto A = M::rand(3, 4);
        std::cout << A << std::endl;
    
        auto B = M::rand(3, 3);
        std::cout << B << std::endl;
        
        auto AB = vstack<float>({A.columns(0, 2).T(), B.columns(0,2).T()}); 
        std::cout << AB << std::endl;

        auto C = M::randn(4, 3);
        std::cout << C << std::endl;
        auto D = elemwise([=](float e) {return e-1; }, C);
        std::cout << D << std::endl;
        

        // auto C = exp(A) + B;
        // std::cout << C << std::endl;

        // auto C2 = A + exp(B);
        // std::cout << C2 << std::endl;

        // HLINE;
        // auto C3 = exp(A+B) + exp(B);
        // std::cout << C3 << std::endl;

    }
  
    return 0;
}
