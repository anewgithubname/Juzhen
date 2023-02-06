#include "../cpp/juzhen.hpp"

int test1(){
    // unit test started
    M A = {"A", {{1,2,3}, {3,4,5}}};
    M B = {"B", {{6,7,8}, {9,10,11}}};

    // compute log(exp(A)+exp(B)) + AT*B(0:2, :) on CPU
    auto C = log(exp(-A/B)+exp(hadmd(B,A))) - (A.T()*B).rows(0,2);
    std::cout << "C = " << C << std::endl; 

    M bench = read<float>("../tests/unittest.matrix");

    if ((C - bench ).norm() < 1e-5){
        LOG_INFO("unit test passed!");
        return 0;
    }
    else{
        LOG_ERROR("unit test failed!");
        return 1;
    }

}

int test2(){    
    // unit test started
    M AA = {"A", {{1,2,3}, {3,4,5}}};
    M BB = {"B", {{6,7,8}, {9,10,11}}};
    
    CM A(AA);
    CM B(BB);

    M bench = read<float>("../tests/unittest.matrix");

    // compute log(exp(A)+exp(B)) + AT*B(0:2, :) on GPU
    auto C = log(exp(-A/B)+exp(hadmd(B,A))) - (A.T()*B).rows(0,2);
    std::cout << "C = " << C << std::endl;

    if ((C- (CM) bench ).norm() < 1e-5){
        LOG_INFO("unit test passed!");
        return 0;
    }
    else{
        LOG_ERROR("unit test failed!");
        return 1;
    }
}

int test3(){
    // unit test started
    M A = {"A", {{1,2,3}, {3,4,5}}};
    M B = {"B", {{6,7,8}, {9,10,11}}};
    
    int ret = 0;
    std::vector<Matrix<float>> vecA = {A,B};
    if((vstack(vecA) - M("C", {{1,2,3}, {3,4,5}, {6,7,8}, {9,10,11}})).norm() > 1e-5){
        LOG_ERROR("vstack failed!");
        ret += 1; 
    }

    if((hstack(vecA) - M("C", {{1,2,3,6,7,8}, {3,4,5,9,10,11}})).norm() > 1e-5){
        LOG_ERROR("hstack failed!");
        ret += 1; 
    }

    LOG_INFO("unit test passed!");
    return ret;
}

int test4(){
    // unit test started
    M AA = {"A", {{1,2,3}, {3,4,5}}};
    M BB = {"B", {{6,7,8}, {9,10,11}}};
    
    CM A(AA);
    CM B(BB);
    
    int ret = 0;
    std::vector<Matrix<CUDAfloat>> vecA = {A,B};
    if((vstack(vecA).to_host() - M("C", {{1,2,3}, {3,4,5}, {6,7,8}, {9,10,11}})).norm() > 1e-5){
        LOG_ERROR("vstack failed!");
        ret += 1;
    }

    vecA = {A,B};
    if((hstack(vecA).to_host() - M("C", {{1,2,3,6,7,8}, {3,4,5,9,10,11}})).norm() > 1e-5){
        LOG_ERROR("hstack failed!");
        std::cout << hstack(vecA).to_host() << std::endl;
        ret += 1;
    }

    LOG_INFO("unit test passed!");
    return ret;
}

int compute()
{
    spdlog::set_level(spdlog::level::debug);
    
    int ret = 0;
    ret += test1();
    std::cout << std:: endl;

    ret += test2();
    std::cout << std:: endl;

    ret += test3();
    std::cout << std:: endl;

    ret += test4();

    if(ret ==0){
        LOG_INFO("--------------------");
        LOG_INFO("|      ALL OK!     |");
        LOG_INFO("--------------------");
    }
    else {
        LOG_ERROR("--------------------");
        LOG_ERROR("|    NOT ALL OK!   |");
        LOG_ERROR("--------------------");
    }
    return ret;
}