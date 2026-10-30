/** Sequential network with an MSE loss function and a separate Adam optimizer. */
#include "../ml/layer.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include <cmath>
#include <iostream>
using namespace Juzhen;
#ifdef CUDA
using Backend=CUDAfloat;
#else
using Backend=float;
#endif

template<class D> float scalar(const Matrix<D>& x){
    if constexpr(std::is_same_v<D,float>)return x.elem(0,0);
    else return x.to_host().elem(0,0);
}
int compute(){try{
    global_rand_gen.seed(42);
#ifdef CUDA
    GPUSampler sampler(42);
#endif
    Matrix<float> hx("features",2,32),hy("target",1,32);
    for(int j=0;j<32;++j){
        hx.elem(0,j)=std::sin(float(j));hx.elem(1,j)=std::cos(float(j)*.7f);
        hy.elem(0,j)=2.f*hx.elem(0,j)-hx.elem(1,j)+.3f;
    }
    Matrix<Backend> x(hx),y(hy);
    LinearLayer<Backend> linear(1,2,32);
    AdamWOptions options;
    options.lr=.03f;
    options.weight_decay=0; // Adam without weight decay.
    Sequential<Backend> network{&linear};
    AdamW<Backend> optimizer(network.parameters(),options);
    float first=0,last=0;
    for(int step=0;step<300;++step){
        optimizer.zero_grad();
        auto [loss,dprediction]=mse(network.eval(x),y);
        last=scalar(loss);
        if(step==0)first=last;
        network.backward(std::move(dprediction));
        optimizer.step();
    }
    std::cout<<"MSE "<<first<<" -> "<<last<<std::endl;
    if(!std::isfinite(last)||last>=1e-5f)throw std::runtime_error("regression did not converge");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
