/** Masked CE: independent scalar references, scalar seeds, legacy and explicit training. */
#include "../ml/masked_cross_entropy.hpp"
#include "../ml/optimizer.hpp"
#include <iostream>
#include <limits>

using namespace Juzhen;
template<class D> Matrix<float> host(const Matrix<D>& x) {
    if constexpr(std::is_same_v<D,float>)return x;else return x.to_host();
}
static void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
static void close(const Matrix<float>& actual,const Matrix<float>& expected,
                  const std::string& message,float tolerance=3e-5f) {
    require(actual.num_row()==expected.num_row()&&actual.num_col()==expected.num_col(),message+" shape");
    double worst=0;
    for(size_t j=0;j<actual.num_col();++j)for(size_t i=0;i<actual.num_row();++i) {
        const double error=std::abs(double(actual.elem(i,j))-expected.elem(i,j));
        require(std::isfinite(error),message+" finite");worst=std::max(worst,error);
    }
    require(worst<=tolerance,message+" error="+std::to_string(worst));
}
template<class F> void rejects(F function,const std::string& message) {
    bool caught=false;try{function();}catch(const std::logic_error&){caught=true;}require(caught,message);
}
static double reference(const Matrix<float>& x,const Matrix<float>& target,const Matrix<float>& weights) {
    double loss=0;
    for(size_t j=0;j<x.num_col();++j) {
        double maximum=-std::numeric_limits<double>::infinity();
        for(size_t i=0;i<x.num_row();++i)maximum=std::max(maximum,double(x.elem(i,j)));
        double partition=0,dot=0;
        for(size_t i=0;i<x.num_row();++i) {
            partition+=std::exp(double(x.elem(i,j))-maximum);
            dot+=double(x.elem(i,j))*target.elem(i,j);
        }
        loss+=weights.elem(0,j)*(std::log(partition)+maximum-dot);
    }
    return loss;
}
template<class D> void checks(const char* backend) {
    constexpr size_t classes=5,batch=4;
    Matrix<float> logits("logits",classes,batch),target("target",classes,batch),weights("weights",1,batch);
    target.zeros();weights.elem(0,0)=0;weights.elem(0,1)=.125f;weights.elem(0,2)=.75f;weights.elem(0,3)=2;
    for(size_t j=0;j<batch;++j) {
        target.elem((j*2+1)%classes,j)=1;
        for(size_t i=0;i<classes;++i)logits.elem(i,j)=.65f*std::sin(float(3*i+7*j));
    }
    // Soft targets are also valid when normalized.
    target.elem(0,2)=.25f;target.elem(1,2)=.75f;
    Matrix<D> x(logits);MaskedCELayer<D> loss(batch,Matrix<D>(target),Matrix<D>(weights));
    require(loss.supports_backward()&&loss.parameters().empty(),"masked loss has no optimizer parameters");
    require(loss.checkpoint_parameters().size()==2&&loss.checkpoint_optimizers().size()==2,
            "masked loss preserves legacy checkpoint schema");
    loss.eval(x);require(std::abs(host(loss.value()).elem(0,0)-reference(logits,target,weights))<3e-6,"weighted CE scalar reference");
    auto gradient=host(loss.backward(x,Matrix<D>::ones(1,1)));
    Matrix<float> expected("finite differences",classes,batch);
    for(size_t j=0;j<batch;++j)for(size_t i=0;i<classes;++i) {
        float original=logits.elem(i,j);logits.elem(i,j)=original+.001f;
        const double upper=logits.elem(i,j),plus=reference(logits,target,weights);
        logits.elem(i,j)=original-.001f;const double lower=logits.elem(i,j),minus=reference(logits,target,weights);
        logits.elem(i,j)=original;expected.elem(i,j)=float((plus-minus)/(upper-lower));
    }
    close(gradient,expected,"masked CE independent finite differences",2e-6f);
    for(size_t i=0;i<classes;++i)require(gradient.elem(i,0)==0,"unmasked token gradient is zero");
    close(host(loss.backward(x,Matrix<D>::ones(1,1)*-2.5f)),expected*-2.5f,"masked CE scalar upstream",6e-6f);
    close(host(loss.backward_update(x,Matrix<D>::ones(1,1))),gradient,"masked loss direct legacy API",0);
    std::list<Layer<D>*> legacy_head{&loss};
    close(host(Juzhen::backward_update(legacy_head,x)),gradient,"legacy loss head uses grad",0);
    Sequential<D> head{&loss};head.eval(x);close(host(head.backward()),gradient,"Sequential masked loss",0);
    AdamW<D> empty_optimizer(head.parameters());require(empty_optimizer.step()==0,"parameter-free loss optimizer");
    close(host(x),logits,"masked loss preserves logits",0);
    rejects([&]{loss.eval(Matrix<D>::ones(classes-1,batch));},"masked CE rejects input shape");
    rejects([&]{loss.grad(Matrix<D>::ones(classes,batch-1));},"masked CE rejects gradient input shape");
    rejects([&]{loss.backward(x,Matrix<D>::ones(1,2));},"masked CE rejects non-scalar upstream");
    rejects([&]{loss.jvp(x,x);},"masked loss rejects network activation JVP");
    rejects([&]{MaskedCELayer<D> invalid(batch,Matrix<D>(target),Matrix<D>::ones(batch,1));},"masked CE rejects weight shape");
    rejects([&]{MaskedCELayer<D> invalid(batch-1,Matrix<D>(target),Matrix<D>(weights));},"masked CE rejects batch mismatch");
    for(float invalid:{-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        auto bad=weights;bad.elem(0,0)=invalid;
        rejects([&]{MaskedCELayer<D> invalid_loss(batch,Matrix<D>(target),Matrix<D>(bad));},"masked CE rejects invalid weights");
    }
    MaskedCELayer<D> zero(batch,Matrix<D>(target),Matrix<D>::zeros(1,batch));zero.eval(x);
    require(host(zero.value()).elem(0,0)==0,"all-unmasked zero loss");
    close(host(zero.backward(x,Matrix<D>::ones(1,1))),Matrix<float>::zeros(classes,batch),"all-unmasked zero gradient",0);
    for(size_t j=0;j<batch;++j)for(size_t i=0;i<classes;++i)logits.elem(i,j)=1000.f*float(i)-2000.f;
    Matrix<D> extreme(logits);loss.eval(extreme);
    require(std::abs(host(loss.value()).elem(0,0)-reference(logits,target,weights))<.001,"large logits stable CE");
    auto stable_gradient=host(loss.grad(extreme));
    for(size_t j=0;j<batch;++j)for(size_t i=0;i<classes;++i)
        require(std::isfinite(stable_gradient.elem(i,j)),"large logits finite gradient");
    // Exercise masked loss in a real trainable chain and compare the legacy Adam update.
    LinearLayer<D> layer(classes,3,batch),old_layer(classes,3,batch);
    old_layer.W()=Matrix<D>(layer.W());old_layer.b()=Matrix<D>(layer.b());
    auto input=Matrix<D>::randn(3,batch);
    MaskedCELayer<D> train_loss(batch,Matrix<D>(target),Matrix<D>(weights));
    MaskedCELayer<D> old_loss(batch,Matrix<D>(target),Matrix<D>(weights));
    Sequential<D> network{&train_loss,&layer};std::list<Layer<D>*> old_network{&old_loss,&old_layer};
    AdamWOptions options;options.weight_decay=0;AdamW<D> optimizer(network.parameters(),options);
    optimizer.zero_grad();network.eval(input);auto dx=host(network.backward());
    Juzhen::forward(old_network,input);close(host(Juzhen::backward_update(old_network,input)),dx,"masked training legacy input gradient");
    require(optimizer.step()==2,"masked training registers only linear parameters");
    close(host(layer.W()),host(old_layer.W()),"masked training legacy Adam weights",2e-6f);
    close(host(layer.b()),host(old_layer.b()),"masked training legacy Adam bias",2e-6f);
    std::cout<<"[PASS] "<<backend<<" masked loss API\n";
}
int compute() {
    try {
        global_rand_gen.seed(927);checks<float>("CPU");
#ifdef CUDA
        GPUSampler sampler(927);checks<CUDAfloat>("CUDA");CudaErrorCheck(cudaDeviceSynchronize());
#endif
        return 0;
    } catch(const std::exception& error) {std::cerr<<"[FAIL] "<<error.what()<<'\n';return 1;}
}
