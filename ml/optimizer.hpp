#pragma once
#include "parameter.hpp"
#include "optimizer_cuda.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Juzhen {
struct AdamWOptions {
    float lr=1e-4f, beta1=.9f, beta2=.999f, eps=1e-8f, weight_decay=.01f;
};

#ifdef CUDA
static __global__ void explicit_adamw_kernel(
    float* w,const float* g,float* m,float* v,size_t rows,size_t cols,
    bool wt,bool gt,bool mt,bool vt,float lr,float beta1,float beta2,
    float eps,float bc1,float bc2,float decay) {
    const size_t count=rows*cols;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        const size_t ti=(i%rows)*cols+i/rows;
        const size_t wi=wt?ti:i,gi=gt?ti:i,mi=mt?ti:i,vi=vt?ti:i;
        const float grad=g[gi];
        const float first=beta1*m[mi]+(1-beta1)*grad;
        const float second=beta2*v[vi]+(1-beta2)*grad*grad;
        m[mi]=first;v[vi]=second;
        w[wi]=w[wi]*(1-lr*decay)-lr*(first*bc1)/(sqrtf(second*bc2)+eps);
    }
}
#endif

// Adam states are borrowed from layers, retaining the existing checkpoint
// representation. Shared parameters must reuse the same ParameterRef storage.
template<class D> class AdamW {
    std::vector<ParameterRef<D>> parameters_;
    float decay_;
#ifdef CUDA
    std::shared_ptr<optimizer_detail::CudaNormScratch> cuda_norm_;
#endif
    static void valid_options(const AdamWOptions& o) {
        if(!std::isfinite(o.lr)||o.lr<0||!std::isfinite(o.beta1)||o.beta1<0||o.beta1>=1||
           !std::isfinite(o.beta2)||o.beta2<0||o.beta2>=1||!std::isfinite(o.eps)||o.eps<=0||
           !std::isfinite(o.weight_decay)||o.weight_decay<0)
            throw std::invalid_argument("invalid AdamW options");
    }
    void validate() const {
        std::unordered_map<const void*,const Matrix<D>*> storage;
        for(const auto& p:parameters_) {
            if(!p.enabled() || !p.has_grad())continue;
            const auto& w=*p.value;const auto& g=p.grad();const auto& a=*p.state;
            const size_t rows=w.num_row(),cols=w.num_col();
            if(!rows||!cols||g.num_row()!=rows||g.num_col()!=cols||a.m.num_row()!=rows||
               a.m.num_col()!=cols||a.v.num_row()!=rows||a.v.num_col()!=cols)
                throw std::invalid_argument("AdamW parameter/gradient/state shape mismatch");
            if(a.iteration<=0 || a.iteration==std::numeric_limits<int>::max())
                throw std::invalid_argument("invalid Adam iteration");
            valid_options(AdamWOptions{a.alpha,a.beta1,a.beta2,a.eps,decay_});
            auto [it,inserted]=storage.emplace(w.data(),p.value);
            if(!inserted && it->second!=p.value)
                throw std::invalid_argument("aliased storage must share one parameter registration");
        }
    }
public:
    explicit AdamW(std::vector<ParameterRef<D>> parameters,AdamWOptions options={})
        :decay_(options.weight_decay) {
        static_assert(std::is_same_v<D,float>
#ifdef CUDA
                      || std::is_same_v<D,CUDAfloat>
#endif
                      ,"Explicit AdamW currently supports CPU float and CUDAfloat");
        valid_options(options);
        std::unordered_map<Matrix<D>*,size_t> unique;
        for(auto p:parameters) {
            if(!p.value||!p.gradient||!p.state)throw std::invalid_argument("null parameter registration");
            auto [it,inserted]=unique.emplace(p.value,parameters_.size());
            if(inserted)parameters_.push_back(p);
            else {
                const auto& prior=parameters_[it->second];
                if(prior.gradient!=p.gradient||prior.state!=p.state||prior.trainable!=p.trainable||
                   prior.parent_trainable!=p.parent_trainable||prior.decay!=p.decay)
                    throw std::invalid_argument("inconsistent shared parameter registration");
            }
        }
        for(auto& p:parameters_) {
            p.state->alpha=options.lr;p.state->beta1=options.beta1;
            p.state->beta2=options.beta2;p.state->eps=options.eps;
        }
    }
    const std::vector<ParameterRef<D>>& parameters() const {return parameters_;}
    void zero_grad() {for(const auto& p:parameters_)p.zero_grad();}
    void set_lr(float lr) {
        if(!std::isfinite(lr)||lr<0)throw std::invalid_argument("invalid learning rate");
        for(auto& p:parameters_)p.state->alpha=lr;
    }
    // Norms are computed before any gradients are scaled. NaN/Inf is rejected
    // without changing gradients or parameters.
    double clip_grad_norm(double max_norm) {
        if(!std::isfinite(max_norm)||max_norm<0)throw std::invalid_argument("invalid gradient norm limit");
        validate();
#ifdef CUDA
        if constexpr(std::is_same_v<D,CUDAfloat>){
            if(!cuda_norm_)cuda_norm_=std::make_shared<optimizer_detail::CudaNormScratch>();
            return cuda_norm_->clip(parameters_,max_norm);
        }
#endif
        double norm=0;
        for(const auto& p:parameters_)if(p.enabled()&&p.has_grad())
            norm=std::hypot(norm,double(p.grad().norm()));
        if(!std::isfinite(norm))throw std::runtime_error("non-finite gradient norm");
        const double scale=std::min(1.0,max_norm/(norm+1e-6));
        if(scale<1)for(auto& p:parameters_)if(p.enabled()&&p.has_grad())p.gradient->value.scale(scale);
        return norm;
    }
    size_t step() {
        validate();size_t updated=0;
        for(auto& p:parameters_) {
            if(!p.enabled()||!p.has_grad())continue;
            auto& a=*p.state;
            const float bc1=1.f/(1.f-std::pow(a.beta1,float(a.iteration)));
            const float bc2=1.f/(1.f-std::pow(a.beta2,float(a.iteration)));
            const float decay=p.decay?decay_:0.f;
#ifdef CUDA
            if constexpr(std::is_same_v<D,CUDAfloat>) {
                const size_t count=p.value->num_row()*p.value->num_col();
                const unsigned blocks=unsigned(std::min(size_t(4096),(count-1)/256+1));
                explicit_adamw_kernel<<<blocks,256>>>(
                    (float*)p.value->data(),(const float*)p.grad().data(),(float*)a.m.data(),(float*)a.v.data(),
                    p.value->num_row(),p.value->num_col(),p.value->get_transpose(),p.grad().get_transpose(),
                    a.m.get_transpose(),a.v.get_transpose(),a.alpha,a.beta1,a.beta2,a.eps,bc1,bc2,decay);
                CudaErrorCheck(cudaGetLastError());
            } else
#endif
            {
                for(size_t j=0;j<p.value->num_col();++j)for(size_t i=0;i<p.value->num_row();++i) {
                    const float g=p.grad().elem(i,j);
                    const float m=a.beta1*a.m.elem(i,j)+(1-a.beta1)*g;
                    const float v=a.beta2*a.v.elem(i,j)+(1-a.beta2)*g*g;
                    a.m.elem(i,j)=m;a.v.elem(i,j)=v;
                    p.value->elem(i,j)=p.value->elem(i,j)*(1-a.alpha*decay)-a.alpha*(m*bc1)/(std::sqrt(v*bc2)+a.eps);
                }
            }
            ++a.iteration;++updated;
        }
        return updated;
    }
};
} // namespace Juzhen
