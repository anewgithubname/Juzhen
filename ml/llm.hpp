#pragma once
#include "layer.hpp"
#include "optimizer.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace Juzhen {
namespace token_detail {
inline size_t dimension(size_t n) {
    if(!n || n>size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("token dimension must be positive and fit int32");
    return n;
}
inline size_t count(size_t rows,size_t cols) {
    if(cols && rows>size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(float)/cols)
        throw std::length_error("token tensor size overflows");
    return rows*cols;
}
#ifdef CUDA
inline unsigned blocks(size_t count) {return unsigned(std::min(size_t(4096),(count-1)/256+1));}
static __global__ void embedding_forward_kernel(float* out,const float* table,const int32_t* ids,
                                                size_t dim,size_t vocab,size_t count,bool transpose) {
    for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(blockDim.x)*gridDim.x) {
        const size_t row=k%dim,id=size_t(ids[k/dim]);
        out[k]=table[transpose?row*vocab+id:id*dim+row];
    }
}
static __global__ void embedding_backward_kernel(float* grad,const float* upstream,const int32_t* ids,
                                                 size_t dim,size_t tokens,size_t count,bool transpose,size_t vocab,bool gradient_transpose) {
    for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(blockDim.x)*gridDim.x) {
        const size_t row=k%dim,token=k/dim;
        const size_t id=size_t(ids[token]);
        atomicAdd(grad+(gradient_transpose?row*vocab+id:id*dim+row),upstream[transpose?row*tokens+token:k]);
    }
}
static __global__ void token_ce_forward_kernel(const float* z,const int32_t* ids,float* maxima,float* lse,float* losses,
                                               size_t vocab,size_t tokens,bool transpose,int32_t ignore) {
    __shared__ float shared[256];
    const size_t j=blockIdx.x;const unsigned t=threadIdx.x;
    const int32_t y=ids[j];
    if(y==ignore){if(!t){maxima[j]=0;lse[j]=0;losses[j]=0;}return;}
    float maximum=-INFINITY;
    for(size_t v=t;v<vocab;v+=blockDim.x)maximum=fmaxf(maximum,z[transpose?v*tokens+j:j*vocab+v]);
    shared[t]=maximum;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride>>=1){if(t<stride)shared[t]=fmaxf(shared[t],shared[t+stride]);__syncthreads();}
    maximum=shared[0];float total=0;
    for(size_t v=t;v<vocab;v+=blockDim.x)total+=expf(z[transpose?v*tokens+j:j*vocab+v]-maximum);
    shared[t]=total;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride>>=1){if(t<stride)shared[t]+=shared[t+stride];__syncthreads();}
    if(!t){const float logsum=logf(shared[0]);maxima[j]=maximum;lse[j]=logsum;losses[j]=(maximum-z[transpose?size_t(y)*tokens+j:j*vocab+size_t(y)])+logsum;}
}
static __global__ void token_ce_backward_kernel(float* dz,const float* z,const int32_t* ids,const float* maxima,const float* lse,
                                                size_t vocab,size_t tokens,size_t count,bool transpose,
                                                int32_t ignore,float scale) {
    for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(blockDim.x)*gridDim.x) {
        const size_t v=k%vocab,j=k/vocab;const int32_t y=ids[j];
        dz[k]=y==ignore?0.f:scale*(expf((z[transpose?v*tokens+j:k]-maxima[j])-lse[j])-(v==size_t(y)?1.f:0.f));
    }
}
#endif
} // namespace token_detail

// Immutable integer IDs. Copies share a device allocation; IDs never pass
// through float storage. Host validation happens before kernels are launched.
template<class D> class TokenIds {
    std::vector<int32_t> host_;
#ifdef CUDA
    mutable std::shared_ptr<int32_t> device_;
#endif
public:
    explicit TokenIds(std::vector<int32_t> ids):host_(std::move(ids)) {
        if(host_.size()>size_t(std::numeric_limits<int32_t>::max()))throw std::length_error("too many token IDs");
    }
    TokenIds(std::initializer_list<int32_t> ids):TokenIds(std::vector<int32_t>(ids)){}
    size_t size() const {return host_.size();}
    const std::vector<int32_t>& host() const {return host_;}
    const int32_t* data() const {
        if constexpr(std::is_same_v<D,float>)return host_.data();
#ifdef CUDA
        else if constexpr(std::is_same_v<D,CUDAfloat>) {
            if(host_.empty())return nullptr;
            if(!device_) {
                int32_t* p=nullptr;CudaErrorCheck(cudaMalloc(&p,host_.size()*sizeof(int32_t)));
                std::shared_ptr<int32_t> owner(p,[](int32_t* x){if(x)CudaErrorCheck(cudaFree(x));});
                CudaErrorCheck(cudaMemcpy(p,host_.data(),host_.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
                device_=std::move(owner);
            }
            return device_.get();
        }
#endif
        else {throw std::logic_error("token operators currently support CPU and CUDA");}
    }
};

// The integer entry point lives outside the float-only network traversal.
// Inheriting Layer retains parameter/checkpoint discovery and W() access.
template<class D> class Embedding : public Layer<D> {
    size_t vocab_,dim_;
    std::unique_ptr<TokenIds<D>> ids_;
    static size_t checked_vocab(size_t vocab,size_t dim) {
        token_detail::dimension(vocab);token_detail::dimension(dim);token_detail::count(dim,vocab);return vocab;
    }
public:
    Embedding(size_t vocab,size_t dim)
        :Layer<D>(token_detail::dimension(dim),checked_vocab(vocab,dim),0),vocab_(vocab),dim_(dim) {}
    using Layer<D>::backward;
    Matrix<D>& weight() {return this->weights;}
    bool trainable() const {return this->need_update;}
    void set_trainable(bool enabled) {this->need_update=enabled;}
    std::vector<ParameterRef<D>> parameters() override {
        return {this->parameter_gradients.parameter("embedding",this->weights,this->adamW,this->need_update)};
    }
    std::vector<std::pair<std::string,Matrix<D>*>> checkpoint_parameters() override {
        return {{"embedding",&this->weights}};
    }
    std::vector<std::pair<std::string,adam_state<D>*>> checkpoint_optimizers() override {
        return {{"adam_embedding",&this->adamW}};
    }
    void accumulate(Matrix<D>&& gradient) {
        if(this->need_update)this->parameter_gradients.add(this->weights,std::move(gradient));
    }
    void eval(const Matrix<D>&) override {throw std::invalid_argument("Embedding requires integer TokenIds");}
    Matrix<D> grad(const Matrix<D>&) const override {
        throw std::logic_error("integer token IDs have no input derivative");
    }
    Matrix<D> jvp(const Matrix<D>&,const Matrix<D>&) override {
        throw std::logic_error("integer token IDs have no input JVP");
    }
    Matrix<D> backward_update(const Matrix<D>&,Matrix<D>&&) override {
        throw std::logic_error("Embedding uses integer eval and explicit backward/step");
    }
    void eval(const TokenIds<D>& ids) {
        ids_.reset();
        if(this->weights.num_row()!=dim_||this->weights.num_col()!=vocab_)throw std::invalid_argument("embedding weight shape mismatch");
        for(int32_t id:ids.host())if(id<0||size_t(id)>=vocab_)throw std::out_of_range("embedding token ID");
        const size_t count=token_detail::count(dim_,ids.size());
        Matrix<D> output("embedding",dim_,ids.size());
        if(count) {
#ifdef CUDA
            if constexpr(std::is_same_v<D,CUDAfloat>) {
                token_detail::embedding_forward_kernel<<<token_detail::blocks(count),256>>>(
                    (float*)output.data(),(const float*)this->weights.data(),ids.data(),dim_,vocab_,count,this->weights.get_transpose());
                CudaErrorCheck(cudaGetLastError());
            } else
#endif
            if constexpr(std::is_same_v<D,float>) {
                for(size_t j=0;j<ids.size();++j)for(size_t i=0;i<dim_;++i)output.elem(i,j)=this->weights.elem(i,size_t(ids.host()[j]));
            } else throw std::logic_error("Embedding currently supports CPU and CUDA");
        }
        this->val=std::move(output);ids_=std::make_unique<TokenIds<D>>(ids);
    }
    void backward(const Matrix<D>& dy) {
        if(!ids_)throw std::logic_error("Embedding backward requires a successful eval");
        if(dy.num_row()!=dim_||dy.num_col()!=ids_->size())throw std::invalid_argument("embedding gradient shape mismatch");
        if(!this->need_update||!ids_->size())return;
        auto& gradient=this->parameter_gradients.buffer(this->weights);
        const size_t count=token_detail::count(dim_,ids_->size());
#ifdef CUDA
        if constexpr(std::is_same_v<D,CUDAfloat>) {
            token_detail::embedding_backward_kernel<<<token_detail::blocks(count),256>>>(
                (float*)gradient.data(),(const float*)dy.data(),ids_->data(),dim_,ids_->size(),count,dy.get_transpose(),vocab_,gradient.get_transpose());
            CudaErrorCheck(cudaGetLastError());
        } else
#endif
        if constexpr(std::is_same_v<D,float>) {
            for(size_t j=0;j<ids_->size();++j)for(size_t i=0;i<dim_;++i)gradient.elem(i,size_t(ids_->host()[j]))+=dy.elem(i,j);
        } else throw std::logic_error("Embedding currently supports CPU and CUDA");
    }
};

// A shared output head has no independent weight/Adam state. Its owner must
// outlive it and be included in checkpoint lists; only that owner is serialized.
template<class D> class TiedProjection : public Layer<D> {
    Embedding<D>& embedding_;
public:
    explicit TiedProjection(Embedding<D>& embedding):Layer<D>(1,1,0),embedding_(embedding) {}
    bool supports_backward() const override {return true;}
    std::vector<ParameterRef<D>> parameters() override {return embedding_.parameters();}
    std::vector<std::pair<std::string,Matrix<D>*>> checkpoint_parameters() override {return {};}
    std::vector<std::pair<std::string,adam_state<D>*>> checkpoint_optimizers() override {return {};}
    void eval(const Matrix<D>& x) override {
        if(x.num_row()!=embedding_.weight().num_row())throw std::invalid_argument("tied projection input shape");
        this->val=embedding_.weight().T()*x;
    }
    Matrix<D> grad(const Matrix<D>& x) const override {
        if(x.num_row()!=embedding_.weight().num_row())throw std::invalid_argument("tied projection input shape");
        return Matrix<D>::ones(embedding_.weight().num_col(),x.num_col());
    }
    Matrix<D> jvp(const Matrix<D>& x,const Matrix<D>& tangent) override {
        if(x.num_row()!=embedding_.weight().num_row()||tangent.num_row()!=x.num_row()||tangent.num_col()!=x.num_col())
            throw std::invalid_argument("tied projection tangent shape");
        return embedding_.weight().T()*tangent;
    }
    Matrix<D> backward_update(const Matrix<D>&,Matrix<D>&&) override {
        throw std::logic_error("TiedProjection requires backward and explicit step");
    }
    Matrix<D> backward(const Matrix<D>& x,Matrix<D>&& dy) override {
        if(x.num_row()!=embedding_.weight().num_row()||dy.num_row()!=embedding_.weight().num_col()||x.num_col()!=dy.num_col())
            throw std::invalid_argument("tied projection gradient shape");
        auto dx=embedding_.weight()*dy;
        if(this->need_update)embedding_.accumulate(x*dy.T());
        return dx;
    }
};

// Mean over valid tokens. The logits are borrowed and must remain alive and
// unchanged until grad(); no full logits/softmax copy is retained by the loss.
template<class D> class TokenCrossEntropy {
    int32_t ignore_;
    const Matrix<D>* logits_=nullptr;
    std::unique_ptr<TokenIds<D>> targets_;
    Matrix<D> maxima_,lse_,value_;
    size_t vocab_=0,tokens_=0,valid_=0;
public:
    explicit TokenCrossEntropy(int32_t ignore_index=-100):ignore_(ignore_index){}
    size_t valid_tokens() const {return valid_;}
    const Matrix<D>& value() const {
        if(!logits_)throw std::logic_error("loss requires eval");return value_;
    }
    void eval(const Matrix<D>&& ,const TokenIds<D>&)=delete;
    void eval(const Matrix<D>& logits,const TokenIds<D>& targets) {
        logits_=nullptr;valid_=0;
        const size_t vocab=logits.num_row(),tokens=logits.num_col();
        token_detail::dimension(vocab);token_detail::count(vocab,tokens);
        if(targets.size()!=tokens)throw std::invalid_argument("token target count mismatch");
        size_t valid=0;
        for(int32_t id:targets.host())if(id!=ignore_) {
            if(id<0||size_t(id)>=vocab)throw std::out_of_range("target token ID");++valid;
        }
        Matrix<D> maxima("token_max",1,tokens),lse("token_lse",1,tokens),losses("token_loss",1,tokens);
        if(tokens) {
#ifdef CUDA
            if constexpr(std::is_same_v<D,CUDAfloat>) {
                token_detail::token_ce_forward_kernel<<<unsigned(tokens),256>>>(
                    (const float*)logits.data(),targets.data(),(float*)maxima.data(),(float*)lse.data(),(float*)losses.data(),vocab,tokens,logits.get_transpose(),ignore_);
                CudaErrorCheck(cudaGetLastError());
            } else
#endif
            if constexpr(std::is_same_v<D,float>) {
                for(size_t j=0;j<tokens;++j) {
                    if(targets.host()[j]==ignore_){maxima.elem(0,j)=0;lse.elem(0,j)=0;losses.elem(0,j)=0;continue;}
                    float maximum=-std::numeric_limits<float>::infinity();
                    for(size_t v=0;v<vocab;++v)maximum=std::max(maximum,logits.elem(v,j));
                    double total=0;for(size_t v=0;v<vocab;++v)total+=std::exp(double(logits.elem(v,j))-maximum);
                    maxima.elem(0,j)=maximum;lse.elem(0,j)=float(std::log(total));
                    losses.elem(0,j)=(maximum-logits.elem(size_t(targets.host()[j]),j))+lse.elem(0,j);
                }
            } else throw std::logic_error("TokenCrossEntropy currently supports CPU and CUDA");
        }
        value_=valid?sum(losses,1)/double(valid):Matrix<D>::zeros(1,1);
        maxima_=std::move(maxima);lse_=std::move(lse);targets_=std::make_unique<TokenIds<D>>(targets);
        vocab_=vocab;tokens_=tokens;valid_=valid;logits_=&logits;
    }
    Matrix<D> grad(float scale=1.f) const {
        if(!logits_||!targets_)throw std::logic_error("loss gradient requires eval");
        if(!std::isfinite(scale))throw std::invalid_argument("non-finite loss gradient scale");
        if(logits_->num_row()!=vocab_||logits_->num_col()!=tokens_)throw std::logic_error("logits shape changed after eval");
        Matrix<D> result("token_dlogits",vocab_,tokens_);
        if(!valid_){result.zeros();return result;}
        const float factor=scale/float(valid_);const size_t count=vocab_*tokens_;
#ifdef CUDA
        if constexpr(std::is_same_v<D,CUDAfloat>) {
            token_detail::token_ce_backward_kernel<<<token_detail::blocks(count),256>>>(
                (float*)result.data(),(const float*)logits_->data(),targets_->data(),(const float*)maxima_.data(),(const float*)lse_.data(),
                vocab_,tokens_,count,logits_->get_transpose(),ignore_,factor);
            CudaErrorCheck(cudaGetLastError());
        } else
#endif
        if constexpr(std::is_same_v<D,float>) {
            for(size_t j=0;j<tokens_;++j)for(size_t v=0;v<vocab_;++v)
                result.elem(v,j)=targets_->host()[j]==ignore_?0.f:factor*(std::exp((logits_->elem(v,j)-maxima_.elem(0,j))-lse_.elem(0,j))-(v==size_t(targets_->host()[j])?1.f:0.f));
        } else throw std::logic_error("TokenCrossEntropy currently supports CPU and CUDA");
        return result;
    }
};

} // namespace Juzhen
