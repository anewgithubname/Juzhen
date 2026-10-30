// Independent scalar gradients and compatibility checks for the unified layer API.
#include "../cpp/juzhen.hpp"
#include "../ml/loss.hpp"
#include "../ml/optimizer.hpp"
#include "../ml/checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

using namespace Juzhen;
#if defined(ROCM_HIP) || defined(APPLE_SILICON) || (defined(CUDA) && !defined(CUDNN_AVAILABLE))
int compute() { return 77; }
#else
#ifdef CUDA
using D = CUDAfloat;
#else
using D = float;
#endif

static void require(bool ok,const std::string& tag) { if(!ok) throw std::runtime_error(tag); }
template<class T> static Matrix<float> host(const Matrix<T>& x) {
    if constexpr(std::is_same_v<T,float>) return x; else return x.to_host();
}
static Matrix<float> values(int rows,int cols,float phase,float scale=.2f) {
    Matrix<float> x("fixture",rows,cols);
    for(int j=0;j<cols;++j) for(int i=0;i<rows;++i)
        x.elem(i,j)=scale*std::sin(float(i+rows*j)*.63f+phase);
    return x;
}
static void close(const Matrix<float>& a,const Matrix<float>& b,const std::string& tag,float atol=3e-5f) {
    require(a.num_row()==b.num_row()&&a.num_col()==b.num_col(),tag+" shape");
    for(size_t j=0;j<a.num_col();++j) for(size_t i=0;i<a.num_row();++i) {
        const float error=std::fabs(a.elem(i,j)-b.elem(i,j));
        const float tolerance=atol==0?0:atol+2e-4f*std::fabs(b.elem(i,j));
        require(std::isfinite(error)&&error<=tolerance,tag+" max_abs="+std::to_string(error)+" tolerance="+std::to_string(tolerance));
    }
}
template<class F> static void rejects(F fn,const std::string& tag) {
    bool rejected=false;try {fn();}catch(const std::logic_error&){rejected=true;}
    require(rejected,tag+" must reject");
}
template<class F> static Matrix<float> finite_difference(Matrix<float>& x,F objective) {
    Matrix<float> result("finite difference",x.num_row(),x.num_col());
    for(size_t j=0;j<x.num_col();++j) for(size_t i=0;i<x.num_row();++i) {
        const float original=x.elem(i,j);
        x.elem(i,j)=original+1e-3f;const double plus=objective();const float upper=x.elem(i,j);
        x.elem(i,j)=original-1e-3f;const double minus=objective();const float lower=x.elem(i,j);
        x.elem(i,j)=original;result.elem(i,j)=float((plus-minus)/(double(upper)-lower));
    }
    return result;
}
template<class T> static void require_empty_storage(const Matrix<T>& matrix,const std::string& tag) {
    require(matrix.num_row()==0&&matrix.num_col()==0&&matrix.data()==nullptr,tag+" has no allocation");
}
template<class T> static void empty_matrix_checks() {
    Matrix<T> empty(nullptr);require_empty_storage(empty,"explicit empty matrix");
    Matrix<T> historical_default;
    if constexpr(std::is_same_v<T,float>)require_empty_storage(historical_default,"unchanged host default matrix");
#ifdef CUDA
    if constexpr(std::is_same_v<T,CUDAfloat>)
        require(historical_default.num_row()==2&&historical_default.num_col()==2&&historical_default.data()!=nullptr,"unchanged CUDA default matrix");
#endif
    const auto fixture=values(3,2,.4f);Matrix<T> source(fixture);
    empty=source;close(host(empty),fixture,"empty matrix copy assignment",0);
    empty=Matrix<T>::ones(2,3);close(host(empty),Matrix<float>::ones(2,3),"empty matrix reused with new shape",0);
    Matrix<T> moved_into(nullptr);require_empty_storage(moved_into,"second explicit empty matrix");
    moved_into=std::move(source);close(host(moved_into),fixture,"empty matrix move assignment",0);
    moved_into=Matrix<T>(fixture*-.5f);close(host(moved_into),fixture*-.5f,"empty matrix reused after move assignment",0);
}

static void accumulation_and_freeze(Layer<D>& layer,const Matrix<float>& xh,const Matrix<float>& gh,
                                    const Matrix<float>& expected_dx,const std::string& tag) {
    auto refs=layer.parameters();AdamWOptions options;options.weight_decay=0;
    AdamW<D> optimizer(refs,options);optimizer.zero_grad();Matrix<D> x(xh);
    layer.eval(x);layer.backward(x,Matrix<D>(gh));
    std::vector<Matrix<float>> first;
    for(const auto& p:refs) {require(p.has_grad(),tag+" gradient ready");first.push_back(host(p.grad()));}
    layer.backward(x,Matrix<D>(gh));
    for(size_t i=0;i<refs.size();++i) close(host(refs[i].grad()),first[i]*2,tag+" accumulate");
    optimizer.zero_grad();for(auto& p:refs) require(!p.has_grad(),tag+" zero_grad absent");
    require(optimizer.step()==0,tag+" empty step");
    std::list<Layer<D>*> net{&layer};freeze(net);
    auto old_w=host(layer.W()),old_b=host(layer.b());const int iteration=layer.adamWstate().iteration;
    const auto old_m=host(layer.adamWstate().m),old_v=host(layer.adamWstate().v);
    close(host(layer.backward(x,Matrix<D>(gh))),expected_dx,tag+" frozen dx");
    require(optimizer.step()==0,tag+" frozen step");
    for(auto& p:refs) require(!p.has_grad(),tag+" frozen gradient absent");
    close(host(layer.W()),old_w,tag+" frozen W",0);close(host(layer.b()),old_b,tag+" frozen b",0);
    require(layer.adamWstate().iteration==iteration,tag+" frozen Adam iteration");unfreeze(net);
    close(host(layer.adamWstate().m),old_m,tag+" frozen first moment",0);
    close(host(layer.adamWstate().v),old_v,tag+" frozen second moment",0);
}
static void dense_checks(bool relu_activation) {
    constexpr int ni=3,no=2,batch=3;
    std::unique_ptr<Layer<D>> a=relu_activation?std::unique_ptr<Layer<D>>(new ReluLayer<D>(no,ni,batch)):
        std::make_unique<Layer<D>>(no,ni,batch);
    auto w=values(no,ni,.7f),b=values(no,1,1.1f),x=values(ni,batch,.2f),g=values(no,batch,1.8f);
    b.elem(0,0)=.6f;b.elem(1,0)=-.6f; // ReLU finite differences stay away from its kink.
    auto objective=[&] {
        double result=0;
        for(int j=0;j<batch;++j) for(int o=0;o<no;++o) {
            double z=b.elem(o,0);for(int i=0;i<ni;++i)z+=double(w.elem(o,i))*x.elem(i,j);
            result+=(relu_activation?std::max(0.,z):std::tanh(z))*g.elem(o,j);
        }return result;
    };
    const auto dw=finite_difference(w,objective),db=finite_difference(b,objective),dx=finite_difference(x,objective);
    a->W()=Matrix<D>(w);a->b()=Matrix<D>(b);Matrix<D> input(x);a->eval(input);
    auto refs=a->parameters();require(a->supports_backward()&&refs.size()==2,"dense supported");
    close(host(a->backward(input,Matrix<D>(g))),dx,"dense dx finite difference");
    close(host(refs[0].grad()),dw,"dense dW finite difference");close(host(refs[1].grad()),db,"dense db finite difference");
    close(host(a->W()),w,"dense backward preserves W",0);close(host(a->b()),b,"dense backward preserves b",0);
    require(refs[0].state->iteration==1,"dense backward preserves Adam");
    accumulation_and_freeze(*a,x,g,dx,"dense");
    rejects([&]{a->backward(input,Matrix<D>::ones(no+1,batch));},"dense gradient shape");
}
static void logistic_checks() {
    constexpr int classes=4,batch=3;
    auto x=values(classes,batch,.4f,2.f);x.elem(0,0)=1000;x.elem(1,0)=999;x.elem(2,0)=-1000;
    auto target=Matrix<float>::zeros(classes,batch);
    // Soft labels test the full normalized-target CE contract as well as one-hot labels.
    target.elem(1,0)=1;target.elem(0,1)=.25f;target.elem(3,1)=.75f;target.elem(2,2)=1;
    Matrix<D> input(x),labels(target);
    auto [loss,gradient]=softmax_cross_entropy(input,labels);
    double reference=0;Matrix<float> dx("CE reference",classes,batch);
    for(int j=0;j<batch;++j) {
        double mx=-1e30;for(int i=0;i<classes;++i) mx=std::max(mx,double(x.elem(i,j)));
        double denom=0;for(int i=0;i<classes;++i)denom+=std::exp(double(x.elem(i,j))-mx);
        for(int i=0;i<classes;++i) {
            reference-=double(target.elem(i,j))*(double(x.elem(i,j))-mx-std::log(denom))/batch;
            dx.elem(i,j)=float((std::exp(double(x.elem(i,j))-mx)/denom-target.elem(i,j))/batch);
        }
    }
    require(std::abs(host(loss).elem(0,0)-reference)<5e-5,"stable CE scalar reference");
    close(host(gradient),dx,"stable CE gradient");
    rejects([&]{softmax_cross_entropy(Matrix<D>::zeros(classes+1,batch),labels);},"CE input shape");
}

struct Shape {int n,ci,h,w,co,kh,kw,pad,stride;bool relu;};
static int out_h(const Shape& s,bool trans) {return trans?(s.h-1)*s.stride-2*s.pad+s.kh:(s.h+2*s.pad-s.kh)/s.stride+1;}
static int out_w(const Shape& s,bool trans) {return trans?(s.w-1)*s.stride-2*s.pad+s.kw:(s.w+2*s.pad-s.kw)/s.stride+1;}
static std::unique_ptr<Layer<D>> convolution(const Shape& s,bool trans) {
    if(trans)return std::make_unique<ConvTransLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu,true);
    return std::make_unique<ConvLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu,true);
}
struct ConvOracle {Matrix<float> y,dx,dw,db;};
static ConvOracle scalar_conv(const Shape& s,bool trans,const Matrix<float>& x,const Matrix<float>& weight,
                              const Matrix<float>& bias,const Matrix<float>& dy) {
    const int H=out_h(s,trans),W=out_w(s,trans),ny=s.co*H*W,nx=s.ci*s.h*s.w,nw=s.ci*s.co*s.kh*s.kw;
    std::vector<double> y(ny*s.n,0),dx(nx*s.n,0),dw(nw,0),db(s.co,0),g(ny*s.n);
    auto edges=[&](auto fn) {
        for(int n=0;n<s.n;++n) for(int ci=0;ci<s.ci;++ci) for(int co=0;co<s.co;++co)
        for(int h=0;h<(trans?s.h:H);++h) for(int w=0;w<(trans?s.w:W);++w)
        for(int kh=0;kh<s.kh;++kh) for(int kw=0;kw<s.kw;++kw) {
            int a=h*s.stride-s.pad+kh,b=w*s.stride-s.pad+kw;
            if(a<0||b<0||a>=(trans?H:s.h)||b>=(trans?W:s.w))continue;
            int xi=trans?(ci*s.h+h)*s.w+w:(ci*s.h+a)*s.w+b;
            int yi=trans?(co*H+a)*W+b:(co*H+h)*W+w;
            int wi=((trans?ci*s.co+co:co*s.ci+ci)*s.kh+kh)*s.kw+kw;
            fn(n,xi,yi,wi);
        }
    };
    edges([&](int n,int xi,int yi,int wi){y[n*ny+yi]+=double(x.elem(xi,n))*weight.elem(wi,0);});
    for(int n=0;n<s.n;++n)for(int co=0;co<s.co;++co)for(int i=0;i<H*W;++i) {
        int idx=n*ny+co*H*W+i;double z=y[idx]+bias.elem(co,0);
        y[idx]=s.relu?std::max(0.,z):z;g[idx]=(s.relu&&z<=0)?0:dy.elem(co*H*W+i,n);db[co]+=g[idx];
    }
    edges([&](int n,int xi,int yi,int wi){dx[n*nx+xi]+=g[n*ny+yi]*weight.elem(wi,0);dw[wi]+=g[n*ny+yi]*x.elem(xi,n);});
    ConvOracle r{Matrix<float>("y",ny,s.n),Matrix<float>("dx",nx,s.n),Matrix<float>("dw",nw,1),Matrix<float>("db",s.co,1)};
    for(int n=0;n<s.n;++n) {
        for(int i=0;i<ny;++i)r.y.elem(i,n)=float(y[n*ny+i]);for(int i=0;i<nx;++i)r.dx.elem(i,n)=float(dx[n*nx+i]);
    }
    for(int i=0;i<nw;++i)r.dw.elem(i,0)=float(dw[i]);for(int i=0;i<s.co;++i)r.db.elem(i,0)=float(db[i]);return r;
}
static void convolution_checks(const Shape& s,bool trans) {
    auto a=convolution(s,trans);
    auto w=values(s.ci*s.co*s.kh*s.kw,1,.3f),b=values(s.co,1,.7f),x=values(s.ci*s.h*s.w,s.n,.2f),
         g=values(s.co*out_h(s,trans)*out_w(s,trans),s.n,1.4f);
    auto expected=scalar_conv(s,trans,x,w,b,g);a->W()=Matrix<D>(w);a->b()=Matrix<D>(b);
    Matrix<D> input(x);a->eval(input);close(host(a->value()),expected.y,"convolution scalar forward");
    auto refs=a->parameters();require(a->supports_backward()&&refs.size()==2,"convolution supported");
    require(refs[0].decay&&!refs[1].decay,"convolution only weights use weight decay");
    const auto old_m=host(a->adamWstate().m),old_v=host(a->adamWstate().v);
    const auto old_bm=host(a->adambstate().m),old_bv=host(a->adambstate().v);
    close(host(a->backward(input,Matrix<D>(g))),expected.dx,"convolution scalar dx");
    close(host(refs[0].grad()),expected.dw,"convolution scalar dw");close(host(refs[1].grad()),expected.db,"convolution scalar db");
    close(host(a->W()),w,"convolution backward preserves W",0);close(host(a->b()),b,"convolution backward preserves b",0);
    require(refs[0].state->iteration==1,"convolution backward preserves Adam");
    close(host(a->adamWstate().m),old_m,"convolution pure backward first moment",0);
    close(host(a->adamWstate().v),old_v,"convolution pure backward second moment",0);
    close(host(a->adambstate().m),old_bm,"convolution pure backward bias first moment",0);
    close(host(a->adambstate().v),old_bv,"convolution pure backward bias second moment",0);
    accumulation_and_freeze(*a,x,g,expected.dx,trans?"transpose":"convolution");
    for(auto& p:refs)p.zero_grad();
    rejects([&]{a->backward(input,Matrix<D>::ones(g.num_row()+1,g.num_col()));},"convolution upstream shape");
    rejects([&]{a->backward(Matrix<D>::ones(x.num_row()+1,x.num_col()),Matrix<D>(g));},"convolution backward input shape");
    rejects([&]{a->jvp(input,Matrix<D>::ones(x.num_row(),x.num_col()+1));},"convolution tangent shape");
    rejects([&]{a->eval(Matrix<D>::ones(x.num_row()+1,x.num_col()));},"convolution forward input shape");
    rejects([&]{a->eval(Matrix<D>::ones(x.num_row(),x.num_col()+1));},"convolution forward batch shape");
    for(auto& p:refs)require(!p.has_grad(),"convolution shape errors do not create gradients");
#ifdef CUDA
    // cuDNN consumes contiguous NCHW memory; a logically shape-compatible
    // lazy transpose must be rejected before interpreting that memory.
    const auto before_w=host(a->W()),before_b=host(a->b()),before_m=host(a->adamWstate().m),before_v=host(a->adamWstate().v);
    const int before_iteration=a->adamWstate().iteration;
    auto transposed_input=Matrix<D>::zeros(x.num_col(),x.num_row()).T();
    auto transposed_output=Matrix<D>::zeros(g.num_col(),g.num_row()).T();
    rejects([&]{a->eval(transposed_input);},"GPU convolution transposed input");
    rejects([&]{a->backward(transposed_input,Matrix<D>(g));},"GPU convolution transposed backward input");
    a->eval(input);
    rejects([&]{a->jvp(input,transposed_input);},"GPU convolution transposed tangent");
    rejects([&]{a->backward(input,Matrix<D>(transposed_output));},"GPU convolution transposed upstream");
    a->W()=Matrix<D>::zeros(1,before_w.num_row()).T();
    rejects([&]{a->eval(input);},"GPU convolution transposed weights");a->W()=Matrix<D>(before_w);
    a->b()=Matrix<D>::zeros(1,before_b.num_row()).T();
    rejects([&]{a->eval(input);},"GPU convolution transposed bias");a->b()=Matrix<D>(before_b);
    require(a->adamWstate().iteration==before_iteration,"GPU view errors preserve Adam iteration");
    close(host(a->adamWstate().m),before_m,"GPU view errors preserve first moment",0);
    close(host(a->adamWstate().v),before_v,"GPU view errors preserve second moment",0);
    close(host(a->W()),before_w,"GPU view errors preserve W",0);close(host(a->b()),before_b,"GPU view errors preserve b",0);
    for(auto& p:refs)require(!p.has_grad(),"GPU view errors do not create gradients");
#endif
    auto fresh=convolution(s,trans);fresh->W()=Matrix<D>(w);fresh->b()=Matrix<D>(b);
    if(s.relu) {
        rejects([&]{fresh->backward(input,Matrix<D>(g));},"ReLU convolution needs eval before backward");
        rejects([&]{fresh->jvp(input,Matrix<D>(input));},"ReLU convolution needs eval before JVP");
    } else {
        close(host(fresh->backward(input,Matrix<D>(g))),expected.dx,"linear convolution backward without eval");
        const auto tangent=scalar_conv(s,trans,x,w,Matrix<float>::zeros(s.co,1),g).y;
        close(host(fresh->jvp(input,Matrix<D>(input))),tangent,"linear convolution JVP without eval");
    }
}

static void normalization_checks() {
    const auto rng_before=global_rand_gen;
    constexpr int features=5,batch=3;LayerNormLayer<D> layer(features,batch);
    require(global_rand_gen==rng_before,"normalization construction does not consume host RNG");
    require_empty_storage(layer.W(),"normalization inherited W");require_empty_storage(layer.b(),"normalization inherited b");
    require_empty_storage(layer.adamWstate().m,"normalization inherited W first moment");
    require_empty_storage(layer.adamWstate().v,"normalization inherited W second moment");
    require_empty_storage(layer.adambstate().m,"normalization inherited b first moment");
    require_empty_storage(layer.adambstate().v,"normalization inherited b second moment");
    auto checkpoint_tensors=layer.checkpoint_parameters();auto checkpoint_states=layer.checkpoint_optimizers();
    require(checkpoint_tensors.size()==2&&checkpoint_tensors[0].first=="gamma"&&checkpoint_tensors[0].second==&layer.gamma&&
            checkpoint_tensors[1].first=="beta"&&checkpoint_tensors[1].second==&layer.beta,"normalization checkpoints contain only gamma and beta");
    require(checkpoint_states.size()==2&&checkpoint_states[0].first=="adam_gamma"&&checkpoint_states[0].second==&layer.adam_g&&
            checkpoint_states[1].first=="adam_beta"&&checkpoint_states[1].second==&layer.adam_b,"normalization checkpoint Adam names preserved");
#ifdef CUDA
    auto reset_rng=[] {
        require(curandSetPseudoRandomGeneratorSeed(GPUSampler::gen,773)==CURAND_STATUS_SUCCESS,"reset CUDA RNG seed");
        require(curandSetGeneratorOffset(GPUSampler::gen,0)==CURAND_STATUS_SUCCESS,"reset CUDA RNG offset");
    };
    reset_rng();auto expected_random=host(Matrix<D>::randn(16,16));reset_rng();
    LayerNormLayer<D> no_random(features,batch);
    close(host(Matrix<D>::randn(16,16)),expected_random,"normalization construction does not consume CUDA RNG",0);
#endif
    auto x=values(features,batch,.3f,.8f),gamma=values(features,1,.7f,.4f),beta=values(features,1,.2f),
         dy=values(features,batch,1.1f);
    gamma+=Matrix<float>::ones(features,1);layer.gamma=Matrix<D>(gamma);layer.beta=Matrix<D>(beta);
    auto objective=[&] {
        double total=0;
        for(int j=0;j<batch;++j) {
            double mean=0,var=0;for(int i=0;i<features;++i)mean+=x.elem(i,j);mean/=features;
            for(int i=0;i<features;++i)var+=std::pow(double(x.elem(i,j))-mean,2);var/=features;
            for(int i=0;i<features;++i)total+=(gamma.elem(i,0)*(x.elem(i,j)-mean)/std::sqrt(var+1e-5)+beta.elem(i,0))*dy.elem(i,j);
        }return total;
    };
    auto dx=finite_difference(x,objective),dg=finite_difference(gamma,objective),db=finite_difference(beta,objective);
    Matrix<D> input(x);layer.eval(input);auto refs=layer.parameters();require(refs.size()==2,"normalization two parameters");
    close(host(layer.backward(input,Matrix<D>(dy))),dx,"normalization dx finite difference",1e-4f);
    close(host(refs[0].grad()),dg,"normalization gamma finite difference");close(host(refs[1].grad()),db,"normalization beta finite difference");
    auto tangent=values(features,batch,2.1f,.6f),jvp=host(layer.jvp(input,Matrix<D>(tangent)));
    double lhs=0,rhs=0;
    for(int j=0;j<batch;++j)for(int i=0;i<features;++i) {
        lhs+=double(dy.elem(i,j))*jvp.elem(i,j);rhs+=double(dx.elem(i,j))*tangent.elem(i,j);
    }
    require(std::abs(lhs-rhs)<3e-5,"normalization JVP/backward adjoint");
    auto x_rows=Matrix<float>::zeros(batch,features);
    for(int j=0;j<batch;++j)for(int i=0;i<features;++i)x_rows.elem(j,i)=x.elem(i,j);
    auto input_view=Matrix<D>(x_rows).T();
    require(input_view.get_transpose(),"normalization transposed input fixture");
    layer.eval(input);const auto expected_value=host(layer.value());
    layer.eval(input_view);
    close(host(layer.value()),expected_value,"normalization transposed forward");
    close(host(layer.backward(input_view,Matrix<D>(dy))),dx,"normalization transposed backward",1e-4f);
    close(host(layer.jvp(input_view,Matrix<D>(tangent))),jvp,"normalization transposed JVP");
    AdamW<D> optimizer(refs);optimizer.zero_grad();std::list<Layer<D>*> net{&layer};freeze(net);
    layer.eval(input);close(host(layer.backward(input,Matrix<D>(dy))),dx,"frozen normalization dx",1e-4f);
    require(optimizer.step()==0,"frozen normalization no optimizer step");
    close(host(layer.gamma),gamma,"frozen normalization gamma",0);unfreeze(net);
    layer.trainable=false;optimizer.zero_grad();layer.eval(input);
    close(host(layer.backward(input,Matrix<D>(dy))),dx,"normalization own freeze flag dx",1e-4f);
    require(optimizer.step()==0,"normalization own freeze flag suppresses updates");
    for(auto& p:refs)require(!p.has_grad(),"normalization own freeze flag no gradients");
    layer.trainable=true;
    Sequential<D> model(net);optimizer.zero_grad();model.eval(input);model.backward(Matrix<D>(dy));
    require(optimizer.step()==2,"Sequential normalization gradients");

    // The standalone and network entry points operate on the same class and
    // expose the same real parameters and Adam state.
    LayerNormLayer<D> standalone(features,batch),network_layer(features,batch);
    for(auto* n:{&standalone,&network_layer}) {n->gamma=Matrix<D>(gamma);n->beta=Matrix<D>(beta);}
    Sequential<D> normalized_network{&network_layer};AdamWOptions same_options;same_options.weight_decay=0;
    AdamW<D> standalone_optimizer(standalone.parameters(),same_options),network_optimizer(normalized_network.parameters(),same_options);
    close(host(standalone.forward(input)),host(normalized_network.eval(input)),"standalone/Sequential normalization forward",0);
    close(host(standalone.jvp(Matrix<D>(tangent))),host(network_layer.jvp(input,Matrix<D>(tangent))),"standalone/Sequential normalization JVP",0);
    close(host(standalone.backward(Matrix<D>(dy))),host(normalized_network.backward(Matrix<D>(dy))),"standalone/Sequential normalization backward",0);
    for(size_t k=0;k<standalone_optimizer.parameters().size();++k)
        close(host(standalone_optimizer.parameters()[k].grad()),host(network_optimizer.parameters()[k].grad()),"standalone/Sequential normalization parameter gradient",0);
    require(standalone_optimizer.step()==2&&network_optimizer.step()==2,"standalone/Sequential normalization optimizer");
    close(host(standalone.gamma),host(network_layer.gamma),"standalone/Sequential normalization updated gamma",0);
    close(host(standalone.beta),host(network_layer.beta),"standalone/Sequential normalization updated beta",0);

    LayerNormLayer<D> frozen_direct(features,batch);frozen_direct.gamma=Matrix<D>(gamma);frozen_direct.beta=Matrix<D>(beta);
    auto frozen_refs=frozen_direct.parameters();std::list<Layer<D>*> frozen_net{&frozen_direct};
    const auto frozen_gm=host(frozen_direct.adam_g.m),frozen_gv=host(frozen_direct.adam_g.v),
               frozen_bm=host(frozen_direct.adam_b.m),frozen_bv=host(frozen_direct.adam_b.v);
    for(int freeze_mode=0;freeze_mode<3;++freeze_mode) {
        frozen_direct.trainable=freeze_mode!=0;unfreeze(frozen_net);if(freeze_mode==1)freeze(frozen_net);
        frozen_direct.forward(input);
        close(host(frozen_direct.backward(Matrix<D>(dy),freeze_mode!=2)),dx,"direct normalization frozen dx",1e-4f);
        close(host(frozen_direct.gamma),gamma,"direct normalization frozen gamma",0);
        close(host(frozen_direct.beta),beta,"direct normalization frozen beta",0);
        close(host(frozen_direct.adam_g.m),frozen_gm,"direct normalization frozen gamma first moment",0);
        close(host(frozen_direct.adam_g.v),frozen_gv,"direct normalization frozen gamma second moment",0);
        close(host(frozen_direct.adam_b.m),frozen_bm,"direct normalization frozen beta first moment",0);
        close(host(frozen_direct.adam_b.v),frozen_bv,"direct normalization frozen beta second moment",0);
        require(frozen_direct.adam_g.iteration==1&&frozen_direct.adam_b.iteration==1,"direct normalization frozen iteration");
        for(auto& p:frozen_refs)require(!p.has_grad(),"direct normalization frozen has no gradients");
    }
    rejects([&]{layer.jvp(input,Matrix<D>::zeros(features+1,batch));},"normalization JVP shape");
    rejects([&]{layer.backward(input,Matrix<D>::zeros(features,batch+1));},"normalization backward shape");
    LayerNormLayer<D> fresh(features,batch);
    rejects([&]{fresh.backward(input,Matrix<D>(dy));},"normalization backward requires eval");
    rejects([&]{fresh.jvp(input,Matrix<D>(dy));},"normalization JVP requires eval");
    fresh.eval(input);
    rejects([&]{fresh.eval(Matrix<D>::zeros(features+1,batch));},"normalization invalid eval input");
    require(!fresh.has_forward,"normalization invalid eval input discards old context");
    rejects([&]{fresh.backward(input,Matrix<D>(dy));},"normalization failed eval invalidates backward");
    rejects([&]{fresh.jvp(input,Matrix<D>(dy));},"normalization failed eval invalidates JVP");
    fresh.forward(input);
    rejects([&]{fresh.forward(Matrix<D>::zeros(features,batch+1));},"normalization invalid direct forward input");
    require(!fresh.has_forward,"normalization invalid direct forward discards old context");
    rejects([&]{fresh.backward(Matrix<D>(dy));},"normalization failed direct forward invalidates backward");
    rejects([&]{fresh.jvp(Matrix<D>(dy));},"normalization failed direct forward invalidates JVP");

    // Public normalization tensors must be validated before raw GPU kernels
    // consume them; a failed eval also discards its old activation context.
    for(int mutation=0;mutation<7;++mutation) {
        LayerNormLayer<D> malformed(features,batch);malformed.eval(input);
        AdamW<D> malformed_optimizer(malformed.parameters());malformed_optimizer.zero_grad();
        const auto saved_gm=host(malformed.adam_g.m),saved_gv=host(malformed.adam_g.v),
                   saved_bm=host(malformed.adam_b.m),saved_bv=host(malformed.adam_b.v);
        const int saved_gi=malformed.adam_g.iteration,saved_bi=malformed.adam_b.iteration;
        switch(mutation) {
            case 0:malformed.gamma=Matrix<D>::ones(features+1,1);break;
            case 1:malformed.beta=Matrix<D>::zeros(1,features);break;
            case 2:malformed.cached_xhat=Matrix<D>::zeros(features+1,batch);break;
            case 3:malformed.cached_inv=Matrix<D>::zeros(2,batch);break;
            case 4:malformed.ones_d1=Matrix<D>::ones(features,2);break;
            case 5:malformed.ones_1N=Matrix<D>::ones(1,batch+1);break;
            case 6:malformed.dim=features+1;break;
        }
        const std::string tag="normalization malformed tensor "+std::to_string(mutation);
        rejects([&]{malformed.backward(input,Matrix<D>(dy));},tag+" backward");
        rejects([&]{malformed.jvp(input,Matrix<D>(dy));},tag+" JVP");
        rejects([&]{malformed.eval(input);},tag+" eval");
        require(!malformed.has_forward,tag+" failed eval invalidates cache");
        malformed.gamma=Matrix<D>::ones(features,1);malformed.beta=Matrix<D>::zeros(features,1);
        malformed.cached_xhat=Matrix<D>::zeros(features,batch);malformed.cached_inv=Matrix<D>::zeros(1,batch);
        malformed.ones_d1=Matrix<D>::ones(features,1);malformed.ones_1N=Matrix<D>::ones(1,batch);
        malformed.dim=features;
        rejects([&]{malformed.backward(input,Matrix<D>(dy));},tag+" restored shape still needs eval");
        rejects([&]{malformed.jvp(input,Matrix<D>(dy));},tag+" restored JVP still needs eval");
        close(host(malformed.adam_g.m),saved_gm,tag+" gamma first moment",0);
        close(host(malformed.adam_g.v),saved_gv,tag+" gamma second moment",0);
        close(host(malformed.adam_b.m),saved_bm,tag+" beta first moment",0);
        close(host(malformed.adam_b.v),saved_bv,tag+" beta second moment",0);
        require(malformed.adam_g.iteration==saved_gi&&malformed.adam_b.iteration==saved_bi,tag+" preserves optimizer iteration");
        for(auto& p:malformed_optimizer.parameters())require(!p.has_grad(),tag+" no partial gradients");
        require(malformed_optimizer.step()==0,tag+" no partial optimizer update");
        malformed.eval(input);malformed.backward(input,Matrix<D>(dy));
        require(malformed_optimizer.step()==2,tag+" fresh eval restores training");
    }
}

static void evaluation_metric_checks() {
    auto labels=Matrix<float>::zeros(2,3),scores=Matrix<float>::zeros(2,3);
    labels.elem(0,0)=1;labels.elem(1,1)=1;labels.elem(1,2)=1;
    scores.elem(0,0)=2;scores.elem(1,1)=3;scores.elem(0,2)=1;
    Matrix<D> input(scores),target(labels);LinearLayer<D> identity(2,2,3);
    auto weights=Matrix<float>::zeros(2,2);weights.elem(0,0)=1;weights.elem(1,1)=1;
    identity.W()=Matrix<D>(weights);identity.b()=Matrix<D>::zeros(2,1);
    Sequential<D> model{&identity};
    require(std::abs(host(error_rate(model.eval(input),target)).elem(0,0)-1.f/3)<1e-6,"error rate of a network output");
}

static void classifier_check() {
    constexpr int batch=4;Shape shape{batch,1,4,4,2,1,1,0,1,true};auto conv=convolution(shape,false);
    ReluLayer<D> readout(2,32,batch);auto x=Matrix<float>::zeros(16,batch),labels=Matrix<float>::zeros(2,batch);
    for(int j=0;j<batch;++j) {for(int i=0;i<16;++i)x.elem(i,j)=j<2?-.8f:.8f;labels.elem(j<2?0:1,j)=1;}
    auto w=Matrix<float>::zeros(2,1);w.elem(0,0)=.5f;w.elem(1,0)=-.5f;conv->W()=Matrix<D>(w);conv->b()=Matrix<D>::zeros(2,1);
    auto rw=Matrix<float>::zeros(2,32);
    for(int i=0;i<16;++i) {rw.elem(0,16+i)=.01f;rw.elem(1,i)=.01f;}
    readout.W()=Matrix<D>(rw);readout.b()=Matrix<D>::ones(2,1)*.1f;
    Matrix<D> input(x),target(labels);Sequential<D> model{&readout,conv.get()};
    AdamWOptions options;options.lr=.02f;options.weight_decay=.01f;AdamW<D> optimizer(model.parameters(),options);
    require(optimizer.parameters().size()==4,"CNN classifier registers all parameter groups");
    auto loss=[&]{return host(softmax_cross_entropy(model.eval(input),target).value).elem(0,0);};
    const float initial=loss();
    for(int i=0;i<35;++i) {
        optimizer.zero_grad();
        auto dlogits=softmax_cross_entropy(model.eval(input),target).grad;
        model.backward(std::move(dlogits));
        optimizer.clip_grad_norm(1);
        require(optimizer.step()==4,"CNN classifier optimizer");
    }
    const float final=loss();require(std::isfinite(final)&&final<initial*.25f,"CNN classification learns");
    std::cout<<"PASS CNN classification loss "<<initial<<" -> "<<final<<'\n';
}

static std::string file_bytes(const std::filesystem::path& p) {
    std::ifstream in(p,std::ios::binary);require(bool(in),"checkpoint file exists");
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
static void restart(const std::string& phase,const std::filesystem::path& output) {
    require(std::filesystem::is_directory(output),"existing external checkpoint output directory required");
    global_rand_gen.seed(phase=="prepare"?57:901);
    Shape shape{2,1,4,4,2,3,3,1,1,true};auto conv=convolution(shape,false);
    shape.ci=2;shape.co=1;shape.relu=false;auto trans=convolution(shape,true);
    Layer<D> tanh_layer(5,16,2);ReluLayer<D> relu_layer(3,5,2);LayerNormLayer<D> norm(3,2);
    auto labels=Matrix<float>::zeros(3,2);labels.elem(0,0)=1;labels.elem(2,1)=1;Matrix<D> target(labels);
    std::list<Layer<D>*> net{&norm,&relu_layer,&tanh_layer,trans.get(),conv.get()};
    Sequential<D> model(net);AdamWOptions options;options.lr=.003f;options.weight_decay=.03f;
    AdamW<D> optimizer(model.parameters(),options);require(optimizer.parameters().size()==10,"all-layer checkpoint parameter count");
    TrainingProgress progress;std::string error;
    if(phase=="resume")require(load_checkpoint(net,(output/"split.ckpt").string(),progress,&error),error);
    for(;progress.step<6;++progress.step) {
        Matrix<D> input(values(16,2,float(progress.step)*.3f,.8f));optimizer.zero_grad();
        auto dlogits=softmax_cross_entropy(model.eval(input),target).grad;
        model.backward(std::move(dlogits));optimizer.step();
        if(phase=="prepare"&&progress.step==2) {auto saved=progress;saved.step=3;require(save_checkpoint(net,(output/"split.ckpt").string(),saved,&error),error);}
    }
    const auto final=output/(phase=="prepare"?"continuous.ckpt":"resumed.ckpt");
    require(save_checkpoint(net,final.string(),progress,&error),error);
    if(phase=="resume")require(file_bytes(output/"continuous.ckpt")==file_bytes(final),"all-layer checkpoint process restart bitwise equality");
    std::cout<<"PASS all-layer checkpoint "<<phase<<'\n';
}
int compute() {
    try {
#ifdef CUDA
        GPUSampler sampler(42);
#endif
        if(const char* phase=std::getenv("CONV_TEST_PHASE")) {
            require(std::string(phase)=="prepare"||std::string(phase)=="resume","valid restart phase");
            const char* output=std::getenv("CONV_TEST_OUTPUT");require(output&&*output,"external checkpoint output required");
            restart(phase,output);return 0;
        }
        empty_matrix_checks<float>();
#ifdef CUDA
        empty_matrix_checks<CUDAfloat>();
#endif
        dense_checks(false);dense_checks(true);logistic_checks();normalization_checks();evaluation_metric_checks();
        for(const Shape& s:std::vector<Shape>{{2,2,5,7,3,3,2,1,1,false},{2,2,5,6,3,2,3,0,2,true}})
            for(bool trans:{false,true})convolution_checks(s,trans);
        classifier_check();
        std::cout<<"PASS all-layer API: scalar/finite-difference gradients, accumulation, freeze, normalization, CNN training\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
#endif
