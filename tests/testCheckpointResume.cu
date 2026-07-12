#include "../cpp/juzhen.hpp"
#include "../ml/checkpoint.hpp"
#include <cmath>
#include <iostream>
#include <list>
using namespace Juzhen;
#if defined(CUDA)
using Backend=CUDAfloat;
#else
using Backend=float;
#endif
template<class D> Matrix<float> h(const Matrix<D>& m) { if constexpr(std::is_same_v<D,float>) return m; else return m.to_host(); }

static void train_step(LinearLayer<Backend>& a,LinearLayer<Backend>& b) {
    auto xh=Matrix<float>::randn(5,4), gh=Matrix<float>::randn(3,4);
    Matrix<Backend> x(xh); a.eval(x); b.eval(a.value());
    auto da=b.backward(a.value(),Matrix<Backend>(gh)); a.backward(x,std::move(da));
}
static float error(const Matrix<Backend>& a,const Matrix<Backend>& b) {
    auto x=h(a),y=h(b); float e=0;
    for(size_t c=0;c<x.num_col();++c) for(size_t r=0;r<x.num_row();++r) e=std::max(e,std::fabs(x.elem(r,c)-y.elem(r,c)));
    return e;
}
static void transformer_step(TransformerLayer<Backend>& layer) {
    auto xh=Matrix<float>::randn(8,8),gh=Matrix<float>::randn(8,8);
    Matrix<Backend> x(xh); layer.eval(x); layer.backward(x,Matrix<Backend>(gh));
}
static float transformer_error(TransformerLayer<Backend>& a,TransformerLayer<Backend>& b) {
    float e=0; auto ap=a.checkpoint_parameters(),bp=b.checkpoint_parameters();
    for(size_t i=0;i<ap.size();++i) e=std::max(e,error(*ap[i].second,*bp[i].second));
    auto ao=a.checkpoint_optimizers(),bo=b.checkpoint_optimizers();
    for(size_t i=0;i<ao.size();++i) {
        e=std::max(e,error(ao[i].second->m,bo[i].second->m));
        e=std::max(e,error(ao[i].second->v,bo[i].second->v));
        auto& x=*ao[i].second; auto& y=*bo[i].second;
        if(x.iteration!=y.iteration||x.alpha!=y.alpha||x.beta1!=y.beta1||
           x.beta2!=y.beta2||x.eps!=y.eps) return INFINITY;
    }
    return e;
}
int compute() {
    global_rand_gen.seed(99);
#if defined(CUDA)
    GPUSampler sampler(99);
#endif
    LinearLayer<Backend> a(7,5,4),b(3,7,4); std::list<Layer<Backend>*> net={&a,&b};
    for(int i=0;i<3;++i) train_step(a,b);
    TrainingProgress saved{2,3,12}; std::string why;
    const std::string path=std::string(PROJECT_DIR)+"/res/checkpoint_resume_test.bin";
    if(!save_checkpoint(net,path,saved,&why)) { std::cout<<why<<"\n"; return 1; }
    for(int i=0;i<4;++i) train_step(a,b);

    LinearLayer<Backend> ar(7,5,4),br(3,7,4); std::list<Layer<Backend>*> restored={&ar,&br};
    TrainingProgress loaded;
    if(!load_checkpoint(restored,path,loaded,&why)) { std::cout<<why<<"\n"; return 1; }
    if(loaded.epoch!=2||loaded.step!=3||loaded.data_position!=12) return 1;
    for(int i=0;i<4;++i) train_step(ar,br);
    float e=std::max({error(a.W(),ar.W()),error(a.b(),ar.b()),error(b.W(),br.W()),error(b.b(),br.b()),
                      error(a.adamWstate().m,ar.adamWstate().m),error(b.adamWstate().v,br.adamWstate().v)});
    std::cout<<"linear checkpoint resume max_abs="<<e<<"\n";
    if(e>1e-6f) return 1;

    TransformerLayer<Backend> tf(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> tfnet={&tf};
    for(int i=0;i<2;++i) transformer_step(tf);
    TrainingProgress tf_saved{4,2,16};
    const std::string tfpath=std::string(PROJECT_DIR)+"/res/checkpoint_transformer_test.bin";
    if(!save_checkpoint(tfnet,tfpath,tf_saved,&why)) { std::cout<<why<<"\n"; return 1; }
    for(int i=0;i<3;++i) transformer_step(tf);

    TransformerLayer<Backend> tfr(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> tfrestored={&tfr}; TrainingProgress tf_loaded;
    if(!load_checkpoint(tfrestored,tfpath,tf_loaded,&why)) { std::cout<<why<<"\n"; return 1; }
    if(tf_loaded.epoch!=4||tf_loaded.step!=2||tf_loaded.data_position!=16) return 1;
    for(int i=0;i<3;++i) transformer_step(tfr);
    const float te=transformer_error(tf,tfr);
    std::cout<<"transformer checkpoint resume max_abs="<<te<<" parameters="
             <<tf.checkpoint_parameters().size()<<" optimizers="<<tf.checkpoint_optimizers().size()<<"\n";
    return te<=1e-6f ? 0:1;
}
