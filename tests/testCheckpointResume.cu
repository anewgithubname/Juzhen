#include "../cpp/juzhen.hpp"
#include "../ml/checkpoint.hpp"
#include <cmath>
#include <iostream>
#include <list>
#include <fstream>
#include <iterator>
#include <cstdlib>
using namespace Juzhen;
#if defined(CUDA)
using Backend=CUDAfloat;
#elif defined(ROCM_HIP)
using Backend=ROCMfloat;
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
    if(x.num_row()!=y.num_row() || x.num_col()!=y.num_col()) return INFINITY;
    for(size_t c=0;c<x.num_col();++c) for(size_t r=0;r<x.num_row();++r) {
        const float d=std::fabs(x.elem(r,c)-y.elem(r,c));
        if(!std::isfinite(d)) return INFINITY;
        e=std::max(e,d);
    }
    return e;
}
static void transformer_step(TransformerLayer<Backend>& layer) {
    auto xh=Matrix<float>::randn(8,8),gh=Matrix<float>::randn(8,8);
    Matrix<Backend> x(xh); layer.eval(x); layer.backward(x,Matrix<Backend>(gh));
}
static float layer_error(Layer<Backend>& a,Layer<Backend>& b) {
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

static std::string test_path(const std::string& name) {
    const auto root=std::filesystem::current_path()/"checkpoint_resume_tests";
    std::filesystem::create_directories(root);
    return (root/name).string();
}
static std::string bytes(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}

static bool matrix_io_roundtrip() {
    for(bool transposed:{false,true}) {
        Matrix<float> host("io",{{1,2},{3,4},{5,6}});
        Matrix<Backend> original(host),restored("small",1,1);
        if(transposed) original=original.T();
        std::unique_ptr<FILE,decltype(&fclose)> file(tmpfile(),&fclose);
        if(!file) return false;
        write(file.get(),original);
        if(ferror(file.get()) || fseek(file.get(),0,SEEK_SET)!=0) return false;
        read(file.get(),restored);
        if(ferror(file.get()) || feof(file.get()) || error(original,restored)!=0) return false;
    }
    return true;
}

// Run the two phases in separate processes: no GPU allocations or RNG objects
// survive the stop/restart boundary. Training randomness intentionally uses
// the serialized host generator, then transfers inputs to the selected device.
static int restart_test(bool prepare) {
    global_rand_gen.seed(prepare?123:456);
    TransformerLayer<Backend> model(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> net={&model};
    TrainingProgress progress; std::string why;
    const auto checkpoint=test_path("restart.bin"),expected=test_path("restart_expected.bin");
    if(prepare) {
        for(auto& item:model.checkpoint_optimizers()) {
            item.second->alpha=0.003f; item.second->beta1=0.85f;
            item.second->beta2=0.995f; item.second->eps=2e-7f;
        }
        for(int i=0;i<2;++i) transformer_step(model);
        progress={3,2,16};
        if(!save_checkpoint(net,checkpoint,progress,&why)) return 1;
    } else {
        if(!load_checkpoint(net,checkpoint,progress,&why)) { std::cout<<why<<"\n"; return 1; }
        if(progress.epoch!=3 || progress.step!=2 || progress.data_position!=16) return 1;
    }
    for(int i=0;i<3;++i) { transformer_step(model); ++progress.step; progress.data_position+=8; }
    if(prepare) return save_checkpoint(net,expected,progress,&why)?0:1;
    const auto continued_rng=global_rand_gen;
    TransformerLayer<Backend> reference(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> reference_net={&reference}; TrainingProgress expected_progress;
    if(!load_checkpoint(reference_net,expected,expected_progress,&why)) return 1;
    const auto e=layer_error(model,reference);
    std::cout<<"separate-process resume max_abs="<<e<<"\n";
    return e<=1e-6f && continued_rng==global_rand_gen && progress.step==expected_progress.step
        && progress.epoch==expected_progress.epoch && progress.data_position==expected_progress.data_position ? 0:1;
}

static bool rejects_invalid_files(const std::list<Layer<Backend>*>& net,const std::string& valid,
                                  bool incompatible=false) {
    std::string why; TrainingProgress progress{9,10,11};
    const auto before=test_path("before.bin"),after=test_path("after.bin"),bad=test_path("bad.bin");
    if(!save_checkpoint(net,before,progress,&why)) return false;
    auto unchanged=[&](const std::string& file) {
        if(load_checkpoint(net,file,progress,&why) || why.empty()) return false;
        return save_checkpoint(net,after,progress,&why) && bytes(before)==bytes(after);
    };
    if(incompatible) return unchanged(valid);
    const auto original=bytes(valid);
    auto write_bad=[&](const std::string& data) {
        std::ofstream f(bad,std::ios::binary|std::ios::trunc); f.write(data.data(),data.size());
    };
    write_bad(original.substr(0,original.size()-1));
    if(!unchanged(bad)) return false;
    auto damaged=original; damaged.back()^=1; write_bad(damaged);
    if(!unchanged(bad)) return false;
    // Reach the first matrix's nested storage header and corrupt only its size.
    write_bad(original);
    FILE* fp=fopen(bad.c_str(),"r+b");
    if(!fp) return false;
    std::string ignored; uint64_t n=0;
    bool parsed=fseek(fp,8+sizeof(uint32_t)+4*sizeof(uint64_t),SEEK_SET)==0
        && checkpoint_detail::get_string(fp,ignored) && checkpoint_detail::get_string(fp,ignored)
        && checkpoint_detail::get(fp,n) && checkpoint_detail::get_string(fp,ignored)
        && checkpoint_detail::get(fp,n) && checkpoint_detail::get(fp,n);
    int invalid_size=0x7fffffff;
    // A positioning operation is required when switching from reads to writes.
    parsed=parsed && fseek(fp,0,SEEK_CUR)==0 && checkpoint_detail::put(fp,invalid_size);
    fclose(fp);
    if(!parsed || !unchanged(bad)) return false;
    if(!unchanged(test_path("missing.bin"))) return false;
    std::cout<<"invalid checkpoints preserve model, Adam, progress and host RNG\n";
    return true;
}

int compute() {
#if defined(ROCM_HIP)
    std::cout<<"checkpoint backend=ROCm (AMD GPU)\n";
#elif defined(CUDA)
    std::cout<<"checkpoint backend=CUDA\n";
#else
    std::cout<<"checkpoint backend=CPU\n";
#endif
    global_rand_gen.seed(99);
#if defined(CUDA)
    GPUSampler sampler(99);
#endif
    if(const char* phase=std::getenv("JUZHEN_CHECKPOINT_PHASE"))
        return restart_test(std::string(phase)=="prepare");
    if(!matrix_io_roundtrip()) return 1;
    LinearLayer<Backend> a(7,5,4),b(3,7,4); std::list<Layer<Backend>*> net={&a,&b};
    for(int i=0;i<3;++i) train_step(a,b);
    TrainingProgress saved{2,3,12}; std::string why;
    const std::string path=test_path("linear.bin");
    if(!save_checkpoint(net,path,saved,&why)) { std::cout<<why<<"\n"; return 1; }
    // Replacing an existing checkpoint must also succeed.
    if(!save_checkpoint(net,path,saved,&why)) return 1;
    for(int i=0;i<4;++i) train_step(a,b);

    LinearLayer<Backend> ar(7,5,4),br(3,7,4); std::list<Layer<Backend>*> restored={&ar,&br};
    TrainingProgress loaded;
    if(!load_checkpoint(restored,path,loaded,&why)) { std::cout<<why<<"\n"; return 1; }
    if(loaded.epoch!=2||loaded.step!=3||loaded.data_position!=12) return 1;
    for(int i=0;i<4;++i) train_step(ar,br);
    float e=std::max(layer_error(a,ar),layer_error(b,br));
    std::cout<<"linear checkpoint resume max_abs="<<e<<"\n";
    if(e>1e-6f) return 1;

    TransformerLayer<Backend> tf(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> tfnet={&tf};
    for(int i=0;i<2;++i) transformer_step(tf);
    TrainingProgress tf_saved{4,2,16};
    const std::string tfpath=test_path("transformer.bin");
    if(!save_checkpoint(tfnet,tfpath,tf_saved,&why)) { std::cout<<why<<"\n"; return 1; }
    for(int i=0;i<3;++i) transformer_step(tf);

    TransformerLayer<Backend> tfr(8,8,16,4,2,2,true);
    std::list<Layer<Backend>*> tfrestored={&tfr}; TrainingProgress tf_loaded;
    if(!load_checkpoint(tfrestored,tfpath,tf_loaded,&why)) { std::cout<<why<<"\n"; return 1; }
    if(tf_loaded.epoch!=4||tf_loaded.step!=2||tf_loaded.data_position!=16) return 1;
    for(int i=0;i<3;++i) transformer_step(tfr);
    const float te=layer_error(tf,tfr);
    std::cout<<"transformer checkpoint resume max_abs="<<te<<" parameters="
             <<tf.checkpoint_parameters().size()<<" optimizers="<<tf.checkpoint_optimizers().size()<<"\n";
    if(te>1e-6f || !rejects_invalid_files(tfrestored,tfpath)) return 1;
    if(!rejects_invalid_files(tfrestored,path,true)) return 1;
    // Same layer type, different tensor shapes must be rejected too.
    TransformerLayer<Backend> wrong_shape(8,8,17,4,2,2,true);
    std::list<Layer<Backend>*> wrong_net={&wrong_shape};
    const auto wrong_path=test_path("wrong_shape.bin");
    if(!save_checkpoint(wrong_net,wrong_path,tf_saved,&why)
       || !rejects_invalid_files(tfrestored,wrong_path,true)) return 1;
    // Failed replacement must not remove the destination directory or its data.
    const auto blocked=test_path("blocked.bin"); std::filesystem::create_directories(blocked);
    { std::ofstream marker(blocked+"/keep"); marker<<"keep"; }
    if(save_checkpoint(tfrestored,blocked,tf_saved,&why) || why.empty()
       || bytes(blocked+"/keep")!="keep" || std::filesystem::exists(blocked+".tmp")) return 1;
    return 0;
}
