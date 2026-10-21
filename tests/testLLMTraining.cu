/** Explicit LLM training: scalar checks, accumulation, sharing and resume. */
#include "../ml/llm.hpp"
#include "../ml/checkpoint.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
using namespace Juzhen;

template<class D> Matrix<float> host(const Matrix<D>& m) {
    if constexpr(std::is_same_v<D,float>)return m;else return m.to_host();
}
static void require(bool ok,const std::string& what) {if(!ok)throw std::runtime_error(what);}
static void close(const Matrix<float>& a,const Matrix<float>& b,const std::string& what,float tol=3e-5f) {
    require(a.num_row()==b.num_row()&&a.num_col()==b.num_col(),what+" shape");
    float worst=0;for(size_t j=0;j<a.num_col();++j)for(size_t i=0;i<a.num_row();++i) {
        const float e=std::fabs(a.elem(i,j)-b.elem(i,j));require(std::isfinite(e),what+" nonfinite");worst=std::max(worst,e);
    }
    require(worst<=tol,what+" max_abs="+std::to_string(worst));
}
template<class F> void rejects(F f,const char* what) {
    bool caught=false;try{f();}catch(const std::logic_error&){caught=true;}
    require(caught,what);
}
static Matrix<float> transpose_storage(const Matrix<float>& m) {
    Matrix<float> result("transposed storage",m.num_col(),m.num_row());
    for(size_t j=0;j<m.num_col();++j)for(size_t i=0;i<m.num_row();++i)result.elem(j,i)=m.elem(i,j);
    return result;
}
struct Dump {
    std::ofstream file;bool first=true;
    explicit Dump(const std::filesystem::path& path) {
        if(!path.empty()){file.open(path);file.exceptions(std::ios::badbit|std::ios::failbit);file<<"{\n";}
    }
    template<class D> void matrix(const std::string& name,const Matrix<D>& m) {
        if(!file.is_open())return;auto h=host(m);
        if(!first)file<<",\n";first=false;
        file<<'"'<<name<<"\":["<<h.num_row()<<','<<h.num_col()<<",["<<std::setprecision(9);
        bool initial=true;for(size_t j=0;j<h.num_col();++j)for(size_t i=0;i<h.num_row();++i){if(!initial)file<<',';initial=false;file<<h.elem(i,j);}
        file<<"]]";
    }
    void scalar(const std::string& name,float value) {Matrix<float> m("scalar",1,1);m.elem(0,0)=value;matrix(name,m);}
    void finish(){if(file.is_open()){file<<"\n}\n";file.close();}}
};

template<class D> void operator_checks(Dump& dump) {
    Matrix<D> empty_tensor("empty fill",0,3);empty_tensor.zeros();empty_tensor.ones();
    require(empty_tensor.num_row()==0&&empty_tensor.num_col()==3,"empty tensor fill");
#ifdef CUDA
    if constexpr(std::is_same_v<D,CUDAfloat>)CudaErrorCheck(cudaGetLastError());
#endif

    Embedding<D> e(7,3);Matrix<float> table("table",3,7);
    for(size_t j=0;j<7;++j)for(size_t i=0;i<3;++i)table.elem(i,j)=float(10*j+i)/17;
    e.weight()=Matrix<D>(table);TokenIds<D> ids{2,2,6,0};e.eval(ids);
    Matrix<float> expected("lookup",3,4),up("up",3,4),gradient=Matrix<float>::zeros(3,7);
    for(size_t j=0;j<4;++j)for(size_t i=0;i<3;++i){expected.elem(i,j)=table.elem(i,size_t(ids.host()[j]));up.elem(i,j)=float(1+i+3*j)/13;gradient.elem(i,size_t(ids.host()[j]))+=up.elem(i,j);}
    close(host(e.value()),expected,"embedding lookup",0);
    auto refs=e.parameters();AdamW<D> opt(refs);opt.zero_grad();e.backward(Matrix<D>(up));
    close(host(refs[0].grad()),gradient,"repeated ID gradients");close(host(e.weight()),table,"embedding unchanged",0);
    e.backward(Matrix<D>(up));close(host(refs[0].grad()),gradient*2,"embedding accumulation");
    dump.matrix("embedding.table",e.weight());dump.matrix("embedding.output",e.value());dump.matrix("embedding.upstream",Matrix<D>(up));dump.matrix("embedding.grad_twice",refs[0].grad());
    rejects([&]{e.eval(TokenIds<D>{-1});},"negative embedding ID");
    rejects([&]{e.eval(TokenIds<D>{7});},"out of range embedding ID");
    rejects([&]{e.backward(Matrix<D>(up));},"failed eval invalidates embedding context");
    rejects([&]{Embedding<D> bad(0,3);},"zero embedding vocabulary");
    opt.zero_grad();e.eval(TokenIds<D>{});e.backward(Matrix<D>("empty",3,0));require(opt.step()==0,"empty embedding has no gradient");
    e.eval(ids);rejects([&]{e.backward(Matrix<D>("bad",2,4));},"embedding upstream shape");
    e.set_trainable(false);e.backward(Matrix<D>(up));require(opt.step()==0,"frozen embedding");e.set_trainable(true);
    // A transposed view has the same logical shape, different physical strides.
    Matrix<D> backing(transpose_storage(table));e.weight()=backing.T();e.eval(ids);
    close(host(e.value()),expected,"transposed embedding table",0);
    Matrix<D> up_backing(transpose_storage(up));auto up_view=up_backing.T();e.backward(up_view);
    close(host(refs[0].grad()),gradient,"transposed embedding upstream");
    rejects([&]{e.grad(Matrix<D>(up));},"integer IDs reject input derivative");
    rejects([&]{e.jvp(Matrix<D>(up),Matrix<D>(up));},"integer IDs reject input JVP");
    TiedProjection<D> projection(e);Matrix<D> hidden(up);
    projection.eval(hidden);
    close(host(projection.value()),table.T()*up,"tied projection transposed weight");
    close(host(projection.jvp(hidden,hidden)),table.T()*up,"tied projection JVP");
    close(host(projection.grad(hidden)),Matrix<float>::ones(7,4),"tied projection activation derivative",0);
    rejects([&]{projection.jvp(hidden,Matrix<D>("bad tangent",3,3));},"tied projection tangent shape");
    // AdamW handles independent transpose layouts for weights, gradients and moments.
    Matrix<float> initial_m=table*.1,initial_v=table*.01;
    auto& state=*refs[0].state;
    Matrix<D> mback(transpose_storage(initial_m));state.m=mback.T();state.v=Matrix<D>(initial_v);
    state.iteration=1;auto transposed_gradient=Matrix<D>(transpose_storage(gradient));
    refs[0].gradient->value=transposed_gradient.T();
    Matrix<float> expected_weight=table,expected_m=initial_m,expected_v=initial_v;
    for(size_t j=0;j<7;++j)for(size_t i=0;i<3;++i) {
        float g=gradient.elem(i,j),m=state.beta1*initial_m.elem(i,j)+(1-state.beta1)*g;
        float v=state.beta2*initial_v.elem(i,j)+(1-state.beta2)*g*g;
        expected_m.elem(i,j)=m;expected_v.elem(i,j)=v;
        expected_weight.elem(i,j)=table.elem(i,j)*(1-state.alpha*.01f)-state.alpha*(m/(1-state.beta1))/(std::sqrt(v/(1-state.beta2))+state.eps);
    }
    require(opt.step()==1,"transposed optimizer step");
    close(host(e.weight()),expected_weight,"transposed optimizer weights",2e-6f);
    close(host(state.m),expected_m,"transposed optimizer first moment",2e-6f);
    close(host(state.v),expected_v,"transposed optimizer second moment",2e-6f);
    opt.zero_grad();e.backward(Matrix<D>(up));
    close(host(refs[0].grad()),gradient,"embedding reuses transposed gradient storage");



    Matrix<float> z("logits",7,4);
    for(size_t j=0;j<4;++j)for(size_t i=0;i<7;++i)z.elem(i,j)=float(int((i*3+j*5)%17)-8)*.3f;
    z.elem(0,0)=10000;z.elem(1,0)=9999.75f;z.elem(2,0)=9999.5f;
    TokenIds<D> targets{1,-100,6,0};Matrix<D> logits(z);TokenCrossEntropy<D> loss;
    loss.eval(logits,targets);require(loss.valid_tokens()==3,"valid token count");
    Matrix<float> ref=Matrix<float>::zeros(7,4);double reference_loss=0;
    for(size_t j=0;j<4;++j){if(targets.host()[j]==-100)continue;double maximum=-1e100;
        for(size_t i=0;i<7;++i)maximum=std::max(maximum,double(z.elem(i,j)));
        double total=0;for(size_t i=0;i<7;++i)total+=std::exp(double(z.elem(i,j))-maximum);
        reference_loss+=(maximum-z.elem(size_t(targets.host()[j]),j))+std::log(total);
        for(size_t i=0;i<7;++i)ref.elem(i,j)=float((std::exp(double(z.elem(i,j))-maximum)/total-(int32_t(i)==targets.host()[j]))/3);
    }
    require(std::abs(host(loss.value()).elem(0,0)-reference_loss/3)<2e-5,"stable cross entropy");
    close(host(loss.grad()),ref,"cross entropy gradient",2e-5f);
    close(host(loss.grad(.3f)),ref*.3f,"cross entropy upstream scale",2e-5f);
    dump.matrix("ce.logits",logits);dump.matrix("ce.loss",loss.value());dump.matrix("ce.grad",loss.grad());
    Matrix<D> zback(transpose_storage(z));auto zview=zback.T();loss.eval(zview,targets);
    close(host(loss.grad()),ref,"transposed logits",2e-5f);
    rejects([&]{loss.eval(logits,TokenIds<D>{0});},"target count");
    rejects([&]{loss.grad();},"failed loss invalidates context");
    rejects([&]{loss.eval(logits,TokenIds<D>{7,0,0,0});},"invalid target");
    rejects([&]{loss.eval(logits,TokenIds<D>{-2,0,0,0});},"invalid negative target");
    loss.eval(logits,TokenIds<D>{-100,-100,-100,-100});
    require(loss.valid_tokens()==0&&host(loss.value()).elem(0,0)==0,"all ignored loss");
    close(host(loss.grad()),Matrix<float>::zeros(7,4),"all ignored gradient",0);
    Matrix<D> empty("empty",7,0);loss.eval(empty,TokenIds<D>{});require(loss.valid_tokens()==0,"empty loss");
    require(loss.grad().num_col()==0,"empty loss gradient");
    // Non-power-of-two and large vocabularies exercise CUDA reduction loops.
    for(int vocab:{1,257,50003}) {
        Matrix<float> large("wide",vocab,2);
        for(int j=0;j<2;++j)for(int i=0;i<vocab;++i)large.elem(i,j)=float(i%19)*.1f;
        Matrix<D> device(large);loss.eval(device,TokenIds<D>{0,vocab-1});auto g=host(loss.grad());
        for(int j=0;j<2;++j){double total=0;for(int i=0;i<vocab;++i)total+=g.elem(i,j);require(std::abs(total)<2e-5,"gradient sums to zero");}
    }
}

template<class D> struct TinyLM {
    int batch;
    Embedding<D> token,position;
    TransformerLayer<D> block;
    TiedProjection<D> head;
    Sequential<D> net;
    TokenIds<D> positions;
    Matrix<D> input;
    static std::vector<int32_t> position_ids(int batch){std::vector<int32_t> ids;for(int b=0;b<batch;++b)for(int t=0;t<4;++t)ids.push_back(t);return ids;}
    explicit TinyLM(int batch=2):batch(batch),token(17,8),position(4,8),block(8,8,16,4,batch,2,true),head(token),
        net{&head,&block},positions(position_ids(batch)){}
    std::vector<ParameterRef<D>> refs() {
        auto p=net.parameters();
        for(auto a:token.parameters())p.push_back(a); // Intentional shared registration.
        for(auto a:position.parameters()){a.name="position";p.push_back(a);}return p;
    }
    void eval(const TokenIds<D>& ids){token.eval(ids);position.eval(positions);input=token.value()+position.value();net.eval(input);}
    Matrix<D> backward(Matrix<D>&& dz){auto dx=net.backward(std::move(dz));token.backward(dx);position.backward(dx);return dx;}
    std::list<Layer<D>*> checkpoint_net(){return {&head,&block,&position,&token};}
};
template<class D> void initialize(const std::vector<ParameterRef<D>>& refs) {
    for(size_t k=0;k<refs.size();++k){auto p=refs[k];Matrix<float> v("parameter",p.value->num_row(),p.value->num_col());
        for(size_t j=0;j<v.num_col();++j)for(size_t i=0;i<v.num_row();++i)
            v.elem(i,j)=(p.name.find("gamma")!=std::string::npos?1.f:0.f)+float(int((i*17+j*13+k*3)%23)-11)*.025f;
        *p.value=Matrix<D>(v);
    }
}
static AdamWOptions options(){return {.002f,.9f,.97f,1e-8f,.03f};}

template<class D> void integration(Dump& dump,const std::filesystem::path& directory,const std::string& backend) {
    TinyLM<D> model;AdamW<D> opt(model.refs(),options());require(opt.parameters().size()==15,"shared parameter deduplication");initialize(opt.parameters());
    TokenIds<D> ids{1,1,3,7,4,5,1,2},targets{2,-100,4,1,8,7,-100,6};
    TokenCrossEntropy<D> loss;
    Matrix<float> hyper("options",5,1);auto o=options();hyper.elem(0,0)=o.lr;hyper.elem(1,0)=o.beta1;hyper.elem(2,0)=o.beta2;hyper.elem(3,0)=o.eps;hyper.elem(4,0)=o.weight_decay;dump.matrix("options",hyper);
    for(const auto& p:opt.parameters()){dump.matrix("initial."+p.name,*p.value);dump.scalar("decay."+p.name,p.decay?1.f:0.f);}
    for(int step=0;step<3;++step) {
        opt.zero_grad();model.eval(ids);loss.eval(model.head.value(),targets);const auto tag="step"+std::to_string(step)+".";
        dump.matrix(tag+"logits",model.head.value());dump.matrix(tag+"loss",loss.value());dump.matrix(tag+"dlogits",loss.grad());
        std::vector<Matrix<float>> before;for(const auto& p:opt.parameters())before.push_back(host(*p.value));
        auto dx=model.backward(loss.grad());dump.matrix(tag+"input_dx",dx);
        size_t index=0;
        for(const auto& p:opt.parameters()) {
            require(p.has_grad(),"all LLM parameters receive gradients");
            close(host(*p.value),before[index++],"backward must not update parameters",0);
            require(p.state->iteration==step+1,"backward must not advance Adam");
            dump.matrix(tag+"raw_grad."+p.name,p.grad());
        }
        dump.scalar(tag+"grad_norm",float(opt.clip_grad_norm(.35)));
        std::vector<Matrix<float>> gradients;for(const auto& p:opt.parameters()){gradients.push_back(host(p.grad()));dump.matrix(tag+"grad."+p.name,p.grad());}
        require(opt.step()==15,"exactly one update per shared parameter");index=0;
        for(const auto& p:opt.parameters()) {
            close(host(p.grad()),gradients[index++],"step preserves gradients",0);
            dump.matrix(tag+"parameter."+p.name,*p.value);dump.matrix(tag+"m."+p.name,p.state->m);dump.matrix(tag+"v."+p.name,p.state->v);
            require(p.state->iteration==step+2,"Adam iteration once per step");
        }
    }
    if(!directory.empty()) {
        const auto path=(directory/("checkpoint-"+backend+".bin")).string();TrainingProgress progress{2,3,24};std::string error;
        require(save_checkpoint(model.checkpoint_net(),path,progress,&error),"save checkpoint: "+error);
        opt.zero_grad();model.eval(ids);loss.eval(model.head.value(),targets);model.backward(loss.grad());opt.clip_grad_norm(.35);opt.step();
        TinyLM<D> restored;AdamW<D> resumed(restored.refs(),options());TrainingProgress read;
        require(load_checkpoint(restored.checkpoint_net(),path,read,&error),"load checkpoint: "+error);
        require(read.step==3&&read.epoch==2&&read.data_position==24,"training progress restored");
        resumed.zero_grad();restored.eval(ids);loss.eval(restored.head.value(),targets);restored.backward(loss.grad());resumed.clip_grad_norm(.35);resumed.step();
        for(size_t k=0;k<opt.parameters().size();++k) {
            const auto& a=opt.parameters()[k];const auto& b=resumed.parameters()[k];
            close(host(*a.value),host(*b.value),"resumed parameter");close(host(a.state->m),host(b.state->m),"resumed moment");close(host(a.state->v),host(b.state->v),"resumed variance");
        }
    }
}

template<class D> void accumulation() {
    TinyLM<D> full(2),micro(1);AdamW<D> large(full.refs(),options()),small(micro.refs(),options());initialize(large.parameters());
    for(size_t k=0;k<large.parameters().size();++k)*small.parameters()[k].value=Matrix<D>(*large.parameters()[k].value);
    TokenIds<D> ids{1,1,3,7,4,5,1,2},targets{2,-100,4,1,8,7,-100,-100};TokenCrossEntropy<D> loss;
    large.zero_grad();small.zero_grad();full.eval(ids);loss.eval(full.head.value(),targets);full.backward(loss.grad());
    for(int b=0;b<2;++b) {
        TokenIds<D> x(std::vector<int32_t>(ids.host().begin()+4*b,ids.host().begin()+4*b+4));
        TokenIds<D> y(std::vector<int32_t>(targets.host().begin()+4*b,targets.host().begin()+4*b+4));
        micro.eval(x);loss.eval(micro.head.value(),y);micro.backward(loss.grad(float(loss.valid_tokens())/5.f));
    }
    for(size_t k=0;k<large.parameters().size();++k) {
        close(host(large.parameters()[k].grad()),host(small.parameters()[k].grad()),"unequal-token microbatch accumulation",6e-5f);
        require(small.parameters()[k].state->iteration==1,"microbatches do not advance Adam");
    }
    large.step();small.step();
    for(size_t k=0;k<large.parameters().size();++k)close(host(*large.parameters()[k].value),host(*small.parameters()[k].value),"accumulated update",8e-5f);
}

template<class D> void suite(const std::string& backend,const std::filesystem::path& directory) {
    Dump dump(directory.empty()?std::filesystem::path{}:directory/(backend+".json"));
    operator_checks<D>(dump);accumulation<D>();integration<D>(dump,directory,backend);dump.finish();
    std::cout<<"[PASS] "<<backend<<" explicit LLM training\n";
}
int compute() {
    try {
        global_rand_gen.seed(919);std::filesystem::path directory;
        if(const char* p=std::getenv("JUZHEN_LLM_DUMP")){directory=p;require(std::filesystem::is_directory(directory),"dump directory must already exist");}
        suite<float>("CPU",directory);
#ifdef CUDA
        GPUSampler sampler(919);suite<CUDAfloat>("CUDA",directory);
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"[FAIL] "<<e.what()<<'\n';return 1;}
}
