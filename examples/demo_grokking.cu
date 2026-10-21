// Modular addition with native Sequential and either a Transformer or an MLP.
// See docs/grokking.md. Data and checkpoints must stay on the mounted drive.
#include "../ml/llm.hpp"
#include "../ml/checkpoint.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
using namespace Juzhen;
namespace fs=std::filesystem;
#ifdef CUDA
using D=CUDAfloat;
#else
using D=float;
#endif
namespace {
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string env(const char* key,std::string fallback=""){
    const char* v=std::getenv(key);return v?std::string(v):fallback;
}
double number(const char* key,double fallback){
    auto s=env(key,std::to_string(fallback));size_t n=0;double v=std::stod(s,&n);
    require(n==s.size()&&std::isfinite(v),std::string("invalid ")+key);return v;
}
int integer(const char* key,int fallback){
    double v=number(key,fallback);require(v>=0&&v<=1000000&&std::floor(v)==v,std::string("invalid ")+key);return int(v);
}
struct Config {
    std::string model=env("GROK_MODEL","transformer");
    int p=integer("GROK_P",113),dim=integer("GROK_DIM",128),seed=integer("GROK_SEED",42);
    int hidden=integer("GROK_HIDDEN",512);
    int steps=integer("GROK_STEPS",20000),every=integer("GROK_EVAL_EVERY",100);
    double fraction=number("GROK_FRACTION",.3),lr=number("GROK_LR",.001),decay=number("GROK_DECAY",1);
    int batch=0;fs::path output=env("GROK_OUTPUT");
    Config(){require(p>=3&&p<=251&&dim>=4&&dim<=512&&dim%4==0&&steps>0&&every>0&&
        hidden>0&&hidden<=2048&&(model=="transformer"||model=="mlp")&&
        fraction>0&&fraction<1&&lr>0&&decay>=0&&!output.empty(),"invalid grokking configuration");
        batch=int(p*p*fraction);require(batch>0&&batch<p*p,"empty training or heldout split");}
};
void check_storage(const fs::path& output){
    const fs::path mount="/mnt/experiments",root=mount/"Backup/Juzhen-data";
    std::ifstream mounts("/proc/self/mountinfo");std::string line;bool mounted=false;
    while(std::getline(mounts,line))if(line.find(" /mnt/experiments ")!=std::string::npos)mounted=true;
    require(mounted,"external drive is not mounted");
    auto relative=fs::weakly_canonical(output).lexically_relative(fs::canonical(root));
    require(!relative.empty()&&!relative.is_absolute()&&*relative.begin()!="..","GROK_OUTPUT must be under external storage root");
}
template<class B> Matrix<float> host(const Matrix<B>& m){
    if constexpr(std::is_same_v<B,float>)return m;else return m.to_host();
}
struct Batch {
    TokenIds<D> x,y;std::vector<int32_t> labels;int count,stride;Matrix<D> features;
    Batch(const std::vector<int>& pairs,size_t start,const Config& c):
        x(inputs(pairs,start,c)),y(targets(pairs,start,c)),labels(targets(pairs,start,c)),
        count(int(std::min(size_t(c.batch),pairs.size()-start))),stride(c.model=="mlp"?1:3){
        if(c.model=="mlp"){
            auto h=Matrix<float>::zeros(2*c.p,c.batch);
            for(int j=0;j<count;++j){h.elem(pairs[start+j]/c.p,j)=1;h.elem(c.p+pairs[start+j]%c.p,j)=1;}
            features=Matrix<D>(h);
        }
    }
    static std::vector<int32_t> inputs(const std::vector<int>& ids,size_t start,const Config& c){
        std::vector<int32_t> x(3*c.batch,0);
        for(int j=0;j<c.batch&&start+j<ids.size();++j){x[3*j]=ids[start+j]/c.p;x[3*j+1]=ids[start+j]%c.p;x[3*j+2]=c.p;}
        return x;
    }
    static std::vector<int32_t> targets(const std::vector<int>& ids,size_t start,const Config& c){
        const int stride=c.model=="mlp"?1:3;std::vector<int32_t> y(stride*c.batch,-100);
        for(int j=0;j<c.batch&&start+j<ids.size();++j)y[stride*(j+1)-1]=(ids[start+j]/c.p+ids[start+j]%c.p)%c.p;
        return y;
    }
};
std::vector<int32_t> positions(int batch){std::vector<int32_t> ids;for(int b=0;b<batch;++b)for(int t=0;t<3;++t)ids.push_back(t);return ids;}
struct TransformerModel {
    Embedding<D> token,position;TransformerLayer<D> block;LinearLayer<D> head;
    Sequential<D> network;TokenIds<D> position_ids;Matrix<D> input;
    explicit TransformerModel(const Config& c):token(c.p+1,c.dim),position(3,c.dim),
        block(c.dim,c.dim,4*c.dim,3,c.batch,4,true),head(c.p,c.dim,3*c.batch),
        network{&head,&block},position_ids(positions(c.batch)){}
    auto parameters(){
        std::vector<ParameterRef<D>> refs;
        auto append=[&](auto entries,const std::string& prefix){for(auto p:entries){p.name=prefix+p.name;refs.push_back(p);}};
        append(token.parameters(),"token.");append(position.parameters(),"position.");
        append(block.parameters(),"block.");append(head.parameters(),"head.");return refs;
    }
    auto layers(){return std::list<Layer<D>*>{&head,&block,&position,&token};}
    const Matrix<D>& eval(const Batch& b){
        token.eval(b.x);position.eval(position_ids);input=token.value()+position.value();return network.eval(input);
    }
    void backward(Matrix<D>&& dy){auto dx=network.backward(std::move(dy));token.backward(dx);position.backward(dx);}
};
// Experiment-local activation; existing ReluLayer combines an affine map with
// ReLU and currently exposes only the legacy updating backward interface.
class ReLU : public Layer<D> {
public:
    ReLU():Layer<D>(1,1,0){}
    void eval(const Matrix<D>& x) override{this->val=relu(Matrix<D>(x));}
    bool supports_backward() const override{return true;}
    Matrix<D> grad(const Matrix<D>& x) const override{return d_relu(Matrix<D>(x));}
    Matrix<D> backward(const Matrix<D>& x,Matrix<D>&& dy) override{return hadmd(grad(x),std::move(dy));}
    Matrix<D> backward_update(const Matrix<D>& x,Matrix<D>&& dy) override{return backward(x,std::move(dy));}
    Matrix<D> jvp(const Matrix<D>& x,const Matrix<D>& dx) override{return hadmd(grad(x),dx);}
    std::vector<ParameterRef<D>> parameters() override{return {};}
    std::vector<std::pair<std::string,Matrix<D>*>> checkpoint_parameters() override{return {};}
    std::vector<std::pair<std::string,adam_state<D>*>> checkpoint_optimizers() override{return {};}
};
struct MLPModel {
    LinearLayer<D> hidden,head;ReLU activation;Sequential<D> network;
    explicit MLPModel(const Config& c):hidden(c.hidden,2*c.p,c.batch),head(c.p,c.hidden,c.batch),
        network{&head,&activation,&hidden}{}
    auto parameters(){
        auto refs=hidden.parameters();for(auto& p:refs)p.name="hidden."+p.name;
        for(auto p:head.parameters()){p.name="head."+p.name;refs.push_back(p);}return refs;
    }
    auto layers(){return network.layers();}
    const Matrix<D>& eval(const Batch& b){return network.eval(b.features);}
    void backward(Matrix<D>&& dy){network.backward(std::move(dy));}
};
// Canonical column-major archive, also readable by scripts/text8_lm.py.
void archive(const fs::path& path,const std::vector<ParameterRef<D>>& refs,bool gradients=false){
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(path.c_str(),"wb"),&fclose);require(bool(f),"cannot open archive");
    require(fwrite("T8PAR001",1,8,f.get())==8,"archive header");uint32_t count=refs.size();
    require(checkpoint_detail::put(f.get(),count),"archive count");
    for(const auto& p:refs){auto m=host(gradients?p.grad():*p.value);
        require(checkpoint_detail::put_string(f.get(),p.name),"archive name");uint64_t rows=m.num_row(),cols=m.num_col();
        require(checkpoint_detail::put(f.get(),rows)&&checkpoint_detail::put(f.get(),cols),"archive shape");
        for(size_t j=0;j<cols;++j)for(size_t i=0;i<rows;++i){float v=m.elem(i,j);require(checkpoint_detail::put(f.get(),v),"archive value");}
    }require(fflush(f.get())==0,"archive flush");
}
struct Metrics{double loss=0,accuracy=0;};
template<class Model> Metrics evaluate(Model& model,const std::vector<Batch>& batches){
    TokenCrossEntropy<D> loss;double total=0;size_t correct=0,count=0;
    for(const auto& b:batches){const auto& logits=model.eval(b);loss.eval(logits,b.y);
        const double value=host(loss.value()).elem(0,0);require(std::isfinite(value),"nonfinite loss");total+=value*b.count;
        auto h=host(logits);
        for(int j=0;j<b.count;++j){const int col=b.stride*(j+1)-1;int best=0;
            for(size_t r=1;r<h.num_row();++r)if(h.elem(r,col)>h.elem(best,col))best=int(r);
            correct+=best==b.labels[col];}count+=b.count;
    }return {total/count,double(correct)/count};
}
template<class Model> void run(const Config& c){
    check_storage(c.output);fs::create_directories(c.output);
    require(!fs::exists(c.output/"metrics.csv"),"output already contains a run; choose a new directory");
    std::vector<int> pairs(c.p*c.p);std::iota(pairs.begin(),pairs.end(),0);std::mt19937 split_rng(c.seed);std::shuffle(pairs.begin(),pairs.end(),split_rng);
    std::vector<int> train(pairs.begin(),pairs.begin()+c.batch),heldout(pairs.begin()+c.batch,pairs.end());
    {std::ofstream data(c.output/"pairs.csv");data<<"split,a,b,target\n";
        for(size_t i=0;i<pairs.size();++i){int a=pairs[i]/c.p,b=pairs[i]%c.p;data<<(i<size_t(c.batch)?"train":"heldout")<<','<<a<<','<<b<<','<<(a+b)%c.p<<'\n';}require(bool(data),"cannot save split");}
    std::vector<Batch> training,validation;training.emplace_back(train,0,c);
    for(size_t i=0;i<heldout.size();i+=c.batch)validation.emplace_back(heldout,i,c);
    Model model(c);auto refs=model.parameters();std::mt19937 rng(c.seed);std::normal_distribution<float> normal(0,1);
    for(auto& p:refs){Matrix<float> w("initial",p.value->num_row(),p.value->num_col());
        const bool gamma=p.name.find("gamma")!=std::string::npos,weight=p.decay;
        const float scale=p.name.starts_with("token.")||p.name.starts_with("position.")?1.f:
            std::sqrt((p.name=="hidden.weight"?2.f:1.f)/float(w.num_col()));
        for(size_t j=0;j<w.num_col();++j)for(size_t i=0;i<w.num_row();++i)w.elem(i,j)=gamma?1.f:(weight?normal(rng)*scale:0.f);
        *p.value=Matrix<D>(w);p.decay=true; // Explicit experiment: AdamW decay on every parameter, including LN and bias.
    }
    archive(c.output/"initial.params",refs);
    AdamWOptions options;options.lr=float(c.lr);options.beta2=.98f;options.weight_decay=float(c.decay);
    AdamW<D> optimizer(refs,options);TokenCrossEntropy<D> loss;TrainingProgress progress;std::string error;
    auto save=[&](const std::string& name){check_storage(c.output);require(save_checkpoint(model.layers(),(c.output/name).string(),progress,&error),error);};
    save("initial.ckpt");
    {std::ofstream config(c.output/"config.json");config<<std::setprecision(12)<<"{\"p\":"<<c.p<<",\"dim\":"<<c.dim<<",\"seed\":"<<c.seed<<",\"train_examples\":"<<c.batch
      <<",\"heldout_examples\":"<<heldout.size()<<",\"fraction\":"<<c.fraction<<",\"steps\":"<<c.steps<<",\"eval_every\":"<<c.every
      <<",\"lr\":"<<c.lr<<",\"weight_decay\":"<<c.decay<<",\"beta1\":0.9,\"beta2\":0.98,\"epsilon\":1e-8,\"decay_all\":true,\"model\":\""<<c.model
      <<"\",\"hidden\":"<<c.hidden<<",\"layernorm\":"<<(c.model=="mlp"?"false":"true")<<",\"tied\":false,\"tf32\":\""<<env("NVIDIA_TF32","0")<<"\"}\n";require(bool(config),"cannot save config");}
    std::ofstream metrics(c.output/"metrics.csv");metrics<<"step,train_loss,train_accuracy,heldout_loss,heldout_accuracy,weight_norm,seconds\n";
    auto start=std::chrono::steady_clock::now();bool memorized=false;
    auto report=[&]{
        auto a=evaluate(model,training),b=evaluate(model,validation);double norm=0;
        for(const auto& p:refs){auto w=host(*p.value);for(size_t j=0;j<w.num_col();++j)for(size_t i=0;i<w.num_row();++i){double v=w.elem(i,j);norm+=v*v;}}
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        metrics<<std::setprecision(12)<<progress.step<<','<<a.loss<<','<<a.accuracy<<','<<b.loss<<','<<b.accuracy<<','<<std::sqrt(norm)<<','<<seconds<<std::endl;
        std::cout<<"step="<<progress.step<<" train="<<a.loss<<" acc="<<a.accuracy<<" heldout="<<b.loss<<" acc="<<b.accuracy<<" seconds="<<seconds<<std::endl;
        if(!memorized&&a.accuracy>=.99){save("memorized.ckpt");memorized=true;}
    };
    report();
    for(int step=1;step<=c.steps;++step){
        optimizer.zero_grad();loss.eval(model.eval(training[0]),training[0].y);model.backward(loss.grad());
        if(step==1)archive(c.output/"first.grad",refs,true);
        optimizer.step();progress.step=step;progress.data_position+=c.batch;
        if(step==1)archive(c.output/"first.params",refs);
        if(step%c.every==0||step==c.steps)report();
        if(step%1000==0)save("latest.ckpt");
    }
    save("final.ckpt");archive(c.output/"final.params",refs);require(bool(metrics),"metrics write failed");
}
}
int compute(){try{global_rand_gen.seed(42);
#ifdef CUDA
    GPUSampler sampler(42);
#endif
    Config c;if(c.model=="mlp")run<MLPModel>(c);else run<TransformerModel>(c);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
