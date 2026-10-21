/** Reproducible character LM. See scripts/text8_lm.py and docs/text8-lm.md.
 * All data/output paths are explicit. The Python runner records the protocol.
 */
#include "../ml/llm.hpp"
#include "../ml/checkpoint.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
using namespace Juzhen;
namespace fs=std::filesystem;
#ifdef CUDA
using Backend=CUDAfloat;
#else
using Backend=float;
#endif
namespace text8 {
static void ensure(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static std::string env(const char* name,const std::string& fallback=""){
    const char* p=std::getenv(name);return p?std::string(p):fallback;
}
static int number(const char* name,int fallback,int lo,int hi){
    auto raw=env(name,std::to_string(fallback));size_t end=0;long long x=std::stoll(raw,&end);
    ensure(end==raw.size()&&x>=lo&&x<=hi,std::string("invalid ")+name);return int(x);
}
struct Config {
    bool legacy=env("T8_LEGACY_TINYSTORIES","0")=="1";
    int vocab=number("T8_VOCAB",27,2,1048576),token_bytes=number("T8_TOKEN_BYTES",1,1,4);
    int dim=number("T8_DIM",256,8,2048),layers=number("T8_LAYERS",4,1,24);
    int seq=number("T8_SEQ",128,2,2048),batch=number("T8_BATCH",32,1,256),heads=number("T8_HEADS",4,1,64);
    int steps=number("T8_STEPS",1000,2,10000000),warmup=number("T8_WARMUP",100,1,1000000);
    int eval_every=number("T8_EVAL_EVERY",100,1,1000000),eval_batches=number("T8_EVAL_BATCHES",32,1,100000);
    int test_batches=number("T8_TEST_BATCHES",128,1,100000),seed=number("T8_SEED",42,0,1000000);
    int stop=number("T8_STOP_AFTER",steps,0,steps),generate=number("T8_GENERATE",128,0,4096);
    fs::path data=env("T8_DATA"),output=env("T8_OUTPUT");
    Config(){ensure(token_bytes==1||token_bytes==4,"token bytes must be 1 or 4");ensure(token_bytes==4||vocab<=256,"byte tokens require vocab <= 256");ensure(dim%heads==0&&warmup<steps,"invalid head count/warmup");ensure(int64_t(seq)*batch<=1048576,"token batch too large");
        ensure(!data.empty()&&!output.empty(),"T8_DATA and T8_OUTPUT are required");}
    std::string spec() const {
        std::ostringstream s;s<<"text8-lm-v1\n"<<fs::absolute(data).string()<<'\n'<<env("T8_DATA_ID")<<'\n'
          <<dim<<' '<<layers<<' '<<seq<<' '<<batch<<' '<<heads<<' '<<steps<<' '<<warmup<<' '
          <<eval_every<<' '<<eval_batches<<' '<<test_batches<<' '<<seed<<"\n";
        if(!legacy)s<<"AdamW .0006 .00006 .9 .95 1e-8 .1 clip1; ReLU; tied; finalLN\n";
        if(legacy)s<<"legacy-tinystories-v1; Adam .0005 .00005 .9 .999 1e-8; no decay/clipping; untied+bias; no finalLN; old init/windows/eval/early-stop; tf32="<<env("NVIDIA_TF32","0")<<"\n";
        if(vocab!=27||token_bytes!=1)s<<"tokens "<<vocab<<' '<<token_bytes<<" little-endian\n";return s.str();
    }
};
static uint64_t mix(uint64_t x){
    x+=0x9e3779b97f4a7c15ull;x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
    x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);
}
static float learning_rate(uint64_t step,const Config& c){
    if(c.legacy){
        if(step<uint64_t(c.warmup))return .0005f*float(step+1)/float(c.warmup);
        const double ratio=double(step-c.warmup)/double(c.steps-c.warmup);
        return float(.00005f+double(.0005f-.00005f)*.5*(1+std::cos(std::acos(-1.)*ratio)));
    }
    if(step<uint64_t(c.warmup))return .0006f*float(step+1)/float(c.warmup);
    const double ratio=double(step-c.warmup)/double(std::max(1,c.steps-c.warmup-1));
    return float(.00006+(.0006-.00006)*.5*(1+std::cos(std::acos(-1.)*ratio)));
}
// Retain byte storage for old character datasets. Decode int32 token files
// explicitly as little-endian, so IDs above 255 never truncate or change sign.
struct TokenData {
    std::vector<uint8_t> bytes;int width=1;
    size_t size() const {return bytes.size()/width;}
    uint32_t operator[](size_t i) const {
        if(width==1)return bytes[i];const size_t k=i*4;
        return uint32_t(bytes[k])|(uint32_t(bytes[k+1])<<8)|(uint32_t(bytes[k+2])<<16)|(uint32_t(bytes[k+3])<<24);
    }
};
static TokenData read_tokens(const fs::path& path,int seq,const Config& c){
    std::ifstream f(path,std::ios::binary|std::ios::ate);ensure(bool(f),"cannot open token data");
    auto size=f.tellg();ensure(size>0&&size<=std::streamoff(16000000000LL)&&size%c.token_bytes==0,"invalid token file size");
    TokenData data;data.width=c.token_bytes;data.bytes.resize(size_t(size));f.seekg(0);f.read(reinterpret_cast<char*>(data.bytes.data()),size);
    ensure(bool(f),"truncated token data");ensure(data.size()>size_t(seq),"token file is too short");
    for(size_t i=0;i<data.size();++i)ensure(data[i]<uint32_t(c.vocab),"invalid encoded token ID");return data;
}
static void write_token_ids(const fs::path& path,const std::vector<int32_t>& ids){
    std::ofstream f(path,std::ios::binary);ensure(bool(f),"cannot write token IDs");
    for(uint32_t id:ids){const char b[4]={char(id&255),char((id>>8)&255),char((id>>16)&255),char((id>>24)&255)};f.write(b,4);}ensure(bool(f),"token ID write failed");
}
struct Batch {std::vector<int32_t> x,y;};
static Batch get_batch(const TokenData& data,uint64_t index,const Config& c,bool training,int batches=0){
    Batch b;b.x.resize(size_t(c.seq)*c.batch);b.y.resize(b.x.size());
    const uint64_t choices=data.size()-c.seq;
    for(int j=0;j<c.batch;++j){
        uint64_t off=training?mix(uint64_t(c.seed)+index*c.batch+j)%choices:
            (choices-1)*(index*c.batch+j)/uint64_t(std::max(int64_t(1),int64_t(batches)*c.batch-1));
        if(c.legacy){
            const long hi=long(data.size())-c.seq-1;ensure(hi>0,"legacy split too short");
            off=training?uint64_t(std::uniform_int_distribution<long>(0,hi-1)(global_rand_gen)):
                uint64_t(double(index*c.batch+j)*double(hi)/double(std::max(1,batches)*c.batch));
        }
        ensure(off+uint64_t(c.seq)<data.size(),"batch crossed split boundary");
        for(int t=0;t<c.seq;++t){const auto k=size_t(j)*c.seq+t;b.x[k]=data[off+t];b.y[k]=data[off+t+1];}
    }return b;
}
// A local adapter keeps the existing standalone LayerNorm and checkpoint APIs.
template<class D> class FinalNorm : public Layer<D>{
    LayerNorm<D> norm_;
public:
    FinalNorm(int dim,int tokens):Layer<D>(1,1,0),norm_(dim,tokens){}
    void eval(const Matrix<D>& x) override {this->val=norm_.forward(x);}
    bool supports_backward() const override{return true;}
    Matrix<D> backward(const Matrix<D>&,Matrix<D>&& dy) override{return norm_.backward(std::move(dy),this->need_update);}
    Matrix<D> backward_update(const Matrix<D>&,Matrix<D>&&) override{throw std::logic_error("FinalNorm uses explicit backward");}
    Matrix<D> grad(const Matrix<D>&) const override{throw std::logic_error("LayerNorm has a full input Jacobian");}
    Matrix<D> jvp(const Matrix<D>&,const Matrix<D>& dx) override{return norm_.jvp(dx);}
    std::vector<ParameterRef<D>> parameters() override{
        auto refs=norm_.parameters();for(auto& p:refs)p.parent_trainable=&this->need_update;return refs;
    }
    std::vector<std::pair<std::string,Matrix<D>*>> checkpoint_parameters() override{return {{"gamma",&norm_.gamma},{"beta",&norm_.beta}};}
    std::vector<std::pair<std::string,adam_state<D>*>> checkpoint_optimizers() override{return {{"gamma",&norm_.adam_g},{"beta",&norm_.adam_b}};}
};
// The old combined token/position one-hot linear map also had an input bias.
// Lookup tables plus this bias compute the same map without dense one-hot data.
template<class D> class InputBias : public Layer<D>{
public:
    explicit InputBias(int dim):Layer<D>(dim,1,0){}
    void eval(const Matrix<D>& x) override{this->val=x+this->bias*Matrix<D>::ones(1,x.num_col());}
    bool supports_backward() const override{return true;}
    Matrix<D> backward(const Matrix<D>&,Matrix<D>&& dy) override{
        if(this->need_update)this->parameter_gradients.add(this->bias,sum(dy,1));return std::move(dy);
    }
    std::vector<ParameterRef<D>> parameters() override{return {this->parameter_gradients.parameter("bias",this->bias,this->adamb,this->need_update,false)};}
    std::vector<std::pair<std::string,Matrix<D>*>> checkpoint_parameters() override{return {{"bias",&this->bias}};}
    std::vector<std::pair<std::string,adam_state<D>*>> checkpoint_optimizers() override{return {{"bias",&this->adamb}};}
};
template<class D> struct Model {
    Embedding<D> token,position;FinalNorm<D> norm;TiedProjection<D> head;
    std::unique_ptr<InputBias<D>> input_bias;std::unique_ptr<LinearLayer<D>> independent_head;
    std::vector<std::unique_ptr<TransformerLayer<D>>> blocks;
    TokenIds<D> positions;Matrix<D> input;Sequential<D> net;
    static std::vector<int32_t> position_ids(const Config& c){std::vector<int32_t> x;for(int b=0;b<c.batch;++b)for(int t=0;t<c.seq;++t)x.push_back(t);return x;}
private:
    std::list<Layer<D>*> build_layers(const Config& c){
        std::list<Layer<D>*> layers;
        for(int i=0;i<c.layers;++i)blocks.push_back(std::make_unique<TransformerLayer<D>>(c.dim,c.dim,4*c.dim,c.seq,c.batch,c.heads,true));
        if(c.legacy){input_bias=std::make_unique<InputBias<D>>(c.dim);independent_head=std::make_unique<LinearLayer<D>>(c.vocab,c.dim,c.seq*c.batch);layers={independent_head.get()};}
        else layers={&head,&norm};
        for(auto i=blocks.rbegin();i!=blocks.rend();++i)layers.push_back(i->get());
        if(input_bias)layers.push_back(input_bias.get());
        return layers;
    }
public:
    explicit Model(const Config& c):token(c.vocab,c.dim),position(c.seq,c.dim),norm(c.dim,c.seq*c.batch),head(token),positions(position_ids(c)),net(build_layers(c)){}
    const Matrix<D>& logits() const{return net.value();}
    std::vector<ParameterRef<D>> parameters(){
        std::vector<ParameterRef<D>> refs;
        auto append=[&](auto entries,const std::string& prefix){for(auto p:entries){p.name=prefix+p.name;refs.push_back(p);}};
        append(token.parameters(),"token.");append(position.parameters(),"position.");
        if(input_bias)append(input_bias->parameters(),"input.");
        for(size_t i=0;i<blocks.size();++i)append(blocks[i]->parameters(),"blocks."+std::to_string(i)+".");
        if(independent_head)append(independent_head->parameters(),"projection.");else append(norm.parameters(),"final.");return refs;
    }
    auto checkpoint_net(){auto list=net.layers();list.push_back(&token);list.push_back(&position);return list;}
    void eval(const std::vector<int32_t>& x){token.eval(TokenIds<D>(x));position.eval(positions);input=token.value()+position.value();net.eval(input);}
    void backward(Matrix<D>&& dy){auto dx=net.backward(std::move(dy));token.backward(dx);position.backward(dx);}
};
static AdamWOptions optimizer_options(const Config& c){return c.legacy?AdamWOptions{.0005f,.9f,.999f,1e-8f,0.f}:AdamWOptions{.0006f,.9f,.95f,1e-8f,.1f};}
template<class D> static Matrix<float> host(const Matrix<D>& m){if constexpr(std::is_same_v<D,float>)return m;else return m.to_host();}
static void synchronize(){
#ifdef CUDA
    CudaErrorCheck(cudaDeviceSynchronize());
#endif
}
template<class D> static void initialize(Model<D>& model,const Config& c){
    std::mt19937 rng(c.seed);std::normal_distribution<float> normal(0,1);
    for(auto p:model.parameters()){
        Matrix<float> w("initial",p.value->num_row(),p.value->num_col());
        const bool gamma=p.name.find("gamma")!=std::string::npos;
        float stddev=.02f;if(p.name.ends_with(".Wo")||p.name.ends_with(".W2"))stddev/=std::sqrt(float(2*c.layers));
        if(c.legacy){
            if(p.name.starts_with("blocks."))stddev=1.f/std::sqrt(float(p.name.ends_with(".W2")?4*c.dim:c.dim));
            else stddev=.001f;
        }
        const bool legacy_random_bias=c.legacy&&(p.name=="input.bias"||p.name=="projection.bias");
        for(size_t j=0;j<w.num_col();++j)for(size_t i=0;i<w.num_row();++i)w.elem(i,j)=gamma?1.f:((p.decay||legacy_random_bias)?stddev*normal(rng):0.f);
        *p.value=Matrix<D>(w);
    }
}
// Canonical column-major float32 archive shared with the independent Torch run.
template<class D> static void archive(const fs::path& path,const std::vector<ParameterRef<D>>& refs,bool gradients=false){
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(path.string().c_str(),"wb"),&fclose);ensure(bool(f),"cannot write parameter archive");
    ensure(fwrite("T8PAR001",1,8,f.get())==8,"archive header");uint32_t count=uint32_t(refs.size());ensure(checkpoint_detail::put(f.get(),count),"archive count");
    for(const auto& p:refs){auto m=host(gradients?p.grad():*p.value);
        ensure(checkpoint_detail::put_string(f.get(),p.name),"archive name");uint64_t rows=m.num_row(),cols=m.num_col();
        ensure(checkpoint_detail::put(f.get(),rows)&&checkpoint_detail::put(f.get(),cols),"archive shape");
        for(size_t j=0;j<cols;++j)for(size_t i=0;i<rows;++i){float x=m.elem(i,j);ensure(checkpoint_detail::put(f.get(),x),"archive data");}
    }ensure(fflush(f.get())==0,"archive flush");
}
template<class D> static double evaluate(Model<D>& model,const TokenData& data,const Config& c,int batches){
    TokenCrossEntropy<D> loss;double total=0;
    for(int i=0;i<batches;++i){auto b=get_batch(data,i,c,false,batches);model.eval(b.x);loss.eval(model.logits(),TokenIds<D>(std::move(b.y)));double value=host(loss.value()).elem(0,0);ensure(std::isfinite(value),"nonfinite evaluation loss");total+=value;}
    return total/batches;
}
static void save_spec(const Config& c,bool resume){
    auto path=c.output/"run-spec.txt";
    if(fs::exists(path)){std::ifstream f(path);std::string old((std::istreambuf_iterator<char>(f)),{});ensure(old==c.spec(),"resume configuration/data differs from saved protocol");}
    else{ensure(!resume,"missing checkpoint protocol");std::ofstream f(path);f<<c.spec();ensure(bool(f),"cannot write run protocol");}
}
static void trim_metrics(const fs::path& path,uint64_t step){
    std::ifstream f(path);ensure(bool(f),"missing resume metrics");std::string line,kept;
    while(std::getline(f,line)){if(line.starts_with("step,")||std::stoull(line.substr(0,line.find(',')))<=step)kept+=line+'\n';}
    f.close();std::ofstream out(path);out<<kept;ensure(bool(out),"cannot trim resume metrics");
}
template<class D> static void train(const Config& c){
    auto train_data=read_tokens(c.data/"train.bin",c.seq,c),valid=read_tokens(c.data/"valid.bin",c.seq,c),test=c.legacy?TokenData{}:read_tokens(c.data/"test.bin",c.seq,c);
    ensure(fs::is_directory(c.output),"output directory must already exist");
    Model<D> model(c);AdamWOptions options=optimizer_options(c);AdamW<D> optimizer(model.parameters(),options);
    auto checkpoint=c.output/"last.ckpt";const bool resume=fs::exists(checkpoint);save_spec(c,resume);
    TrainingProgress progress;std::string error;
    if(resume){ensure(load_checkpoint(model.checkpoint_net(),checkpoint.string(),progress,&error),"checkpoint load: "+error);
        ensure(progress.step<=uint64_t(c.steps)&&progress.data_position==progress.step*uint64_t(c.seq)*c.batch,"invalid checkpoint progress");trim_metrics(c.output/"metrics.csv",progress.step);}
    else{ensure(!fs::exists(c.output/"initial.params"),"incomplete run exists; choose a new output directory");initialize(model,c);if(c.legacy)global_rand_gen.seed(c.seed);archive(c.output/"initial.params",optimizer.parameters());}
    std::ofstream csv(c.output/"metrics.csv",resume?std::ios::app:std::ios::out);csv<<std::setprecision(10);ensure(bool(csv),"cannot open metrics");
    if(!resume){csv<<"step,train_loss,lr,grad_norm,step_ms,val_loss\n";double initial=evaluate(model,valid,c,c.eval_batches);csv<<"0,,,,,"<<initial<<'\n';csv.flush();std::cout<<"step=0 val_loss="<<initial<<std::endl;}
    size_t count=0;for(auto p:optimizer.parameters())count+=p.value->num_row()*p.value->num_col();std::cout<<"parameters="<<count<<" resumed_step="<<progress.step<<std::endl;
    double best_val=std::numeric_limits<double>::infinity();uint64_t best_step=0;
    const auto best_meta=c.output/"best-validation.txt";
    if(resume&&fs::exists(best_meta)){
        std::ifstream f(best_meta);f>>best_step>>best_val;
        ensure(bool(f)&&std::isfinite(best_val)&&best_step<=progress.step,"invalid best validation metadata");
        ensure(fs::exists(c.output/"best.ckpt")&&fs::exists(c.output/"best.params"),"missing best validation model");
    }
    TokenCrossEntropy<D> loss;
    while(progress.step<uint64_t(c.stop)){
        synchronize();const auto started=std::chrono::steady_clock::now();
        auto b=get_batch(train_data,progress.step,c,true);
        if(c.legacy&&progress.step==0){write_token_ids(c.output/"first-input.ids",b.x);write_token_ids(c.output/"first-target.ids",b.y);}
        optimizer.zero_grad();model.eval(b.x);
        loss.eval(model.logits(),TokenIds<D>(std::move(b.y)));model.backward(loss.grad());
        if(progress.step==0)archive(c.output/"first.grad",optimizer.parameters(),true);
        double norm=optimizer.clip_grad_norm(c.legacy?1e30:1);float lr=learning_rate(progress.step,c);optimizer.set_lr(lr);optimizer.step();
        double train_loss=host(loss.value()).elem(0,0);synchronize();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        ensure(std::isfinite(train_loss)&&std::isfinite(norm),"nonfinite training");++progress.step;progress.data_position+=uint64_t(c.seq)*c.batch;
        if(progress.step==1)archive(c.output/"first.params",optimizer.parameters());
        csv<<progress.step<<','<<train_loss<<','<<lr<<','<<norm<<','<<ms<<',';
        if((c.legacy?(progress.step-1)%c.eval_every==0:progress.step%c.eval_every==0)||progress.step==uint64_t(c.stop)){
            double val=evaluate(model,valid,c,c.eval_batches);csv<<val;csv<<'\n';csv.flush();
            ensure(save_checkpoint(model.checkpoint_net(),checkpoint.string(),progress,&error),"checkpoint save: "+error);
            if(val<best_val){
                fs::copy_file(checkpoint,c.output/"best.ckpt.tmp",fs::copy_options::overwrite_existing);
                fs::rename(c.output/"best.ckpt.tmp",c.output/"best.ckpt");
                archive(c.output/"best.params",optimizer.parameters());
                std::ofstream f(best_meta);f<<std::setprecision(17)<<progress.step<<' '<<val<<'\n';
                ensure(bool(f),"cannot save best validation metadata");best_val=val;best_step=progress.step;
            }
            std::cout<<"step="<<progress.step<<" train_loss="<<train_loss<<" val_loss="<<val<<" step_ms="<<ms<<std::endl;
            if(c.legacy&&progress.step>=best_step+uint64_t(12*c.eval_every)){
                std::ofstream done(c.output/"early-stopping.json");done<<"{\"steps\":"<<progress.step<<",\"best_step\":"<<best_step<<",\"best_val_loss\":"<<best_val<<"}\n";break;
            }
        }else csv<<'\n';
    }
    csv.flush();archive(c.output/"final.params",optimizer.parameters());
    if(c.legacy){
        std::ofstream out(c.output/"summary.json");out<<std::setprecision(12)<<"{\"steps\":"<<progress.step<<",\"budget\":"<<c.steps<<",\"parameters\":"<<count<<",\"best_step\":"<<best_step<<",\"best_val_loss\":"<<best_val<<",\"best_val_ppl\":"<<std::exp(best_val)<<",\"early_stopped\":"<<(progress.step<uint64_t(c.stop)?"true":"false")<<"}\n";
    }else if(progress.step==uint64_t(c.steps)){
        double test_loss=evaluate(model,test,c,c.test_batches);std::ofstream out(c.output/"summary.json");
        out<<std::setprecision(10)<<"{\"steps\":"<<progress.step<<",\"parameters\":"<<count<<",\"test_loss\":"<<test_loss
           <<",\"test_ppl\":"<<std::exp(test_loss)<<(c.token_bytes==1?",\"test_bpc\":":",\"test_bits_per_token\":")<<test_loss/std::log(2.)<<",\"test_tokens\":"<<int64_t(c.test_batches)*c.seq*c.batch<<"}\n";
        ensure(bool(out),"cannot write summary");std::cout<<"test_loss="<<test_loss<<" test_ppl="<<std::exp(test_loss)<<std::endl;
    }
}
// Frozen evaluation writes only to an explicit destination; training files stay sealed.
template<class D> static void evaluate_only(const Config& c){
    fs::path destination=env("T8_EVALUATION_OUTPUT");ensure(!destination.empty(),"T8_EVALUATION_OUTPUT is required");
    ensure(fs::is_directory(destination.parent_path()),"evaluation output directory must exist");
    auto data=read_tokens(c.data/"valid.bin",c.seq,c);Model<D> model(c);TrainingProgress progress;std::string error;
    ensure(load_checkpoint(model.checkpoint_net(),env("T8_CHECKPOINT",(c.output/"best.ckpt").string()),progress,&error),"evaluation load: "+error);
    const double loss=evaluate(model,data,c,c.eval_batches);std::ofstream f(destination);
    f<<std::setprecision(12)<<"{\"checkpoint_step\":"<<progress.step<<",\"validation_loss\":"<<loss<<",\"validation_ppl\":"<<std::exp(loss)<<",\"batches\":"<<c.eval_batches<<",\"tokens\":"<<int64_t(c.eval_batches)*c.batch*c.seq<<"}\n";
    ensure(bool(f),"cannot write frozen evaluation");
}
// Profiling is separate from throughput measurement: events add overhead.
// Stream elapsed time includes launch gaps, not just time executing kernels.
struct PhaseProfile {
    static constexpr int phases=8;
    const char* names[phases]={"batch","zero_grad","forward","loss","backward","clip","step","loss_read"};
    double wall[phases]{};
    bool isolated=env("T8_PROFILE_SYNC","0")=="1";
#ifdef CUDA
    cudaEvent_t begin[phases]{},end[phases]{};
    PhaseProfile(){for(int i=0;i<phases;++i){CudaErrorCheck(cudaEventCreate(&begin[i]));CudaErrorCheck(cudaEventCreate(&end[i]));}}
    ~PhaseProfile(){for(int i=0;i<phases;++i){CudaErrorCheck(cudaEventDestroy(begin[i]));CudaErrorCheck(cudaEventDestroy(end[i]));}}
#endif
    template<class F> void measure(int i,F&& f){
        if(isolated)synchronize();
#ifdef CUDA
        CudaErrorCheck(cudaEventRecord(begin[i]));
#endif
        auto start=std::chrono::steady_clock::now();f();
#ifdef CUDA
        CudaErrorCheck(cudaEventRecord(end[i]));
#endif
        if(isolated)synchronize();
        wall[i]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    void write(std::ofstream& csv,int iteration){
        synchronize();
        for(int i=0;i<phases;++i){double stream=wall[i];
#ifdef CUDA
            float elapsed=0;CudaErrorCheck(cudaEventElapsedTime(&elapsed,begin[i],end[i]));stream=elapsed;
#endif
            csv<<iteration<<','<<names[i]<<','<<wall[i]<<','<<stream<<','<<isolated<<'\n';
        }
    }
};
template<class D> static void profile(const Config& c){
    ensure(fs::is_directory(c.output),"profile output directory must exist");
    auto data=read_tokens(c.data/"train.bin",c.seq,c);Model<D> model(c);initialize(model,c);
    AdamW<D> optimizer(model.parameters(),optimizer_options(c));TokenCrossEntropy<D> loss;
    PhaseProfile timer;std::ofstream csv(c.output/"profile.csv");csv<<std::setprecision(10)<<"step,phase,host_ms,stream_elapsed_ms,isolated\n";
    for(int iteration=0;iteration<150;++iteration){
        synchronize();Batch b;double value=0;
        timer.measure(0,[&]{b=get_batch(data,iteration,c,true);});
        timer.measure(1,[&]{optimizer.zero_grad();});
        timer.measure(2,[&]{model.eval(b.x);});
        timer.measure(3,[&]{loss.eval(model.logits(),TokenIds<D>(std::move(b.y)));});
        timer.measure(4,[&]{model.backward(loss.grad());});
        timer.measure(5,[&]{optimizer.clip_grad_norm(c.legacy?1e30:1);});
        timer.measure(6,[&]{optimizer.step();});
        timer.measure(7,[&]{value=host(loss.value()).elem(0,0);});
        if(iteration>=50)timer.write(csv,iteration);ensure(std::isfinite(value),"nonfinite profiled loss");
    }ensure(bool(csv),"profile write failed");
}
// Matched throughput protocol: 3 rounds, each 50 warmup + 100 timed steps.
template<class D> static void benchmark(const Config& c){
    ensure(fs::is_directory(c.output),"benchmark output directory must exist");
    auto data=read_tokens(c.data/"train.bin",c.seq,c);Model<D> model(c);initialize(model,c);
    AdamW<D> optimizer(model.parameters(),optimizer_options(c));TokenCrossEntropy<D> loss;
    std::vector<double> samples,medians;
    for(int step=0;step<450;++step){
        synchronize();auto started=std::chrono::steady_clock::now();auto b=get_batch(data,step,c,true);
        optimizer.zero_grad();model.eval(b.x);loss.eval(model.logits(),TokenIds<D>(std::move(b.y)));model.backward(loss.grad());
        optimizer.clip_grad_norm(c.legacy?1e30:1);optimizer.step();double value=host(loss.value()).elem(0,0);synchronize();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();ensure(std::isfinite(value),"nonfinite benchmark loss");
        if(step%150>=50)samples.push_back(ms);
    }
    for(int round=0;round<3;++round){std::vector<double> xs(samples.begin()+100*round,samples.begin()+100*(round+1));std::sort(xs.begin(),xs.end());medians.push_back((xs[49]+xs[50])*.5);}
    auto ordered=medians;std::sort(ordered.begin(),ordered.end());double median=ordered[1];
    std::ofstream out(c.output/"benchmark.json");out<<std::setprecision(10)<<"{\"rounds\":3,\"warmup_steps_per_round\":50,\"measured_steps\":300,\"median_step_ms\":"<<median
       <<",\"median_tokens_per_second\":"<<c.batch*c.seq*1000./median<<",\"round_medians_ms\":[";
    for(size_t i=0;i<medians.size();++i){if(i)out<<',';out<<medians[i];}out<<"],\"samples_ms\":[";
    for(size_t i=0;i<samples.size();++i){if(i)out<<',';out<<samples[i];}out<<"]}\n";ensure(bool(out),"benchmark write failed");
    std::cout<<"benchmark median_ms="<<median<<std::endl;
}
template<class D> static void generate(Config c){
    if(!c.generate||c.stop<c.steps)return;c.batch=1;Model<D> model(c);TrainingProgress progress;std::string error;
    ensure(load_checkpoint(model.checkpoint_net(),env("T8_CHECKPOINT",(c.output/"last.ckpt").string()),progress,&error),"generation load: "+error);
    fs::path generated=env("T8_GENERATION_OUTPUT",c.output.string());ensure(fs::is_directory(generated),"generation output directory must exist");
    const auto raw_temperature=env("T8_TEMPERATURE","0.8");size_t parsed=0;double temperature=std::stod(raw_temperature,&parsed);
    ensure(parsed==raw_temperature.size()&&std::isfinite(temperature)&&temperature>=0,"invalid T8_TEMPERATURE");
    const int top_k=number("T8_TOP_K",10,0,c.vocab);
    const int sample_seed=number("T8_SAMPLE_SEED",c.seed+1000,0,1000000);
    const std::string alphabet=" abcdefghijklmnopqrstuvwxyz";std::vector<int32_t> tokens;
    if(c.token_bytes==1){ensure(c.vocab==27,"character generation requires the text8 alphabet");for(char ch:std::string("the meaning of life is "))tokens.push_back(int32_t(alphabet.find(ch)));}
    else {const auto path=env("T8_PROMPT_IDS");ensure(!path.empty(),"T8_PROMPT_IDS is required for token generation");auto data=read_tokens(path,0,c);for(size_t i=0;i<data.size();++i)tokens.push_back(int32_t(data[i]));}
    std::mt19937 rng(sample_seed);ensure(int64_t(c.vocab)*c.generate<=100000000,"generation trace too large");
    Matrix<float> trace("generation logits",c.vocab,c.generate);
    for(int n=0;n<c.generate;++n){
        const auto start=tokens.size()>size_t(c.seq)?tokens.size()-c.seq:0;
        std::vector<int32_t> ids(tokens.begin()+start,tokens.end());size_t last=ids.size()-1;
        if(c.legacy){ids.insert(ids.begin(),c.seq-ids.size(),number("T8_EOT_ID",0,0,c.vocab-1));last=c.seq-1;}else ids.resize(c.seq,0);
        model.eval(ids);auto logits=host(model.logits());
        std::vector<int> order(c.vocab);std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int a,int b){return logits.elem(a,last)>logits.elem(b,last);});
        for(int i=0;i<c.vocab;++i)trace.elem(i,n)=logits.elem(i,last);
        if(temperature==0)tokens.push_back(order[0]);
        else{
            std::vector<double> weights;for(int i=0;i<(top_k?top_k:c.vocab);++i)weights.push_back(std::exp(double(logits.elem(order[i],last)-logits.elem(order[0],last))/temperature));
            std::discrete_distribution<int> pick(weights.begin(),weights.end());tokens.push_back(order[pick(rng)]);
        }
    }
    archive(generated/"generation.params",std::vector<ParameterRef<float>>{{"logits",&trace,nullptr,nullptr,nullptr}});
    if(c.token_bytes==4)write_token_ids(generated/"generated.ids",tokens);
    else {std::string text;for(auto id:tokens)text+=alphabet[id];std::ofstream out(generated/"sample.txt");out<<text<<'\n';ensure(bool(out),"cannot write sample");std::cout<<"sample: "<<text<<std::endl;}

}
} // namespace text8
int compute(){try{global_rand_gen.seed(42);
#ifdef CUDA
    GPUSampler sampler(42);
#endif
#ifdef CUDA
    text8::ensure(text8::env("T8_EXPECT_DEVICE","cuda")=="cuda","binary backend differs from requested device");
#else
    text8::ensure(text8::env("T8_EXPECT_DEVICE","cpu")=="cpu","binary backend differs from requested device");
#endif
    text8::Config c;
    if(text8::env("T8_EVALUATE_ONLY","0")=="1")text8::evaluate_only<Backend>(c);
    else if(text8::env("T8_GENERATE_ONLY","0")=="1")text8::generate<Backend>(c);
    else if(text8::env("T8_PROFILE","0")=="1")text8::profile<Backend>(c);
    else if(text8::env("T8_BENCHMARK","0")=="1")text8::benchmark<Backend>(c);
    else{text8::train<Backend>(c);text8::generate<Backend>(c);}return 0;
}catch(const std::exception& e){std::cerr<<"text8: "<<e.what()<<std::endl;return 1;}}
