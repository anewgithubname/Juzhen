/** Small token LM demonstrating explicit gradients, tied weights and resume.
 * The synthetic cyclic token stream checks learning, not language quality.
 * JUZHEN_LLM_STEPS sets the total update count (default 200).
 * JUZHEN_LLM_CHECKPOINT optionally selects a checkpoint file. Models and data
 * are written only when the caller explicitly supplies that path.
 */
#include "../ml/llm.hpp"
#include "../ml/checkpoint.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
using namespace Juzhen;
#ifdef CUDA
using Backend=CUDAfloat;
#else
using Backend=float;
#endif

template<class D> float scalar(const Matrix<D>& m) {
    if constexpr(std::is_same_v<D,float>)return m.elem(0,0);
    else return m.to_host().elem(0,0);
}
int compute() {
    try {
        global_rand_gen.seed(42);
#ifdef CUDA
        GPUSampler sampler(42);
#endif
        constexpr int vocab=32,dim=16,seq=8,batch=2,accumulation=2;
        constexpr int tokens=seq*batch;
        int steps=200;
        if(const char* raw=std::getenv("JUZHEN_LLM_STEPS")) {
            char* end=nullptr;const long n=std::strtol(raw,&end,10);
            if(end==raw||*end||n<1||n>10000000)throw std::invalid_argument("invalid JUZHEN_LLM_STEPS");
            steps=int(n);
        }
        Embedding<Backend> token(vocab,dim),position(seq,dim);
        token.weight().scale(20.f);position.weight().scale(20.f);
        TransformerLayer<Backend> block(dim,dim,4*dim,seq,batch,4,true);
        TiedProjection<Backend> output(token);
        Sequential<Backend> network{&output,&block};
        auto refs=network.parameters();
        for(auto p:position.parameters()){p.name="position";refs.push_back(p);}
        AdamWOptions config;config.lr=.003f;config.weight_decay=.01f;
        AdamW<Backend> optimizer(refs,config);
        std::vector<int32_t> positions;for(int b=0;b<batch;++b)for(int t=0;t<seq;++t)positions.push_back(t);
        TokenIds<Backend> position_ids(std::move(positions));
        TokenCrossEntropy<Backend> loss;
        TrainingProgress progress;std::string error,path;
        const std::list<Layer<Backend>*> checkpoint_network{&output,&block,&position,&token};
        if(const char* selected=std::getenv("JUZHEN_LLM_CHECKPOINT")) {
            path=selected;
            if(path.empty())throw std::invalid_argument("checkpoint path is empty");
            if(std::filesystem::exists(path)&&!load_checkpoint(checkpoint_network,path,progress,&error))
                throw std::runtime_error("checkpoint load: "+error);
        }
        const uint64_t initial_step=progress.step;
        if(initial_step>=uint64_t(steps)) {
            std::cout<<"Checkpoint already completed "<<initial_step<<" updates.\n";
            return 0;
        }
        float first=-1,last=-1;
        while(progress.step<uint64_t(steps)) {
            optimizer.zero_grad();double total_loss=0;
            for(int micro=0;micro<accumulation;++micro) {
                std::vector<int32_t> x,y;
                for(int b=0;b<batch;++b)for(int t=0;t<seq;++t) {
                    const int id=int((progress.step*11+micro*5+b*3+t)%vocab);
                    x.push_back(id);y.push_back((id+1)%vocab);
                }
                token.eval(TokenIds<Backend>(std::move(x)));position.eval(position_ids);
                auto input=token.value()+position.value();
                loss.eval(network.eval(input),TokenIds<Backend>(std::move(y)));
                // Equal valid-token counts across microbatches: average their means.
                auto dx=network.backward(loss.grad(1.f/accumulation));
                token.backward(dx);position.backward(dx);
                total_loss+=scalar(loss.value())/accumulation;
            }
            optimizer.clip_grad_norm(1.0);
            optimizer.step();++progress.step;progress.data_position+=tokens*accumulation;
            last=float(total_loss);if(first<0)first=last;
            if(!std::isfinite(last))throw std::runtime_error("non-finite training loss");
            if(progress.step==1||progress.step%25==0||progress.step==uint64_t(steps))
                std::cout<<"step="<<progress.step<<" token_loss="<<last<<'\n';
        }
        if(!path.empty()&&!save_checkpoint(checkpoint_network,path,progress,&error))
            throw std::runtime_error("checkpoint save: "+error);
        std::cout<<"RESULT first_loss="<<first<<" final_loss="<<last<<" completed_steps="<<progress.step<<'\n';
        if(initial_step==0 && steps>=100 && first>0 && progress.step==uint64_t(steps) && !(last<first*.7f))
            throw std::runtime_error("synthetic token LM did not learn");
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
