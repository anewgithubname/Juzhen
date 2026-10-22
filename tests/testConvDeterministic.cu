// Independent scalar oracle, bitwise replay, and process-restart coverage.
#include "../cpp/juzhen.hpp"
#include "../ml/checkpoint.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
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
static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
template<class T> static Matrix<float> host(const Matrix<T>& x) {
    if constexpr (std::is_same_v<T,float>) return x;
    else return x.to_host();
}
static Matrix<float> values(int rows, int cols, float phase) {
    Matrix<float> x("fixture",rows,cols);
    for(int j=0;j<cols;++j) for(int i=0;i<rows;++i)
        x.elem(i,j)=.15f*std::sin((i+rows*j)*.173f+phase);
    return x;
}
static void close(const Matrix<float>& a,const Matrix<float>& b,const std::string& tag) {
    require(a.num_row()==b.num_row()&&a.num_col()==b.num_col(),tag+" shape");
    float worst=0;
    for(size_t j=0;j<a.num_col();++j) for(size_t i=0;i<a.num_row();++i) {
        float d=std::abs(a.elem(i,j)-b.elem(i,j)); worst=std::max(worst,d);
        require(std::isfinite(d)&&d<=3e-5f+2e-4f*std::abs(b.elem(i,j)),tag+" numerical mismatch");
    }
    std::cout<<tag<<" max_abs="<<worst<<'\n';
}
static void exact(const Matrix<float>& a,const Matrix<float>& b,const std::string& tag) {
    require(a.num_row()==b.num_row()&&a.num_col()==b.num_col(),tag+" shape");
    require(std::memcmp(a.data(),b.data(),a.num_row()*a.num_col()*sizeof(float))==0,tag+" bitwise mismatch");
}
struct Shape {int n,ci,h,w,co,kh,kw,pad,stride; bool relu;};
static int oh(const Shape& s,bool trans) {return trans?(s.h-1)*s.stride-2*s.pad+s.kh:(s.h+2*s.pad-s.kh)/s.stride+1;}
static int ow(const Shape& s,bool trans) {return trans?(s.w-1)*s.stride-2*s.pad+s.kw:(s.w+2*s.pad-s.kw)/s.stride+1;}
static std::unique_ptr<Layer<D>> layer(const Shape& s,bool trans,int mode) {
    // mode -1 exercises the old call signature with no new argument.
    if(trans) {
        if(mode<0) return std::make_unique<ConvTransLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu);
        return std::make_unique<ConvTransLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu,mode!=0);
    }
    if(mode<0) return std::make_unique<ConvLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu);
    return std::make_unique<ConvLayer>(s.n,s.ci,s.h,s.w,s.co,s.kh,s.kw,s.pad,s.stride,s.relu,mode!=0);
}
struct Oracle {Matrix<float> y,dx,dw,db;};
static Oracle oracle(const Shape& s,bool trans,const Matrix<float>& x,const Matrix<float>& w,
                     const Matrix<float>& b,const Matrix<float>& g) {
    int H=oh(s,trans),W=ow(s,trans);
    Oracle r{Matrix<float>::zeros(s.co*H*W,s.n),Matrix<float>::zeros(s.ci*s.h*s.w,s.n),
             Matrix<float>::zeros(s.ci*s.co*s.kh*s.kw,1),Matrix<float>::zeros(s.co,1)};
    // Enumerate the linear operator's edges; transpose swaps source/target
    // coordinates and uses the documented input-major filter layout.
    auto edges=[&](auto visit) {
        for(int n=0;n<s.n;++n) for(int ci=0;ci<s.ci;++ci) for(int co=0;co<s.co;++co)
        for(int h=0;h<(trans?s.h:H);++h) for(int v=0;v<(trans?s.w:W);++v)
        for(int kh=0;kh<s.kh;++kh) for(int kw=0;kw<s.kw;++kw) {
            int a=h*s.stride-s.pad+kh,c=v*s.stride-s.pad+kw;
            if(a<0||c<0||a>=(trans?H:s.h)||c>=(trans?W:s.w)) continue;
            int xi=trans?(ci*s.h+h)*s.w+v:(ci*s.h+a)*s.w+c;
            int yi=trans?(co*H+a)*W+c:(co*H+h)*W+v;
            int wi=((trans?ci*s.co+co:co*s.ci+ci)*s.kh+kh)*s.kw+kw;
            visit(n,xi,yi,wi);
        }
    };
    edges([&](int n,int xi,int yi,int wi){r.y.elem(yi,n)+=x.elem(xi,n)*w.elem(wi,0);});
    auto t=Matrix<float>(g);
    for(int n=0;n<s.n;++n) for(int co=0;co<s.co;++co) for(int i=0;i<H*W;++i) {
        int yi=co*H*W+i;float y=r.y.elem(yi,n)+b.elem(co,0);
        r.y.elem(yi,n)=s.relu?std::max(0.f,y):y;
        if(s.relu&&y<=0) t.elem(yi,n)=0;
        r.db.elem(co,0)+=t.elem(yi,n);
    }
    edges([&](int n,int xi,int yi,int wi){
        r.dx.elem(xi,n)+=t.elem(yi,n)*w.elem(wi,0);
        r.dw.elem(wi,0)+=t.elem(yi,n)*x.elem(xi,n);
    });
    return r;
}
static std::vector<Matrix<float>> snapshot(Layer<D>& a) {
    std::vector<Matrix<float>> out;
    for(auto p:a.checkpoint_parameters()) out.push_back(host(*p.second));
    for(auto p:a.checkpoint_optimizers()) {out.push_back(host(p.second->m));out.push_back(host(p.second->v));}
    return out;
}
static void fixture(const Shape& s,bool trans) {
    auto x=values(s.ci*s.h*s.w,s.n,.2f),w=values(s.ci*s.co*s.kh*s.kw,1,.7f),
         b=values(s.co,1,1.2f),g=values(s.co*oh(s,trans)*ow(s,trans),s.n,1.9f);
    auto ref=oracle(s,trans,x,w,b,g);
    std::vector<Matrix<float>> first,default_state;
    Matrix<float> first_y,first_dx,first_jvp;
    for(int mode:{-1,0,1}) for(int repeat=0;repeat<(mode==1?10:1);++repeat) {
        auto a=layer(s,trans,mode);a->W()=Matrix<D>(w);a->b()=Matrix<D>(b);
        Matrix<D> input(x);a->eval(input);auto y=host(a->value());
        auto jvp=host(a->jvp(input,input));auto dx=host(a->backward_update(input,Matrix<D>(g)));
        auto state=snapshot(*a);
        if(repeat==0) {
            std::string tag=std::string(trans?"transpose":"conv")+" mode="+std::to_string(mode);
            close(y,ref.y,tag+" y");close(dx,ref.dx,tag+" dx");
            // First Adam moment exposes the raw parameter gradient without
            // relying on Adam's approximately sign-only first weight update.
            close(host(a->adamWstate().m)/(1-a->adamWstate().beta1),ref.dw,tag+" dw");
            close(host(a->adambstate().m)/(1-a->adambstate().beta1),ref.db,tag+" db");
            auto tangent=oracle(s,trans,x,w,Matrix<float>::zeros(s.co,1),g).y;
            // Oracle tangent must use the original forward activation mask.
            if(s.relu) {
                Shape linear=s;linear.relu=false;
                tangent=oracle(linear,trans,x,w,Matrix<float>::zeros(s.co,1),g).y;
                for(size_t j=0;j<tangent.num_col();++j) for(size_t i=0;i<tangent.num_row();++i)
                    if(ref.y.elem(i,j)<=0) tangent.elem(i,j)=0;
            }
            close(jvp,tangent,tag+" jvp");
        }
        if(mode==-1) {default_state=state;first_y=y;first_dx=dx;first_jvp=jvp;}
        if(mode==0) {
            close(y,first_y,"default y");close(dx,first_dx,"default dx");close(jvp,first_jvp,"default jvp");
            for(size_t i=0;i<state.size();++i) close(state[i],default_state[i],"default state");
        }
        if(mode==1&&repeat==0) {first=state;first_y=y;first_dx=dx;first_jvp=jvp;}
        if(mode==1&&repeat>0) {
            exact(y,first_y,"repeat y");exact(dx,first_dx,"repeat dx");exact(jvp,first_jvp,"repeat jvp");
            for(size_t i=0;i<state.size();++i) exact(state[i],first[i],"repeat state");
        }
    }
}
static std::string bytes(const std::filesystem::path& p) {
    std::ifstream f(p,std::ios::binary);require(bool(f),"missing checkpoint");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
static void restart(const std::string& phase,const std::filesystem::path& out) {
    require(std::filesystem::is_directory(out),"checkpoint output must already exist");
    global_rand_gen.seed(phase=="prepare"?123:987);
    Shape s{2,3,11,13,4,3,3,1,1,true};auto a=layer(s,false,1);
    s.ci=4;s.co=3;s.relu=false;auto b=layer(s,true,1);
    std::list<Layer<D>*> net={b.get(),a.get()};TrainingProgress p;std::string why;
    if(phase=="resume") require(load_checkpoint(net,(out/"split.ckpt").string(),p,&why),why);
    for(;p.step<8;++p.step) {
        Matrix<D> x(values(3*11*13,2,float(p.step))),g(values(3*11*13,2,float(p.step)+2));
        a->eval(x);b->eval(a->value());auto dx=b->backward_update(a->value(),std::move(g));a->backward_update(x,std::move(dx));
        if(phase=="prepare"&&p.step==2) {auto saved=p;saved.step=3;require(save_checkpoint(net,(out/"split.ckpt").string(),saved,&why),why);}
    }
    auto file=out/(phase=="prepare"?"continuous.ckpt":"resumed.ckpt");
    require(save_checkpoint(net,file.string(),p,&why),why);
    if(phase=="resume") require(bytes(out/"continuous.ckpt")==bytes(file),"separate-process checkpoint differs");
    std::cout<<"PASS separate-process "<<phase<<'\n';
}
static void device_sync() {
#ifdef CUDA
    CudaErrorCheck(cudaDeviceSynchronize());
#endif
}
static void bench() {
    for(bool trans:{false,true}) for(int size:{16,64}) {
        Shape s{8,16,size,size,32,trans?4:3,trans?4:3,1,trans?2:1,true};
        for(int mode:{-1,0,1}) {
            auto a=layer(s,trans,mode);Matrix<D> x(values(s.ci*s.h*s.w,s.n,.2f));
            auto g=values(s.co*oh(s,trans)*ow(s,trans),s.n,.7f);
            std::list<Layer<D>*> net={a.get()};freeze(net);
            for(int i=0;i<20;++i) {a->eval(x);a->backward_update(x,Matrix<D>(g));}device_sync();
            std::vector<double> samples;
            for(int round=0;round<5;++round) {
                auto begin=std::chrono::steady_clock::now();
                for(int i=0;i<40;++i) {a->eval(x);a->backward_update(x,Matrix<D>(g));}device_sync();
                samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/40);
            }
            std::sort(samples.begin(),samples.end());
            std::cout<<"BENCH "<<trans<<' '<<size<<' '<<mode<<' '<<std::setprecision(9)<<samples[2]<<" ms\n";
        }
    }
}
int compute() {
    try {
#ifdef CUDA
        GPUSampler sampler(42);
#endif
        if(const char* p=std::getenv("CONV_TEST_PHASE")) {
            require(std::string(p)=="prepare"||std::string(p)=="resume","invalid restart phase");
            const char* out=std::getenv("CONV_TEST_OUTPUT");require(out&&*out,"missing checkpoint output");
            restart(p,out);return 0;
        }
        if(std::getenv("CONV_TEST_BENCH")) {bench();return 0;}
        for(const auto& s:std::vector<Shape>{{2,2,5,7,3,3,3,1,1,false},{2,3,8,10,4,4,4,1,2,true},
             {1,4,7,9,2,1,1,0,1,false},{3,2,9,7,5,3,2,0,2,true},
             {2,8,16,17,8,3,3,1,1,true},{1,1,4,5,1,2,3,0,1,false}})
            for(bool trans:{false,true}) fixture(s,trans);
        std::cout<<"PASS deterministic convolution: 12 shapes, scalar gradients, JVP, 10 bitwise replays, default compatibility\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
#endif
