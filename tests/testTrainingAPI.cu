/** Core training API: finite differences, accumulation, Adam, clipping and normalization. */
#include "../ml/layer.hpp"
#include "../ml/optimizer.hpp"
#include <iostream>
#include <limits>
using namespace Juzhen;
// An unregistered custom subclass must remain an explicit opt-in.
template<class D> struct UnadaptedLayer : Layer<D> { using Layer<D>::Layer; };

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
template<class D> void ordinary_loss_checks(bool use_sequential) {
    constexpr int batch=3;
    struct CountedLinear : LinearLayer<D> {
        using LinearLayer<D>::LinearLayer;
        int calls=0;
        bool fail=false;
        void eval(const Matrix<D>& x) override {
            ++calls;
            if(fail)throw std::logic_error("injected eval failure");
            LinearLayer<D>::eval(x);
        }
    };
    CountedLinear hidden(2,2,batch),readout(2,2,batch);
    Matrix<float> xh("input",2,batch),yh("target",2,batch);
    Matrix<float> w1("w1",2,2),b1("b1",2,1),w2("w2",2,2),b2("b2",2,1);
    int counter=0;
    for(auto* matrix:{&xh,&yh,&w1,&b1,&w2,&b2})
        for(size_t j=0;j<matrix->num_col();++j)for(size_t i=0;i<matrix->num_row();++i)
            matrix->elem(i,j)=.5f*std::sin(float(++counter));
    hidden.W()=Matrix<D>(w1);hidden.b()=Matrix<D>(b1);
    readout.W()=Matrix<D>(w2);readout.b()=Matrix<D>(b2);
    Matrix<D> x(xh),y(yh);
    LossLayer<D> loss(batch,y);
    std::list<Layer<D>*> net{&loss,&readout,&hidden};
    Sequential<D> model(net);
    AdamWOptions settings;settings.weight_decay=0;
    AdamW<D> optimizer(model.parameters(),settings);
    require(optimizer.parameters().size()==4,"loss registers no trainable parameters");
    require(loss.checkpoint_parameters().empty()&&loss.checkpoint_optimizers().empty(),"loss has no checkpoint parameter state");
    static_assert(!std::is_constructible_v<LossLayer<D>,int,const Matrix<D>&,
                  const std::list<Layer<D>*>&,const Matrix<D>&>);
    static_assert(!std::is_constructible_v<LossLayer<D>,int,const Matrix<D>&,
                  Sequential<D>&,const Matrix<D>&>);
    static_assert(std::is_same_v<decltype(model.eval(x)),const Matrix<D>&>);
    rejects([&]{model.backward();},"Sequential backward needs eval");
    rejects([&]{model.value();},"Sequential value needs eval");
    auto eval=[&]() -> const Matrix<D>& {
        return use_sequential?model.eval(x):Juzhen::forward(net,x);
    };
    auto backward=[&] {
        return use_sequential?model.backward():Juzhen::backward(net,x,Matrix<D>::ones(1,1));
    };
    auto reference=[&] {
        double result=0;
        for(int j=0;j<batch;++j)for(int o=0;o<2;++o) {
            double prediction=b2.elem(o,0);
            for(int h=0;h<2;++h) {
                double activation=b1.elem(h,0);
                for(int i=0;i<2;++i)activation+=double(w1.elem(h,i))*xh.elem(i,j);
                prediction+=double(w2.elem(o,h))*activation;
            }
            const double error=prediction-yh.elem(o,j);result+=error*error/batch;
        }
        return result;
    };
    optimizer.zero_grad();eval();
    require(hidden.calls==1&&readout.calls==1,"network evaluates every layer once");
    if(use_sequential)close(host(model.value()),host(loss.value()),"Sequential output is loss",0);
    require(std::abs(host(loss.value()).elem(0,0)-reference())<3e-6,"ordinary loss scalar reference");
    auto dx=host(backward());
    require(hidden.calls==1&&readout.calls==1,"backward does not rerun eval");
    // Independent scalar FP64 finite differences check both layers and input.
    std::vector<Matrix<float>*> tensors{&w1,&b1,&w2,&b2,&xh};
    const auto& refs=optimizer.parameters();
    for(size_t k=0;k<tensors.size();++k) {
        auto& matrix=*tensors[k];Matrix<float> expected("finite differences",matrix.num_row(),matrix.num_col());
        for(size_t j=0;j<matrix.num_col();++j)for(size_t i=0;i<matrix.num_row();++i) {
            const float original=matrix.elem(i,j);
            matrix.elem(i,j)=original+1e-3f;const double plus=reference();const float upper=matrix.elem(i,j);
            matrix.elem(i,j)=original-1e-3f;const double minus=reference();const float lower=matrix.elem(i,j);
            matrix.elem(i,j)=original;expected.elem(i,j)=float((plus-minus)/(double(upper)-lower));
        }
        close(k<refs.size()?host(refs[k].grad()):dx,expected,"ordinary loss finite differences",2e-5f);
        if(k<refs.size()) {
            close(host(*refs[k].value),matrix,"backward preserves weights",0);
            require(refs[k].state->iteration==1,"backward preserves Adam state");
        }
    }
    std::vector<Matrix<float>> first_gradients;
    for(const auto& p:refs)first_gradients.push_back(host(p.grad()));
    backward();
    for(size_t k=0;k<refs.size();++k)close(host(refs[k].grad()),first_gradients[k]*2,"network accumulates gradients");
    optimizer.zero_grad();
    auto seed=Matrix<D>::ones(1,1)*-.25f;
    auto scaled_dx=use_sequential?model.backward(std::move(seed)):Juzhen::backward(net,x,std::move(seed));
    close(host(scaled_dx),dx*-.25f,"scalar upstream scales input gradient");
    for(size_t k=0;k<refs.size();++k)close(host(refs[k].grad()),first_gradients[k]*-.25f,"scalar upstream scales parameter gradient");
    optimizer.zero_grad();backward();
    // The original list-based updating driver still agrees with separate Adam.
    LinearLayer<D> old_hidden(2,2,batch),old_readout(2,2,batch);
    old_hidden.W()=Matrix<D>(w1);old_hidden.b()=Matrix<D>(b1);
    old_readout.W()=Matrix<D>(w2);old_readout.b()=Matrix<D>(b2);
    LossLayer<D> old_loss(batch,y);
    std::list<Layer<D>*> old_net{&old_loss,&old_readout,&old_hidden};
    Juzhen::forward(old_net,x);
    close(host(Juzhen::backward_update(old_net,x)),dx,"updating network input gradient");
    require(optimizer.step()==4,"Adam updates only trainable layers");
    const Matrix<D>* updated[]={&old_hidden.W(),&old_hidden.b(),&old_readout.W(),&old_readout.b()};
    for(size_t k=0;k<refs.size();++k)close(host(*refs[k].value),host(*updated[k]),"ordinary loss legacy Adam parity",2e-6f);
    LossLayer<D> legacy(batch,y);legacy.eval(readout.value());
    close(host(legacy.value()),host(loss.value()),"original LossLayer eval",0);
    rejects([&]{loss.backward(readout.value(),Matrix<D>::ones(1,2));},"loss requires a scalar upstream gradient");
    UnadaptedLayer<D> unsupported(2,2,batch);std::list<Layer<D>*> bad{&loss,&readout,&unsupported};
    Sequential<D> bad_model(bad);optimizer.zero_grad();bad_model.eval(x);
    rejects([&]{bad_model.backward();},"network rejects unsupported layer");
    for(const auto& p:refs)require(!p.has_grad(),"unsupported network leaves no partial gradients");
    Sequential<D> prediction{&readout,&hidden};prediction.eval(x);
    rejects([&]{prediction.backward();},"non-scalar network needs an explicit upstream gradient");
    LinearLayer<D> scalar(1,2,1);scalar.W()=Matrix<D>::ones(1,2);
    auto one_input=x.columns(0,1);Sequential<D> scalar_model{&scalar};scalar_model.eval(one_input);
    close(host(scalar_model.backward()),Matrix<float>::ones(2,1),"scalar seed is independent of loss layer type",0);
    model.eval(x);hidden.fail=true;
    rejects([&]{model.eval(x);},"Sequential propagates failed eval");
    rejects([&]{model.backward();},"failed eval invalidates Sequential context");
    rejects([&]{Sequential<D> empty{std::list<Layer<D>*>{}};},"Sequential rejects empty network");
    rejects([&]{Sequential<D> invalid{nullptr};},"Sequential rejects null layer");
    rejects([&]{Sequential<D> reused{&hidden,&hidden};},"Sequential rejects repeated layer instance");
}

template<class D> void compatibility_and_errors() {
    LinearLayer<D> legacy(3,2,4),explicit_layer(3,2,4);
    explicit_layer.W()=Matrix<D>(legacy.W());explicit_layer.b()=Matrix<D>(legacy.b());
    auto x=Matrix<D>::randn(2,4),up=Matrix<D>::randn(3,4);AdamWOptions settings;settings.weight_decay=0;
    AdamW<D> opt(explicit_layer.parameters(),settings);
    for(int k=0;k<3;++k) {
        legacy.eval(x);explicit_layer.eval(x);opt.zero_grad();
        auto expected=legacy.backward_update(x,Matrix<D>(up));auto actual=explicit_layer.backward(x,Matrix<D>(up));
        close(host(actual),host(expected),"legacy vs deferred input gradient");opt.step();
        close(host(legacy.W()),host(explicit_layer.W()),"legacy vs deferred Adam",2e-6f);
    }
    opt.zero_grad();std::list<Layer<D>*> net{&explicit_layer};freeze(net);
    auto before=host(explicit_layer.W());const int iteration=explicit_layer.adamWstate().iteration;
    explicit_layer.eval(x);explicit_layer.backward(x,Matrix<D>(up));require(opt.step()==0,"frozen parameters skipped");
    close(host(explicit_layer.W()),before,"frozen weights",0);require(explicit_layer.adamWstate().iteration==iteration,"frozen Adam state");unfreeze(net);
    UnadaptedLayer<D> unsupported(3,2,4);std::list<Layer<D>*> bad{&explicit_layer,&unsupported};
    rejects([&]{Juzhen::backward(bad,x,Matrix<D>(up));},"unsupported layer rejected before partial backward");
    require(!opt.parameters()[0].has_grad(),"unsupported network left no partial gradients");
    rejects([&]{Juzhen::parameters(bad);},"unsupported parameter traversal");
    explicit_layer.eval(x);explicit_layer.backward(x,Matrix<D>(up));
    auto r=explicit_layer.parameters();GradientBuffer<D> different;auto wrong=r[0];wrong.gradient=&different;
    rejects([&]{AdamW<D> inconsistent({r[0],wrong});},"inconsistent shared registration");
    auto invalid=settings;invalid.beta1=1;rejects([&]{AdamW<D> badopt(r,invalid);},"invalid optimizer hyperparameters");
    // Validation covers all parameters before any are updated.
    const auto old_weight=host(explicit_layer.W());r[1].gradient->value=Matrix<D>::zeros(2,1);
    rejects([&]{opt.step();},"optimizer gradient shape mismatch");close(host(explicit_layer.W()),old_weight,"failed step atomic validation",0);
    opt.zero_grad();explicit_layer.backward(x,Matrix<D>(up));auto nan=host(r[0].grad());nan.elem(0,0)=std::numeric_limits<float>::quiet_NaN();r[0].gradient->value=Matrix<D>(nan);
    bool caught=false;try{opt.clip_grad_norm(1);}catch(const std::runtime_error&){caught=true;}require(caught,"nonfinite norm rejected");close(host(explicit_layer.W()),old_weight,"nonfinite norm leaves weights unchanged",0);
    opt.zero_grad();require(opt.step()==0,"zero_grad marks gradients absent");
    // Standalone LayerNorm also exposes a pure backward path.
    TransformerLayer<D> uninitialized(2,2,4,4,1,1,true);
    rejects([&]{uninitialized.backward(x,Matrix<D>(x));},"Transformer needs eval");
    rejects([&]{uninitialized.eval(Matrix<D>("bad",2,3));},"Transformer forward shape");
    LayerNormLayer<D> norm(2,4);
    rejects([&]{norm.backward(Matrix<D>(x));},"LayerNorm needs eval");
    rejects([&]{norm.forward(Matrix<D>("bad",2,3));},"LayerNorm forward shape");
    auto norm_refs=norm.parameters();AdamW<D> norm_opt(norm_refs,settings);
    norm.forward(x);auto saved=host(norm.gamma);norm.backward(Matrix<D>(x));close(host(norm.gamma),saved,"layernorm deferred update",0);
    require(norm_opt.step()==2,"standalone LayerNorm gradients");
    norm_opt.zero_grad();norm.trainable=false;norm.forward(x);norm.backward(Matrix<D>(x));require(norm_opt.step()==0,"frozen LayerNorm");
}

// Exercise the portable device formulation directly on every available
// backend, as well as the production optimizer dispatch. All 16 independent
// weight/gradient/moment storage layouts represent the same non-square shape.
template<class D> void optimizer_math_checks() {
    for(int layout=0;layout<16;++layout)for(bool decay:{false,true}) {
        Matrix<float> expected_w("weight reference",3,2),expected_m("first moment",3,2),expected_v("second moment",3,2);
        for(size_t j=0;j<2;++j)for(size_t i=0;i<3;++i) {
            expected_w.elem(i,j)=float(int(i+3*j)-2)*.2f;
            expected_m.elem(i,j)=float(int(i+3*j)-3)*.025f;
            expected_v.elem(i,j)=float(i+3*j+1)*.017f;
        }
        auto stored=[&](const Matrix<float>& value,int bit) {
            return (layout&bit)?Matrix<D>(transpose_storage(value)).T():Matrix<D>(value);
        };
        auto w=stored(expected_w,1),portable_w=stored(expected_w,1);
        adam_state<D> state(w),portable_state(w);GradientBuffer<D> gradient;bool trainable=true;
        AdamWOptions settings{.0125f,.7f,.93f,.02f,.2f};
        AdamW<D> opt({{"fixture",&w,&gradient,&state,&trainable,decay}},settings);
        state.iteration=7;state.m=stored(expected_m,4);state.v=stored(expected_v,8);
        portable_state=state;
        for(int step=0;step<4;++step) {
            const float lr=step==1?0.f:settings.lr;opt.set_lr(lr);portable_state.alpha=lr;
            Matrix<float> g("gradient",3,2);
            for(size_t j=0;j<2;++j)for(size_t i=0;i<3;++i)
                g.elem(i,j)=step==3?0.f:float(int((i+3*j+step*2)%7)-3)*.13f;
            gradient.value=stored(g,2);gradient.ready=true;
            const int iteration=state.iteration;
            const double bc1=1/(1-std::pow(double(settings.beta1),iteration));
            const double bc2=1/(1-std::pow(double(settings.beta2),iteration));
            for(size_t j=0;j<2;++j)for(size_t i=0;i<3;++i) {
                const double gi=g.elem(i,j);
                const double m=settings.beta1*double(expected_m.elem(i,j))+(1-settings.beta1)*gi;
                const double v=settings.beta2*double(expected_v.elem(i,j))+(1-settings.beta2)*gi*gi;
                expected_w.elem(i,j)=float(expected_w.elem(i,j)*(1-lr*(decay?settings.weight_decay:0))-
                    lr*(m*bc1)/(std::sqrt(v*bc2)+settings.eps));
                expected_m.elem(i,j)=float(m);expected_v.elem(i,j)=float(v);
            }
            const float portable_bc1=1.f/(1.f-std::pow(settings.beta1,float(iteration)));
            const float portable_bc2=1.f/(1.f-std::pow(settings.beta2,float(iteration)));
            optimizer_detail::adamw_matrix_update(portable_w,gradient.value,portable_state,
                portable_bc1,portable_bc2,decay?settings.weight_decay:0);
            ++portable_state.iteration;
            require(opt.step()==1,"AdamW updates registered parameter once");
            const std::string label="AdamW layout="+std::to_string(layout)+" decay="+std::to_string(decay)+" step="+std::to_string(step);
            close(host(w),expected_w,label+" production weight",2e-6f);
            close(host(portable_w),expected_w,label+" portable weight",2e-6f);
            for(auto* a:{&state,&portable_state}) {
                close(host(a->m),expected_m,label+" first moment",2e-6f);
                close(host(a->v),expected_v,label+" second moment",2e-6f);
                require(a->iteration==iteration+1,label+" iteration");
            }
            close(host(gradient.value),g,label+" preserves accumulated gradient",0);
            require(gradient.ready&&gradient.value.get_transpose()==bool(layout&2),label+" preserves gradient metadata");
        }
        // Both freeze flags and missing gradients suppress moments, decay and
        // iteration changes, even when old gradients and moments are still stored.
        bool parent_trainable=true;
        auto registration=opt.parameters()[0];registration.parent_trainable=&parent_trainable;
        AdamW<D> guarded({registration},settings);
        const auto saved_w=host(w),saved_m=host(state.m),saved_v=host(state.v);
        const int saved_iteration=state.iteration;
        for(int mode=0;mode<3;++mode) {
            trainable=mode!=0;parent_trainable=mode!=1;gradient.ready=mode!=2;
            require(guarded.step()==0,"AdamW frozen/absent parameter skipped");
            close(host(w),saved_w,"skipped AdamW weight",0);
            close(host(state.m),saved_m,"skipped AdamW first moment",0);
            close(host(state.v),saved_v,"skipped AdamW second moment",0);
            require(state.iteration==saved_iteration,"skipped AdamW iteration");
        }
        trainable=true;parent_trainable=true;gradient.ready=true;
        // Full validation precedes updates, including on the portable backends.
        auto bad_w=Matrix<D>::ones(2,2);adam_state<D> bad_state(bad_w);GradientBuffer<D> bad_gradient;
        bad_gradient.value=Matrix<D>::ones(3,2);bad_gradient.ready=true;
        AdamW<D> invalid({registration,{"invalid",&bad_w,&bad_gradient,&bad_state,nullptr}},settings);
        rejects([&]{invalid.step();},"AdamW rejects later malformed gradient");
        close(host(w),saved_w,"invalid AdamW preserves earlier weight",0);
        close(host(state.m),saved_m,"invalid AdamW preserves earlier moment",0);
        require(state.iteration==saved_iteration,"invalid AdamW preserves earlier iteration");
    }
}

// Scalar references exercise reduction tails, zero/absent/frozen groups,
// scratch growth, copied optimizers and pointer replacement after a reload.
template<class D> void clipping_checks(){
    std::vector<std::unique_ptr<LinearLayer<D>>> layers;
    std::vector<ParameterRef<D>> refs;
    for(int count:{1,255,256,257,4095,4096,4097,17003}){
        layers.push_back(std::make_unique<LinearLayer<D>>(count,1,1));
        for(auto p:layers.back()->parameters())refs.push_back(p);
    }
    AdamW<D> opt(refs);require(opt.clip_grad_norm(1)==0,"empty gradient norm");
    auto populate=[&](int active){
        opt.zero_grad();double squares=0;
        for(int k=0;k<active;++k){auto p=refs[k];Matrix<float> g("clip fixture",p.value->num_row(),p.value->num_col());
            for(size_t j=0;j<g.num_col();++j)for(size_t i=0;i<g.num_row();++i){float x=float(int((i*17+j*3+k)%31)-15)*.0625f;g.elem(i,j)=x;squares+=double(x)*x;}
            p.gradient->value=Matrix<D>(g);p.gradient->ready=true;
        }return std::sqrt(squares);
    };
    for(int active:{1,int(refs.size()),3}){
        double expected=populate(active);std::vector<Matrix<float>> before;
        for(int i=0;i<active;++i)before.push_back(host(refs[i].grad()));
        double norm=opt.clip_grad_norm(2.);require(std::abs(norm-expected)<=1e-5*std::max(1.,expected),"global norm vs scalar double reference");
        float scale=float(std::min(1.,2./(expected+1e-6)));
        for(int i=0;i<active;++i)close(host(refs[i].grad()),before[i]*scale,"global gradient scale",2e-6f);
    }
    populate(int(refs.size()));auto copy=opt;require(copy.clip_grad_norm(1)>0,"copied optimizer scratch");
    opt.zero_grad();refs.back().gradient->value=Matrix<D>::zeros(refs.back().value->num_row(),1);refs.back().gradient->ready=true;
    require(opt.clip_grad_norm(0)==0,"zero gradients clip at zero");
    populate(3);auto nan=host(refs[2].grad());nan.elem(0,0)=std::numeric_limits<float>::infinity();refs[2].gradient->value=Matrix<D>(nan);
    auto first=host(refs[0].grad());bool caught=false;try{opt.clip_grad_norm(0);}catch(const std::runtime_error&){caught=true;}
    require(caught,"nonfinite norm before any scaling");close(host(refs[0].grad()),first,"nonfinite norm preserves other gradients",0);
    *refs[2].trainable=false;require(std::isfinite(opt.clip_grad_norm(1)),"frozen nonfinite gradient excluded");*refs[2].trainable=true;
    // Non-square, genuinely transposed gradient storage.
    LinearLayer<D> transposed(3,7,1);AdamW<D> transpose_opt(transposed.parameters());auto p=transposed.parameters()[0];
    Matrix<float> g("g",3,7);for(size_t j=0;j<7;++j)for(size_t i=0;i<3;++i)g.elem(i,j)=float(i+3*j+1)/32;
    Matrix<D> backing(transpose_storage(g));p.gradient->value=backing.T();p.gradient->ready=true;
    double expected=0;for(size_t j=0;j<7;++j)for(size_t i=0;i<3;++i)expected=std::hypot(expected,double(g.elem(i,j)));
    double norm=transpose_opt.clip_grad_norm(.5);require(std::abs(norm-expected)<1e-6,"transposed gradient norm");
    close(host(p.grad()),g*float(.5/(expected+1e-6)),"transposed gradient scale",2e-6f);
#ifdef CUDA
    if constexpr(std::is_same_v<D,CUDAfloat>){
        // FP64 partials avoid squaring FP32 inputs into overflow/underflow.
        for(float magnitude:{1e20f,1e-30f}){
            Matrix<float> extreme("extreme",3,7);for(size_t j=0;j<7;++j)for(size_t i=0;i<3;++i)extreme.elem(i,j)=magnitude;
            p.gradient->value=Matrix<D>(extreme);p.gradient->ready=true;
            double expected=double(magnitude)*std::sqrt(21.);double norm=transpose_opt.clip_grad_norm(expected*2);
            require(std::abs(norm/expected-1)<1e-12,"stable extreme gradient norm");
        }
        // Distinct registrations sharing one gradient keep sequential semantics.
        LinearLayer<D> a(3,7,1),b(3,7,1);auto ar=a.parameters()[0],br=b.parameters()[0];br.gradient=ar.gradient;
        ar.gradient->value=Matrix<D>(g);ar.gradient->ready=true;AdamW<D> alias_opt({ar,br});
        double norm=alias_opt.clip_grad_norm(.5);float scale=float(.5/(norm+1e-6));
        close(host(ar.grad()),g*scale*scale,"aliased gradient sequential fallback",2e-6f);
    }
#endif
}


// Independent FP64 LayerNorm reference covers both the CUDA warp path and
// narrow/transposed fallbacks. Tiny variance also exercises cancellation.
template<class D> void normalization_checks() {
    auto check_reference=[](const Matrix<float>& actual,const Matrix<float>& expected,const std::string& label){
        double error2=0,reference2=0,max_error=0;
        for(size_t j=0;j<actual.num_col();++j)for(size_t i=0;i<actual.num_row();++i){
            double e=double(actual.elem(i,j))-expected.elem(i,j);
            require(std::isfinite(e),label+" nonfinite");error2+=e*e;
            reference2+=double(expected.elem(i,j))*expected.elem(i,j);max_error=std::max(max_error,std::abs(e));
        }
        require(max_error<3e-5 || error2<=4e-8*reference2,label+" FP64 reference max_abs="+std::to_string(max_error));
    };
    for(int d:{1,7,16,31,32,33,127,128,129,255,256,257,1024,4097})for(int pattern=0;pattern<3;++pattern){
        const int n=d==33?129:(d==256?1:5);
        Matrix<float> x("x",d,n),up("up",d,n),g("gamma",d,1),b("beta",d,1);
        Matrix<float> y("reference y",d,n),dx("reference dx",d,n),db=Matrix<float>::zeros(d,1);
        std::vector<double> bd(d);
        for(int i=0;i<d;++i){g.elem(i,0)=.3f+std::sin(.37f*i);b.elem(i,0)=std::cos(.23f*i);
            for(int c=0;c<n;++c){float z=std::sin(.17f*i+.31f*c);x.elem(i,c)=pattern==0?z:(pattern==1?1.f:1.f+1e-4f*z);up.elem(i,c)=std::cos(.13f*i-.29f*c);}}
        for(int c=0;c<n;++c){
            double mean=0,var=0,m1=0,m2=0;for(int i=0;i<d;++i)mean+=x.elem(i,c);mean/=d;
            for(int i=0;i<d;++i){double h=double(x.elem(i,c))-mean;var+=h*h;}double inv=1/std::sqrt(var/d+1e-5);
            for(int i=0;i<d;++i){double h=(double(x.elem(i,c))-mean)*inv;double v=double(g.elem(i,0))*up.elem(i,c);m1+=v;m2+=v*h;bd[i]+=up.elem(i,c);y.elem(i,c)=float(g.elem(i,0)*h+b.elem(i,0));}
            for(int i=0;i<d;++i){double h=(double(x.elem(i,c))-mean)*inv;dx.elem(i,c)=float(inv*(double(g.elem(i,0))*up.elem(i,c)-m1/d-h*m2/d));}
        }
        for(int i=0;i<d;++i){db.elem(i,0)=float(bd[i]);}
        for(bool transposed:{false,true}){
            LayerNormLayer<D> layer(d,n);layer.gamma=Matrix<D>(g);layer.beta=Matrix<D>(b);auto refs=layer.parameters();AdamW<D> opt(refs);
            Matrix<D> input=transposed?Matrix<D>(transpose_storage(x)).T():Matrix<D>(x);
            Matrix<D> upstream=transposed?Matrix<D>(transpose_storage(up)).T():Matrix<D>(up);
            const std::string label="LayerNorm dim="+std::to_string(d)+" pattern="+std::to_string(pattern)+" transposed="+std::to_string(transposed);
            check_reference(host(layer.forward(input)),y,label+" forward");auto cache=host(layer.cached_xhat);
            // Parameter gradients differentiate the actual cached normalization.
            // For tiny variance, rounding its mean can dominate a small gamma
            // gradient; check that reduction independently against this cache.
            Matrix<float> cached_dg("cached gamma reference",d,1);
            for(int i=0;i<d;++i){double v=0;for(int c=0;c<n;++c)v+=double(up.elem(i,c))*cache.elem(i,c);cached_dg.elem(i,0)=float(v);}
            check_reference(host(layer.backward(Matrix<D>(upstream))),dx,label+" input gradient");
            check_reference(host(refs[0].grad()),cached_dg,label+" gamma gradient");check_reference(host(refs[1].grad()),db,label+" beta gradient");
            layer.backward(Matrix<D>(upstream));check_reference(host(refs[0].grad()),cached_dg*2,label+" accumulated gamma");check_reference(host(refs[1].grad()),db*2,label+" accumulated beta");
            close(host(layer.gamma),g,label+" deferred gamma",0);close(host(layer.beta),b,label+" deferred beta",0);
            close(host(layer.cached_xhat),cache,label+" preserved cache",0);close(host(input),x,label+" preserved input",0);close(host(upstream),up,label+" preserved upstream",0);
            opt.zero_grad();layer.trainable=false;check_reference(host(layer.backward(Matrix<D>(upstream))),dx,label+" frozen input gradient");require(!refs[0].has_grad()&&!refs[1].has_grad(),label+" frozen parameters");
        }
    }
#ifdef CUDA
    if constexpr(std::is_same_v<D,CUDAfloat>){
        cuda_layernorm_forward(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,32,0);
        cuda_layernorm_backward(nullptr,nullptr,nullptr,nullptr,nullptr,32,0);
        cuda_relu_backward_inplace(nullptr,nullptr,0);
        for(int count:{1,255,256,257,4097}){
            Matrix<float> a("activation",count,1),up("upstream",count,1),expected("expected",count,1);
            for(int i=0;i<count;++i){a.elem(i,0)=float(i%7-3);up.elem(i,0)=float(i%11-5)*.25f;}
            if(count>7){a.elem(0,0)=-0.f;a.elem(1,0)=std::numeric_limits<float>::quiet_NaN();a.elem(2,0)=std::numeric_limits<float>::infinity();a.elem(3,0)=-std::numeric_limits<float>::infinity();up.elem(4,0)=std::numeric_limits<float>::infinity();up.elem(5,0)=std::numeric_limits<float>::quiet_NaN();a.elem(6,0)=-1.f;up.elem(6,0)=std::numeric_limits<float>::infinity();a.elem(7,0)=0.f;up.elem(7,0)=std::numeric_limits<float>::quiet_NaN();}
            for(int i=0;i<count;++i)expected.elem(i,0)=(a.elem(i,0)>0.f?1.f:0.f)*up.elem(i,0);
            Matrix<D> activation(a),gradient(up);cuda_relu_backward_inplace((const float*)activation.data(),(float*)gradient.data(),count);auto actual=host(gradient),saved=host(activation);
            for(int i=0;i<count;++i){float e=expected.elem(i,0),v=actual.elem(i,0);require(std::isnan(e)?std::isnan(v):e==v,"fused ReLU derivative");if(e==0)require(std::signbit(e)==std::signbit(v),"fused ReLU signed zero");require(std::isnan(a.elem(i,0))?std::isnan(saved.elem(i,0)):a.elem(i,0)==saved.elem(i,0),"fused ReLU preserves activation");}
        }
        CudaErrorCheck(cudaDeviceSynchronize());
    }
#endif
}


template<class D> void suite(const char* backend) {
    Matrix<D> empty("empty",0,3);empty.zeros();empty.ones();
    require(empty.num_row()==0&&empty.num_col()==3,"empty tensor fill");
    ordinary_loss_checks<D>(false);
    ordinary_loss_checks<D>(true);
    compatibility_and_errors<D>();
    optimizer_math_checks<D>();
    clipping_checks<D>();
    normalization_checks<D>();
    std::cout << "[PASS] " << backend << " core training API\n";
}
int compute() {
    try {
        global_rand_gen.seed(919);
        suite<float>("CPU");
#ifdef CUDA
        GPUSampler sampler(919);
        suite<CUDAfloat>("CUDA");
#endif
#ifdef APPLE_SILICON
        optimizer_math_checks<MPSfloat>();
        std::cout << "[PASS] Metal AdamW scalar references\n";
#endif
#ifdef ROCM_HIP
        optimizer_math_checks<ROCMfloat>();
        std::cout << "[PASS] ROCm AdamW scalar references\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
