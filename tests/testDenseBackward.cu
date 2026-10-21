/** Regression: input gradients must use the weights of the forward pass. */
#include "../ml/layer.hpp"
#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <type_traits>

using namespace Juzhen;

template<class D> static Matrix<float> host(const Matrix<D>& x) {
    if constexpr (std::is_same_v<D, float>) return x;
    else return x.to_host();
}

static bool close(const Matrix<float>& actual, const Matrix<float>& expected,
                  const std::string& label, float tolerance = 5e-6f) {
    if (actual.num_row() != expected.num_row() || actual.num_col() != expected.num_col()) return false;
    float maximum = 0;
    for (size_t c = 0; c < actual.num_col(); ++c)
        for (size_t r = 0; r < actual.num_row(); ++r) {
            const float e = std::fabs(actual.elem(r,c) - expected.elem(r,c));
            if (!std::isfinite(e)) return false;
            maximum = std::max(maximum, e);
        }
    std::cout << label << " max_abs=" << maximum << '\n';
    return maximum <= tolerance;
}

template<class D, template<class> class Dense>
static bool check(const std::string& name, int activation) {
    // Mixed signs, non-square matrices and a large update expose the old bug.
    const Matrix<float> w("w", {{.7f,-.4f,.2f},{-.3f,.6f,.5f}});
    const Matrix<float> b("b", {{.13f},{-.21f}});
    const Matrix<float> x("x", {{.8f,-.5f,.4f,.9f},{-.2f,.7f,.5f,-.6f},{.3f,.1f,-.8f,.2f}});
    const Matrix<float> upstream("upstream", {{.5f,-.3f,.7f,.2f},{-.6f,.4f,.1f,-.8f}});
    Matrix<float> t(upstream);
    for (size_t c=0;c<x.num_col();++c) for (size_t r=0;r<w.num_row();++r) {
        double pre = b.elem(r,0);
        for (size_t k=0;k<w.num_col();++k) pre += double(w.elem(r,k))*x.elem(k,c);
        const double derivative = activation==0 ? 1.0 : activation==1 ?
            1.0-std::pow(std::tanh(pre),2) : pre>0 ? 1.0 : 0.0;
        t.elem(r,c) *= float(derivative);
    }
    const auto expected_dx=w.T()*t, expected_dw=t*x.T(), expected_db=sum(t,1);
    Dense<D> layer(2,3,4);
    layer.W()=Matrix<D>(w); layer.b()=Matrix<D>(b);
    layer.adamWstate().alpha=.08f; layer.adambstate().alpha=.08f;
    Matrix<D> device_x(x);
    layer.eval(device_x);
    auto dx=host(layer.backward_update(device_x,Matrix<D>(upstream)));
    bool ok=close(dx,expected_dx,name+" training input gradient");
    // Verify the update really happened, rather than masking the bug by freezing.
    const auto changed=host(layer.W()); float movement=0;
    for(size_t c=0;c<w.num_col();++c) for(size_t r=0;r<w.num_row();++r)
        movement=std::max(movement,std::fabs(changed.elem(r,c)-w.elem(r,c)));
    ok = movement>.05f && layer.adamWstate().iteration==2 && ok;
    ok = close(host(layer.adamWstate().m)/(1-layer.adamWstate().beta1),expected_dw,name+" weight gradient") && ok;
    ok = close(host(layer.adambstate().m)/(1-layer.adambstate().beta1),expected_db,name+" bias gradient") && ok;

    // Frozen backward must preserve both parameter values and optimizer state.
    layer.W()=Matrix<D>(w); layer.b()=Matrix<D>(b);
    std::list<Layer<D>*> net={&layer}; freeze(net);
    const auto before_m=host(layer.adamWstate().m),before_v=host(layer.adamWstate().v);
    const int before_iter=layer.adamWstate().iteration;
    layer.eval(device_x);
    auto frozen_dx=host(layer.backward_update(device_x,Matrix<D>(upstream)));
    ok = close(frozen_dx,expected_dx,name+" frozen input gradient") && ok;
    ok = close(host(layer.W()),w,name+" frozen weights",0) && ok;
    ok = close(host(layer.b()),b,name+" frozen bias",0) && ok;
    ok = close(host(layer.adamWstate().m),before_m,name+" frozen moment",0) && ok;
    ok = close(host(layer.adamWstate().v),before_v,name+" frozen variance",0) && ok;
    return layer.adamWstate().iteration==before_iter && ok;
}

template<class D> static bool suite(const std::string& backend) {
    bool ok=check<D,LinearLayer>(backend+" linear",0);
    ok=check<D,Layer>(backend+" tanh",1) && ok;
    ok=check<D,ReluLayer>(backend+" relu",2) && ok;
    return ok;
}

int compute() {
    global_rand_gen.seed(17);
    bool ok=suite<float>("CPU");
#ifdef CUDA
    GPUSampler sampler(17);
    ok=suite<CUDAfloat>("CUDA") && ok;
#elif defined(ROCM_HIP)
    ok=suite<ROCMfloat>("ROCm") && ok;
#elif defined(APPLE_SILICON)
    ok=suite<MPSfloat>("MPS") && ok;
#endif
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " dense backward\n";
    return ok ? 0 : 1;
}
