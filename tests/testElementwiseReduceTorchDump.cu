/** Dump elementwise/reduce results for the PyTorch oracle test. */
#include "../cpp/juzhen.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#if defined(CUDA)
using Backend = CUDAfloat;
#elif defined(APPLE_SILICON)
using Backend = MPSfloat;
#elif defined(ROCM_HIP)
using Backend = ROCMfloat;
#else
using Backend = float;
#endif

template<class D> static Matrix<float> host(const Matrix<D>& m) {
    if constexpr(std::is_same_v<D,float>) return m;
    else return m.to_host();
}

static void write_matrix(FILE* f,const Matrix<float>& m) {
    int32_t r=(int32_t)m.num_row(),c=(int32_t)m.num_col();
    fwrite(&r,4,1,f); fwrite(&c,4,1,f);
    for(int32_t j=0;j<c;++j) for(int32_t i=0;i<r;++i) {
        float v=m.elem(i,j); fwrite(&v,4,1,f);
    }
}

int compute() {
    global_rand_gen.seed(123);
#if defined(CUDA)
    GPUSampler sampler(123);
#endif
    Matrix<float> input_h("input",3,4);
    const float values[12]={-3.0f,-1.5f,-0.25f,0.0f,0.5f,1.25f,2.0f,3.5f,4.0f,-2.0f,0.75f,5.0f};
    for(int c=0;c<4;++c) for(int r=0;r<3;++r) input_h.elem(r,c)=values[c*3+r];
    Matrix<Backend> input(input_h);

    auto ew=[] __GPU_CPU__(float x) { return x*x+2.0f*x-0.5f; };
    auto elem_l=elemwise(ew,static_cast<const Matrix<Backend>&>(input));
    auto unchanged=host(input);

    Matrix<Backend> movable(input_h);
    const void* before=(const void*)movable.data();
    auto elem_r=elemwise(ew,std::move(movable));
    const bool rvalue_reused=before==(const void*)elem_r.data();

    auto sum_reduce=[] __GPU_CPU__(float* src,float* dst,int n,int) {
        float sum=0.0f; for(int i=0;i<n;++i) sum+=src[i]; dst[0]=sum;
    };
    auto stats_reduce=[] __GPU_CPU__(float* src,float* dst,int n,int) {
        float sum=0.0f,maximum=src[0];
        for(int i=0;i<n;++i) { sum+=src[i]; maximum=maximum>src[i]?maximum:src[i]; }
        dst[0]=sum; dst[1]=maximum;
    };

    auto reduce_l_dim0=reduce(sum_reduce,static_cast<const Matrix<Backend>&>(input),0,1);
    auto reduce_l_dim1=reduce(sum_reduce,static_cast<const Matrix<Backend>&>(input),1,1);
    // There is currently no Matrix&& overload for reduce. These rvalues bind
    // to the const Matrix& API and must still produce the correct result.
    auto reduce_r_dim0=reduce(stats_reduce,Matrix<Backend>(input_h),0,2);
    auto reduce_r_dim1=reduce(stats_reduce,Matrix<Backend>(input_h),1,2);

    float unchanged_error=0.0f;
    for(int c=0;c<4;++c) for(int r=0;r<3;++r)
        unchanged_error=std::max(unchanged_error,std::fabs(unchanged.elem(r,c)-input_h.elem(r,c)));
    if(unchanged_error!=0.0f || !rvalue_reused) {
        std::cout<<"[FAIL] ownership: lvalue_error="<<unchanged_error
                 <<" rvalue_buffer_reused="<<rvalue_reused<<"\n"; return 1;
    }

    const std::string path=std::string(PROJECT_DIR)+"/res/elementwise_reduce_torch_dump.bin";
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return 1;
    fwrite("JZERDMP1",1,8,f);
    write_matrix(f,input_h); write_matrix(f,host(elem_l)); write_matrix(f,host(elem_r));
    write_matrix(f,host(reduce_l_dim0)); write_matrix(f,host(reduce_l_dim1));
    write_matrix(f,host(reduce_r_dim0)); write_matrix(f,host(reduce_r_dim1)); fclose(f);
    std::cout<<"[PASS] elementwise lvalue preserves input\n"
             <<"[PASS] elementwise rvalue reuses buffer\n"
             <<"[INFO] reduce rvalues bind to const Matrix& (no Matrix&& overload)\n"
             <<"Wrote "<<path<<"\n";
    return 0;
}
