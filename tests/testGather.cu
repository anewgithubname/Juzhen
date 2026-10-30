/** Independent scalar reference for indexed copying, including transposes. */
#include "../cpp/juzhen.hpp"
#include <limits>
#include <iostream>
#include <stdexcept>
template<class T> Matrix<float> host(const Matrix<T>& m){if constexpr(std::is_same_v<T,float>)return m;else return m.to_host();}
template<class T> bool check(const Matrix<T>& m,const Matrix<float>& h,const idxlist& rows,const idxlist& cols){
    auto result=host(m.slice(rows,cols));if(result.num_row()!=rows.size()||result.num_col()!=cols.size())return false;
    for(size_t j=0;j<cols.size();++j)for(size_t i=0;i<rows.size();++i)if(result.elem(i,j)!=h.elem(rows[i],cols[j]))return false;
    return true;
}
template<class T> bool suite(){
    Matrix<float> h("source",5,7);for(size_t j=0;j<7;++j)for(size_t i=0;i<5;++i)h.elem(i,j)=100*i+j;
    Matrix<T> m(h);idxlist r={4,0,4,2},c={6,1,6,3};
    bool ok=check(m,h,r,c)&&check(m.T(),h.T(),c,r);
    auto rr=host(m.rows(r)),cc=host(m.columns(c));
    for(size_t j=0;j<7;++j)for(size_t i=0;i<r.size();++i)ok=ok&&rr.elem(i,j)==h.elem(r[i],j);
    for(size_t j=0;j<c.size();++j)for(size_t i=0;i<5;++i)ok=ok&&cc.elem(i,j)==h.elem(i,c[j]);
    ok=ok&&check(m,h,{},c)&&check(m,h,r,{})&&check(m,h,{},{});
    ok=ok&&check(m.T(),h.T(),{},r)&&check(m.T(),h.T(),c,{});
    // More than 4096*256 outputs exercises a bounded grid-stride CUDA launch.
    idxlist large_r(1100),large_c(1100);for(size_t i=0;i<1100;++i){large_r[i]=i%5;large_c[i]=(i*3)%7;}
    ok=ok&&check(m,h,large_r,large_c);
    auto copy=m.slice(r,c);copy.zeros();ok=ok&&host(m).elem(4,6)==406;
    // Device backends validate every index before launching the gather.
    if constexpr(!std::is_same_v<T,float>){
        for(auto bad:idxlist{5,std::numeric_limits<size_t>::max()}){bool caught=false;try{m.slice(idxlist{bad},idxlist{0});}catch(const std::out_of_range&){caught=true;}ok=ok&&caught;}
        bool caught=false;try{m.slice(idxlist{0},idxlist{7});}catch(const std::out_of_range&){caught=true;}ok=ok&&caught;
        caught=false;try{m.slice(idxlist{},idxlist{7});}catch(const std::out_of_range&){caught=true;}ok=ok&&caught;
#ifdef CUDA
        ok=ok&&cudaGetLastError()==cudaSuccess;
#endif
    }
    return ok;
}
int compute(){
    bool ok=suite<float>();
#ifdef CUDA
    // A safe logical row violation maps inside the input's physical allocation.
    // This reproduces absent bounds checking without deliberately reading OOB.
    if(getenv("JUZHEN_GATHER_BOUNDS_PROBE")){
        Matrix<float> h("probe",5,7);h.ones();Matrix<CUDAfloat> m(h);bool caught=false;
        try{m.slice(idxlist{5},idxlist{0});}catch(const std::out_of_range&){caught=true;}
        std::cout<<"bounds rejected="<<caught<<'\n';return caught?0:1;
    }
    ok=suite<CUDAfloat>()&&ok;
#endif
#ifdef APPLE_SILICON
    ok=suite<MPSfloat>()&&ok;
#endif
    std::cout<<(ok?"PASS":"FAIL")<<" indexed gather\n";return ok?0:1;
}
