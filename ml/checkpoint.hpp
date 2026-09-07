#ifndef JUZHEN_CHECKPOINT_HPP
#define JUZHEN_CHECKPOINT_HPP

#include "layer.hpp"
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Juzhen {

struct TrainingProgress {
    uint64_t epoch = 0;
    uint64_t step = 0;
    uint64_t data_position = 0;
};

// Save/load at optimizer-step boundaries using the same network configuration.
// Host global_rand_gen is serialized; device RNGs and external data-loader or
// scheduler state are not. See docs/checkpoint-resume.md for the resume contract.

namespace checkpoint_detail {
constexpr char magic[8] = {'J','Z','C','K','P','T','0','2'};
constexpr uint32_t version = 2;
constexpr uint32_t footer = 0x4a5a454e; // JZEN

template<class T> bool put(FILE* fp,const T& value) { return fwrite(&value,sizeof(T),1,fp)==1; }
template<class T> bool get(FILE* fp,T& value) { return fread(&value,sizeof(T),1,fp)==1; }

inline bool put_string(FILE* fp,const std::string& value) {
    uint64_t n=value.size(); return put(fp,n) && (!n || fwrite(value.data(),1,n,fp)==n);
}
inline bool get_string(FILE* fp,std::string& value) {
    uint64_t n=0; if(!get(fp,n) || n>(1ull<<24)) return false;
    value.resize((size_t)n); return !n || fread(value.data(),1,(size_t)n,fp)==n;
}

template<class D> bool put_matrix(FILE* fp,const Matrix<D>& matrix) {
    uint64_t rows=matrix.num_row(),cols=matrix.num_col();
    if(!put(fp,rows)||!put(fp,cols)) return false; write(fp,matrix); return !ferror(fp);
}
template<class D> bool get_matrix(FILE* fp,Matrix<D>& matrix) {
    uint64_t rows=0,cols=0; if(!get(fp,rows)||!get(fp,cols)||rows!=matrix.num_row()||cols!=matrix.num_col()) return false;
    // Matrix I/O has its own storage header. Validate it as well, before read()
    // allocates or transfers memory using sizes supplied by the file.
    fpos_t start; int raw_rows=0,raw_cols=0,transposed=0;
    if(fgetpos(fp,&start)!=0 || !get(fp,raw_rows) || !get(fp,raw_cols) || !get(fp,transposed)
       || raw_rows<0 || raw_cols<0 || (transposed!=0 && transposed!=1)
       || uint64_t(transposed?raw_cols:raw_rows)!=rows
       || uint64_t(transposed?raw_rows:raw_cols)!=cols || fsetpos(fp,&start)!=0) return false;
    read(fp,matrix); return !ferror(fp)&&!feof(fp);
}
template<class D> bool put_adam(FILE* fp,const adam_state<D>& s) {
    return put(fp,s.iteration)&&put(fp,s.alpha)&&put(fp,s.beta1)&&put(fp,s.beta2)&&put(fp,s.eps)
        &&put_matrix(fp,s.m)&&put_matrix(fp,s.v);
}
template<class D> bool get_adam(FILE* fp,adam_state<D>& s) {
    return get(fp,s.iteration)&&get(fp,s.alpha)&&get(fp,s.beta1)&&get(fp,s.beta2)&&get(fp,s.eps)
        &&get_matrix(fp,s.m)&&get_matrix(fp,s.v);
}
}

template<class D>
bool save_checkpoint(const std::list<Layer<D>*>& network,const std::string& path,
                     const TrainingProgress& progress,std::string* error=nullptr) try {
    namespace cd=checkpoint_detail;
    if(error) error->clear();
    const std::string temporary=path+".tmp";
    std::unique_ptr<FILE,decltype(&fclose)> file(fopen(temporary.c_str(),"wb"),&fclose);
    FILE* fp=file.get();
    if(!fp) { if(error)*error="cannot open temporary checkpoint"; return false; }
    bool ok=fwrite(cd::magic,1,8,fp)==8 && cd::put(fp,cd::version);
    uint64_t count=network.size(); ok=ok&&cd::put(fp,count)&&cd::put(fp,progress.epoch)
        &&cd::put(fp,progress.step)&&cd::put(fp,progress.data_position);
    std::ostringstream rng; rng<<global_rand_gen; ok=ok&&cd::put_string(fp,rng.str());
    for(auto* layer:network) {
        ok=ok&&cd::put_string(fp,typeid(*layer).name());
        auto parameters=layer->checkpoint_parameters();
        uint64_t parameter_count=parameters.size(); ok=ok&&cd::put(fp,parameter_count);
        for(auto& [name,matrix]:parameters) ok=ok&&cd::put_string(fp,name)&&cd::put_matrix(fp,*matrix);
        auto optimizers=layer->checkpoint_optimizers();
        uint64_t optimizer_count=optimizers.size(); ok=ok&&cd::put(fp,optimizer_count);
        for(auto& [name,state]:optimizers) ok=ok&&cd::put_string(fp,name)&&cd::put_adam(fp,*state);
    }
    ok=ok&&cd::put(fp,cd::footer) && fflush(fp)==0;
    if(fclose(file.release())!=0) ok=false;
    if(ok) {
        std::error_code ec;
#ifdef _WIN32
        if(!MoveFileExW(std::filesystem::path(temporary).c_str(),std::filesystem::path(path).c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            ec=std::error_code(GetLastError(),std::system_category());
#else
        std::filesystem::rename(temporary,path,ec);
#endif
        if(ec) { ok=false; if(error)*error="cannot atomically replace checkpoint: "+ec.message(); }
    }
    if(!ok) { std::error_code ec; std::filesystem::remove(temporary,ec); if(error&&error->empty())*error="checkpoint write failed"; }
    return ok;
} catch(const std::exception& exception) {
    std::error_code ec; std::filesystem::remove(path+".tmp",ec);
    if(error) *error=std::string("checkpoint write failed: ")+exception.what();
    return false;
}

template<class D>
bool load_checkpoint(const std::list<Layer<D>*>& network,const std::string& path,
                     TrainingProgress& progress,std::string* error=nullptr) try {
    namespace cd=checkpoint_detail;
    if(error) error->clear();
    std::unique_ptr<FILE,decltype(&fclose)> file(fopen(path.c_str(),"rb"),&fclose);
    FILE* fp=file.get(); if(!fp) { if(error)*error="cannot open checkpoint"; return false; }
    TrainingProgress pending_progress;
    auto pending_rng=global_rand_gen;
    std::vector<std::pair<Matrix<D>*,Matrix<D>>> pending_parameters;
    std::vector<std::pair<adam_state<D>*,adam_state<D>>> pending_optimizers;
    char magic[8]; uint32_t version=0; uint64_t count=0; std::string rng_state;
    bool ok=fread(magic,1,8,fp)==8 && memcmp(magic,cd::magic,8)==0
        &&cd::get(fp,version)&&version==cd::version&&cd::get(fp,count)&&count==network.size()
        &&cd::get(fp,pending_progress.epoch)&&cd::get(fp,pending_progress.step)&&cd::get(fp,pending_progress.data_position)
        &&cd::get_string(fp,rng_state);
    for(auto* layer:network) {
        std::string type; ok=ok&&cd::get_string(fp,type)&&type==typeid(*layer).name();
        if(!ok) break;
        auto parameters=layer->checkpoint_parameters(); uint64_t parameter_count=0;
        ok=ok&&cd::get(fp,parameter_count)&&parameter_count==parameters.size();
        for(auto& [expected,matrix]:parameters) {
            std::string name; ok=ok&&cd::get_string(fp,name)&&name==expected;
            if(!ok) break;
            pending_parameters.emplace_back(matrix,*matrix);
            ok=cd::get_matrix(fp,pending_parameters.back().second);
        }
        auto optimizers=layer->checkpoint_optimizers(); uint64_t optimizer_count=0;
        ok=ok&&cd::get(fp,optimizer_count)&&optimizer_count==optimizers.size();
        for(auto& [expected,state]:optimizers) {
            std::string name; ok=ok&&cd::get_string(fp,name)&&name==expected;
            if(!ok) break;
            pending_optimizers.emplace_back(state,*state);
            ok=cd::get_adam(fp,pending_optimizers.back().second);
        }
    }
    uint32_t footer=0; ok=ok&&cd::get(fp,footer)&&footer==cd::footer;
    if(ok) { std::istringstream rng(rng_state); rng>>pending_rng; ok=!rng.fail(); }
    if(ok) {
        for(auto& item:pending_parameters) *item.first=std::move(item.second);
        for(auto& item:pending_optimizers) *item.first=std::move(item.second);
        progress=pending_progress;
        global_rand_gen=pending_rng;
    }
    if(!ok&&error)*error="invalid, incompatible, or truncated checkpoint";
    return ok;
} catch(const std::exception& exception) {
    if(error) *error=std::string("checkpoint load failed: ")+exception.what();
    return false;
}
}
#endif
