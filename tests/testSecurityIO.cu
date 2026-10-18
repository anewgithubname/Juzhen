/** Regression checks for legacy matrix/CSV and DataLoader file validation. */
#include "../ml/dataloader.hpp"
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif
using namespace Juzhen;
template<class D> bool stream_lifetime(const std::filesystem::path& root){
    const auto path=(root/("lifetime-"+std::to_string(sizeof(D))+".bin")).string();
    Matrix<D> input("source",2,3);
    for(size_t j=0;j<3;++j)for(size_t i=0;i<2;++i)input.elem(i,j)=D(1+i+2*j);
    write(path,input);
    Matrix<D> output("destination",1,1);
    FILE* file=fopen(path.c_str(),"rb");if(!file)return false;
    // Alternate pooled copies and standalone stream buffers in one destination.
    for(int round=0;round<4;++round){
        const auto name=output.get_name();
        std::rewind(file);read(file,output);
        if(output.get_name()!=name||output.num_row()!=2||output.num_col()!=3||output.elem(1,2)!=D(6)){fclose(file);return false;}
        auto copy=output;copy.elem(0,0)=D(99);
        if(output.elem(0,0)!=D(1)){fclose(file);return false;}
        output=std::move(copy); // A pooled buffer will be replaced next time.
    }
    std::rewind(file);read(file,output);
    fclose(file);
    auto moved=std::move(output);
    auto independent=moved;moved.elem(0,0)=D(99);
    return moved.num_row()==2&&moved.num_col()==3&&moved.elem(1,2)==D(6)&&moved.elem(0,0)==D(99)&&independent.elem(0,0)==D(1);
}
int main(){
    Memory<float> mf;Memory<int> mi;Memory<double> md;
    auto raw=getenv("JUZHEN_IO_CASE");auto dir=getenv("JUZHEN_IO_ROOT");if(!raw||!dir)return 2;
    std::string c=raw;auto root=std::filesystem::path(dir);auto path=(root/"matrix.bin").string();
    if(c=="csv"||c=="csv_negative"||c=="csv_missing"){
        auto m=readfromcsv<float>((root/"matrix.csv").string(),c=="csv_negative"?-1:2,2);
        if(c!="csv")return 0;
        return m.num_row()!=2||m.num_col()!=2||m.elem(0,0)!=1||m.elem(0,1)!=2||m.elem(1,0)!=3||m.elem(1,1)!=4;
    }
    if(c.rfind("loader",0)==0){
        DataLoader<float,int> data(root.string(),"train",c=="loader_zero_batch"?0:2);
        if(c!="loader_valid")return 0;
        for(int round=0;round<2;++round){size_t sample=0;
            for(int b=0;b<3;++b){auto [x,y]=data.next_batch();size_t count=b==2?1:2;
                if(x.num_row()!=2||x.num_col()!=count||y.num_col()!=count)return 1;
                for(size_t j=0;j<count;++j){if(x.elem(0,j)!=float(sample+j)||x.elem(1,j)!=float(100+sample+j)||y.elem(0,j)!=int(sample+j))return 1;}
                sample+=count;
            }
        }return 0;
    }
    if(c=="stream_lifetime")return stream_lifetime<float>(root)&&stream_lifetime<double>(root)&&stream_lifetime<int>(root)?0:1;
    if(c=="stream_empty"){
        FILE* f=fopen(path.c_str(),"rb");if(!f)return 2;
        Matrix<float> m("old",2,2);read(f,m);fclose(f);
        return m.num_row()!=0||m.num_col()!=3||m.get_name()!="old";
    }
    if(c=="null_stream"){Matrix<float> m;read<float>(nullptr,m);return 0;}
    if(c=="double_overflow"){auto m=read<double>(path);return 0;}
    if(c=="file_stream"||c=="file_sequence"){
        FILE* f=fopen(path.c_str(),"rb");if(!f)return 2;Matrix<float> m("old",1,1);read(f,m);
        if(c=="file_stream"){fclose(f);return 0;}
        bool ok=m.num_row()==2&&m.num_col()==3&&m.elem(1,2)==6;read(f,m);
        ok=ok&&m.num_row()==3&&m.num_col()==2&&m.get_transpose()&&m.elem(2,1)==6;
        ok=ok&&!feof(f)&&!ferror(f);fclose(f);return ok?0:1;
    }
    if(c=="pipe"){
#ifndef _WIN32
        int fd[2];if(pipe(fd))return 2;FILE* out=fdopen(fd[1],"wb");Matrix<float> a("a",{{1,2},{3,4}});write(out,a);fclose(out);
        FILE* in=fdopen(fd[0],"rb");Matrix<float> b;read(in,b);bool ok=b.num_row()==2&&b.num_col()==2&&b.elem(1,1)==4;fclose(in);return ok?0:1;
#else
        return 77;
#endif
    }
    auto m=read<float>(path);
    if(c=="valid")return m.num_row()!=2||m.num_col()!=3||m.elem(1,2)!=6;
    if(c=="transposed")return m.num_row()!=3||m.num_col()!=2||!m.get_transpose()||m.elem(2,1)!=6;
    if(c=="empty")return m.num_row()!=0||m.num_col()!=3;
    return 0;
}
