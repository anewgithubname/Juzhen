/**
 * @file dataloader.cu
 * @brief Dataset loader
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2022 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

 */

#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "../cpp/juzhen.hpp"


namespace Juzhen
{
    template <class D1, class D2>
    class DataLoader {
        size_t batch_size,position=0,n=0,d=0;
        FILE *fp_input=nullptr,*fp_output=nullptr;
    public:
        DataLoader(std::string folder,std::string split,size_t batch_size):batch_size(batch_size) {
            if(!batch_size)matrix_io_detail::fail("zero DataLoader batch size");
            fp_input=std::fopen((folder+"/"+split+"_x.matrix").c_str(),"rb");
            fp_output=std::fopen((folder+"/"+split+"_y.matrix").c_str(),"rb");
            if(!fp_input || !fp_output)matrix_io_detail::fail("cannot open dataset");
            const auto input=matrix_io_detail::header(fp_input),output=matrix_io_detail::header(fp_output);
            if(!input.rows || !input.cols || output.rows!=1 || output.cols!=input.cols || input.transpose || output.transpose)
                matrix_io_detail::fail("invalid or mismatched dataset shape");
            d=input.rows;n=input.cols;
            matrix_io_detail::payload<D1>(fp_input,d,n);
            matrix_io_detail::payload<D2>(fp_output,1,n);
        }
        ~DataLoader(){if(fp_input)std::fclose(fp_input);if(fp_output)std::fclose(fp_output);}
        std::tuple<Matrix<D1>,Matrix<D2>> next_batch(){
            if(position==n){
                if(std::fseek(fp_input,3*sizeof(int),SEEK_SET)!=0 || std::fseek(fp_output,3*sizeof(int),SEEK_SET)!=0)
                    matrix_io_detail::fail("cannot rewind dataset");
                position=0;
            }
            const size_t take=std::min(batch_size,n-position);
            Matrix<D1> x("input",d,take);Matrix<D2> y("output",1,take);
            if(std::fread((D1*)x.data(),sizeof(D1),take*d,fp_input)!=take*d ||
               std::fread((D2*)y.data(),sizeof(D2),take,fp_output)!=take)
                matrix_io_detail::fail("incomplete dataset batch");
            position+=take;
            return {std::move(x),std::move(y)};
        }
    };
}
#endif
