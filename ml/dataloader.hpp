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
        std::string folder;
        std::string split;

        size_t batch_size;
        size_t batch_idx;

        size_t n; 
        size_t d;

        FILE *fp_input;
        FILE *fp_output;

    public:
        /**
         * @brief Construct a new Data Loader object
         * 
         * @param folder dataset folder
         * @param split train or test
         * @param batch_size batch size
         */
        DataLoader(std::string folder, std::string split, size_t batch_size): 
            folder(folder), split(split), batch_size(batch_size), batch_idx(0) {
        
            std::string input_file = folder + "/" + split + "_x.matrix";
            std::string output_file = folder + "/" + split + "_y.matrix";
            
            fp_input = fopen(input_file.c_str(), "rb");
            fp_output = fopen(output_file.c_str(), "rb");

            if (fp_input == NULL || fp_output == NULL) {
                std::cout << "Error opening file" << std::endl;
                ERROR_OUT;
            }

            d = getw(fp_input); 
            int dout = getw(fp_output);
            n = getw(fp_input); 
            int nout = getw(fp_output);
            
            int trans1 = getw(fp_input);
            int trans2 = getw(fp_output);
            if( trans1 || trans2 ) {
                std::cout << "Dataset matrix cannot be transposed!" << std::endl;
                ERROR_OUT;
            }
        }

        ~DataLoader() {
            fclose(fp_input);
            fclose(fp_output);
        }

        std::tuple<Matrix<D1>, Matrix<D2>> next_batch() {
            // check if at the previous round, we have reached the end of the dataset
            if (batch_idx*batch_size >= n) {
                batch_idx = 0;
                // skip the first 12 bytes
                fseek(fp_input, 12, SEEK_SET);
                fseek(fp_output, 12, SEEK_SET);
            }

            // how many samples to read
            size_t samples_to_read = batch_size;
            if (n - batch_idx*batch_size < batch_size) {
                samples_to_read = n - batch_idx*batch_size;
            }

            Matrix<D1> x("input", d, samples_to_read);
            Matrix<D2> y("output", 1, samples_to_read);

            // read the input
            fread((D1 *)x.data(), sizeof(D1), samples_to_read * d, fp_input);
            fread((D2 *)y.data(), sizeof(D2), samples_to_read * 1, fp_output);

            // update the batch index
            batch_idx++;
            return {std::move(x), std::move(y)};
        }

    };
}
#endif 
