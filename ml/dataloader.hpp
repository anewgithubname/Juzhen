//
// Created by songa on 16/11/2023.
//

#ifndef ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP
#define ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP

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
        DataLoader(std::string folder, std::string split, size_t batch_size): 
            folder(folder), split(split), batch_size(batch_size), batch_idx(0) {
        
            std::string input_file = folder + "/" + split + "_x.matrix";
            std::string output_file = folder + "/" + split + "_y.matrix";
            
            fp_input = fopen(input_file.c_str(), "r");
            fp_output = fopen(output_file.c_str(), "r");

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
            // check if we have reached the end of the dataset
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
#endif //ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP
