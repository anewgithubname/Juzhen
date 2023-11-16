//
// Created by songa on 16/11/2023.
//

#ifndef ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP
#define ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP

#include "../cpp/juzhen.hpp"


namespace Juzhen
{
    template <class D>
    class DataLoader {
        std::string folder;
        std::string split;

        size_t batch_size;
        size_t n; 
        size_t d;

        FILE *fp_input;
        FILE *fp_output;

    public:
        DataLoader(std::string folder, std::string split, size_t batch_size): folder(folder), split(split), batch_size(batch_size) {
            std::string input_file = folder + "/" + split + "_x.txt";
            std::string output_file = folder + "/" + split + "_y.txt";
            fp_input = fopen(input_file.c_str(), "r");
            fp_output = fopen(output_file.c_str(), "r");

            if (fp_input == NULL || fp_output == NULL) {
                std::cout << "Error opening file" << std::endl;
                exit(1);
            }

            d = getw(fp_input);
            n = getw(fp_input);
        }

        ~DataLoader() {
            fclose(fp_input);
            fclose(fp_output);
        }

        std::tuple<Matrix<D>, Matrix<D>> next_batch() {
            Matrix<D> x(batch_size, d);
            Matrix<D> y(batch_size, 1);

            // determine how many bytes to read given the batch size and the remaining data
            size_t bytes_to_read = sizeof(D) * d * batch_size;
            // check how many 

            // fread(x.data(), sizeof(D), d * batch_size, fp_input);
            // fread(y.data(), sizeof(D), batch_size, fp_output);

            // return {x, y};
        }

    };
}
#endif //ANIMATED_OCTO_SNIFFLE_DATALOADER_HPP
