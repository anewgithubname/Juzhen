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
        using File = std::unique_ptr<FILE, matrix_io_detail::FileCloser>;

        File fp_input;
        File fp_output;

        size_t batch_size;
        size_t position = 0;  // index of the next sample to read
        size_t n = 0;         // number of samples
        size_t d = 0;         // input dimension

        // a .matrix file starts with three ints: rows, cols, transpose
        static constexpr long header_bytes = 3 * sizeof(int);

    public:
        /**
         * @param folder     dataset folder
         * @param split      "train" or "test"
         * @param batch_size samples per batch (the last batch may be smaller)
         */
        DataLoader(std::string folder, std::string split, size_t batch_size)
            : batch_size(batch_size) {
            if (!batch_size)
                matrix_io_detail::fail("zero DataLoader batch size");

            fp_input.reset(std::fopen((folder + "/" + split + "_x.matrix").c_str(), "rb"));
            fp_output.reset(std::fopen((folder + "/" + split + "_y.matrix").c_str(), "rb"));
            if (!fp_input || !fp_output)
                matrix_io_detail::fail("cannot open dataset");

            const auto input  = matrix_io_detail::header(fp_input.get());
            const auto output = matrix_io_detail::header(fp_output.get());

            // x is d-by-n, y is 1-by-n, and neither may be stored transposed
            const bool valid_shape = input.rows && input.cols
                                  && output.rows == 1 && output.cols == input.cols
                                  && !input.transpose && !output.transpose;
            if (!valid_shape)
                matrix_io_detail::fail("invalid or mismatched dataset shape");

            d = input.rows;
            n = input.cols;

            // check that the files really contain that much data
            matrix_io_detail::payload<D1>(fp_input.get(), d, n);
            matrix_io_detail::payload<D2>(fp_output.get(), 1, n);
        }

        std::tuple<Matrix<D1>, Matrix<D2>> next_batch() {
            // after a full pass, rewind both files to just past their headers
            if (position == n) {
                if (std::fseek(fp_input.get(), header_bytes, SEEK_SET) != 0 ||
                    std::fseek(fp_output.get(), header_bytes, SEEK_SET) != 0)
                    matrix_io_detail::fail("cannot rewind dataset");
                position = 0;
            }

            const size_t take = std::min(batch_size, n - position);
            Matrix<D1> x("input", d, take);
            Matrix<D2> y("output", 1, take);

            const bool ok =
                std::fread((D1 *)x.data(), sizeof(D1), take * d, fp_input.get()) == take * d &&
                std::fread((D2 *)y.data(), sizeof(D2), take, fp_output.get()) == take;
            if (!ok)
                matrix_io_detail::fail("incomplete dataset batch");

            position += take;
            return {std::move(x), std::move(y)};
        }
    };
}
#endif
