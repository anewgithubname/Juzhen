
/**
 * @file helloworld.cu
 * @brief Hello world example
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

#include "imgui.h"
#include "implot.h"
#include "../cpp/juzhen.hpp"
#include "../ml/layer.hpp"
using namespace std;
using namespace Juzhen;
#include "../ml/plotting.hpp"
#include <mutex>

#ifndef CPU_ONLY
#define FLOAT CUDAfloat
inline Matrix<CUDAfloat> randn(int m, int n) { return Matrix<CUDAfloat>::randn(m, n); }
inline Matrix<CUDAfloat> ones(int m, int n) { return Matrix<CUDAfloat>::ones(m, n); }
inline Matrix<CUDAfloat> vs(std::vector<MatrixView<CUDAfloat>> matrices) { return vstack(matrices); }
inline Matrix<CUDAfloat> hs(std::vector<MatrixView<CUDAfloat>> matrices) { return hstack(matrices); }
inline const float *getdata(const Matrix<CUDAfloat> &m) { return m.to_host().data(); }
#else
#define FLOAT float
inline Matrix<float> randn(int m, int n) { return Matrix<float>::randn(m, n); }
inline Matrix<float> ones(int m, int n) { return Matrix<float>::ones(m, n); }
inline Matrix<float> vs(std::vector<MatrixView<float>> matrices) { return vstack<float>(matrices); }
inline Matrix<float> hs(std::vector<MatrixView<float>> matrices) { return hstack<float>(matrices); }
inline const float *getdata(const Matrix<float> &m) { return m.data(); }
#endif

// for measure time
#include <chrono>

auto sample_X0(int n, int d)
{
    return randn(d, n);
}

auto sample_X1(int n, int d)
{
#ifndef CPU_ONLY
    return hstack({randn(d, n / 2) * .25 - 1, randn(d, n / 2) * .25 + 1});
#else
    return hstack<float>({randn(d, n / 2) * .25 - 1, randn(d, n / 2) * .25 + 1});
#endif
}

class Codeimp
{
    int batchsize = 200;
    int d = 2;
    const int n = 20000;

    float progress = 0.0f;
    float elapsed_time = 0.0f;

    std::vector<float> losses;
    Matrix<float> Z0, Z1, Ztrue;

public:
    Codeimp() : Z0("Z0", 2, 500), Z1("Z1", 2, 500), Ztrue("Zture", 2, 500)
    {
        Z0 = sample_X0(500, d);
        Ztrue = sample_X1(500, d);
    }

    int run()
    {
        //    spdlog::set_level(spdlog::level::debug);
#ifndef CPU_ONLY
        GPUSampler sampler(1);
#endif

        Profiler p("comp");
        using namespace Juzhen;
        std::string base = PROJECT_DIR;

        auto t1 = Clock::now();

        auto X0 = sample_X0(n, d); // reference data
        auto X1 = sample_X1(n, d); // target data

        const size_t numbatches = X0.num_col() / batchsize;

        // create a neural network
        // define layers, out - in - batchsize
        ReluLayer<FLOAT> L0(133, d + 1, batchsize),
            L1(133, 133, batchsize),
            L2(133, 133, batchsize),
            L3(133, 133, batchsize);
        LinearLayer<FLOAT> L10(d, 133, batchsize);

        // nns are linked lists containing layers
        list<Layer<FLOAT> *> trainnn({&L10, &L3, &L2, &L1, &L0});

        // start the training loop
        for (int i = 0; i < numbatches * 480; i++)
        {

            size_t batch_id = i % numbatches;

            // obtain batch
            auto X0_i = X0.columns(batchsize * batch_id, batchsize * (batch_id + 1));
            auto X1_i = X1.columns(batchsize * batch_id, batchsize * (batch_id + 1));

            // sample time uniformly from [0, 1]
            auto t = Matrix<FLOAT>::rand(1, batchsize);

            // compute the interpolation between X0i and X1i
            auto Xt_i = hadmd(X0_i, ones(d, 1) * (1 - t)) + hadmd(X1_i, ones(d, 1) * t);
            // add time to the input
            auto inp_i = vs({Xt_i, t});
            auto Yt_i = X1_i - X0_i;

            // forward-backward pass
            LossLayer<FLOAT> L11(batchsize, Yt_i);
            trainnn.push_front(&L11);

            if (i % (25 * numbatches) == 0)
            {
                float loss = item(forward(trainnn, inp_i));
                losses.push_back(loss);
                std::string msg = ", training loss: " + std::to_string(loss);
                dumpweights(trainnn, base + "/res/net.weights");
            }
            else
            {
                progress = (float)i / (numbatches * 480);
                forward(trainnn, inp_i);
            }

            backprop(trainnn, inp_i);
            trainnn.pop_front();

            if (i % (numbatches) == 0)
            {
                Z1 = euler_integration(Z0, trainnn, 100).back();
            }
        }

        progress = 1.0f;

        auto t2 = Clock::now();
        elapsed_time = time_in_ms(t1, t2) / 1000.0f;

        dumpweights(trainnn, base + "/res/net.weights");
        return 0;
    }
    int render()
    {
        ImGui::Begin("Hello, world!");
        ImGui::Text("song.liu@bristol.ac.uk");
        ImGui::ProgressBar(progress);

        // two columns
        ImGui::Columns(2, "mycolumns");

        // find the maximum value of the losses
        float max_loss = *std::max_element(losses.begin(), losses.end());
        float min_loss = *std::min_element(losses.begin(), losses.end());
        ImPlot::SetNextAxesLimits(0, losses.size(), min_loss - .05, max_loss + .05, ImGuiCond_Always);
        // get the width of the current view port
        ImVec2 size = ImGui::GetWindowSize();
        float width = size.x - 20;
        ImPlot::BeginPlot("Training Loss", ImVec2(width / 2, width / 2));
        // set line width and color
        ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 1), 2.0f);
        ImPlot::PlotLine("", losses.data(), losses.size());

        ImPlot::EndPlot();

        ImGui::NextColumn();

        // plot a random matrix
        ImPlot::BeginPlot("Samples", ImVec2(width / 2, width / 2));
        // set x axis and y axis ranges
        ImPlot::SetupAxisLimits(ImAxis_X1, -4, 4);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -4, 4);

        ImPlotStyle &style = ImPlot::GetStyle();
        style.MarkerSize = 2.0f;
        style.Colors[0] = ImVec4(0, 0, 1, .5); // blue

        float x[500], y[500];
        for (int i = 0; i < 500; i++)
        {
            x[i] = Ztrue(0, i);
            y[i] = Ztrue(1, i);
        }

        ImPlot::PlotScatter("True Sample", x, y, Ztrue.num_col());

        style.MarkerSize = 2.0f;
        style.Colors[0] = ImVec4(1, 0, 0, .5); // red

        for (int i = 0; i < 500; i++)
        {
            x[i] = Z1(0, i);
            y[i] = Z1(1, i);
        }

        ImPlot::PlotScatter("Generated Samples", x, y, Z1.num_col());

        ImPlot::EndPlot();
        ImGui::Columns(1);
        if (progress >= 1.0f)
        {
            ImGui::Text("Time: %.2fs", elapsed_time);
        }
        ImGui::End();

        return 0;
    }
};

Code::Code() : pimpl(std::make_unique<Codeimp>())
{
}
Code::~Code() = default;

int Code::run()
{
    return pimpl->run();
}

int Code::render()
{
    return pimpl->render();
}
