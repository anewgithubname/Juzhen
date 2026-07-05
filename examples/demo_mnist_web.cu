/**
 * @file demo_mnist_web.cu
 * @brief Web demo: train the MNIST classifier, then serve a handwriting
 *        recognition page over HTTP (see examples/web/mnist.html).
 * @author Song Liu (song.liu@bristol.ac.uk)
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

// #define CPU_ONLY

#include "../ml/layer.hpp"
#include "../external/httplib.h"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

using namespace std;
using namespace Juzhen;

#if defined(CUDA)
#define FLOAT CUDAfloat
#elif defined(ROCM_HIP)
#define FLOAT ROCMfloat
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
#else
#define FLOAT float
#endif

namespace {

// parse every number in the request body (e.g. a JSON array of 784 pixels)
vector<float> parse_numbers(const string &body) {
    vector<float> vals;
    const char *p = body.c_str();
    while (*p) {
        if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '.') {
            char *end = nullptr;
            vals.push_back(strtof(p, &end));
            p = end;
        } else {
            p++;
        }
    }
    return vals;
}

vector<float> softmax(const vector<float> &scores) {
    float maxscore = *max_element(scores.begin(), scores.end());
    vector<float> probs(scores.size());
    float total = 0;
    for (size_t i = 0; i < scores.size(); i++) {
        probs[i] = exp(scores[i] - maxscore);
        total += probs[i];
    }
    for (auto &p : probs) p /= total;
    return probs;
}

}  // namespace

int compute() {
    const int d = 28 * 28, k = 10, batchsize = 32;

#ifdef CUDA
    GPUSampler sampler(1);
#endif
#ifdef ROCM_HIP
    global_rand_gen.seed(1);
#endif

    auto vecXY = mnist_dataset();
    auto X = vecXY[0];
    auto Y = vecXY[1];
    const size_t numbatches = X.num_col() / batchsize;

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
    auto XT = Matrix<FLOAT>(vecXY[2]);
    auto YT = Matrix<FLOAT>(vecXY[3]);
#else
    auto &XT = vecXY[2];
    auto &YT = vecXY[3];
#endif

    // define layers, same architecture as demo_mnist
    Layer<FLOAT> L0(1024, d, batchsize), L1(128, 1024, batchsize);
    LinearLayer<FLOAT> L2(k, 128, batchsize);
    ZeroOneLayer<FLOAT> L3t(XT.num_col(), YT);

    list<Layer<FLOAT> *> trainnn({&L2, &L1, &L0}), testnn({&L3t, &L2, &L1, &L0});

    // sgd
    int iter = 0;
    while (iter < 10000) {
        size_t batch_id = (iter % numbatches);

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
        auto X_i = Matrix<FLOAT>(X.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
        auto Y_i = Matrix<FLOAT>(Y.columns(batchsize * batch_id, batchsize * (batch_id + 1)));
#else
        auto X_i = X.columns(batchsize * batch_id, batchsize * (batch_id + 1));
        auto Y_i = Y.columns(batchsize * batch_id, batchsize * (batch_id + 1));
#endif

        forward(trainnn, X_i);
        LogisticLayer<FLOAT> L3(batchsize, std::move(Y_i));
        trainnn.push_front(&L3);

        backprop(trainnn, X_i);
        trainnn.pop_front();
        if (iter % 1000 == 0) {
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            cout << "Misclassification Rate: " << forward(testnn, XT).to_host().elem(0, 0) << endl;
#else
            cout << "Misclassification Rate: " << forward(testnn, XT).elem(0, 0) << endl;
#endif
        }

        iter++;
    }

#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
    float testerr = forward(testnn, XT).to_host().elem(0, 0);
#else
    float testerr = forward(testnn, XT).elem(0, 0);
#endif
    cout << "Training done. Final misclassification rate: " << testerr << endl;

    // the matrix library is not thread-safe; serialize predictions
    mutex predict_mutex;

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
        ifstream f(PROJECT_DIR + string("/examples/web/mnist.html"));
        if (!f) {
            res.status = 500;
            res.set_content("examples/web/mnist.html not found", "text/plain");
            return;
        }
        stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    svr.Get("/info", [&](const httplib::Request &, httplib::Response &res) {
        ostringstream ss;
        ss << "{\"testerror\":" << testerr << "}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/predict", [&](const httplib::Request &req, httplib::Response &res) {
        auto pixels = parse_numbers(req.body);
        if ((int)pixels.size() != d) {
            res.status = 400;
            res.set_content("{\"error\":\"expected 784 pixel values\"}", "application/json");
            return;
        }

        Matrix<float> x("input", d, 1);
        x.zeros();
        for (int i = 0; i < d; i++) x.elem(i, 0) = pixels[i];

        vector<float> scores(k);
        {
            lock_guard<mutex> lock(predict_mutex);
#if defined(CUDA) || defined(ROCM_HIP) || defined(APPLE_SILICON)
            auto out = forward(trainnn, Matrix<FLOAT>(x)).to_host();
#else
            const auto &out = forward(trainnn, x);
#endif
            for (int c = 0; c < k; c++) scores[c] = (float)out.elem(c, 0);
        }

        auto probs = softmax(scores);
        int pred = (int)(max_element(probs.begin(), probs.end()) - probs.begin());

        ostringstream ss;
        ss << "{\"prediction\":" << pred << ",\"probabilities\":[";
        for (int c = 0; c < k; c++) ss << (c ? "," : "") << probs[c];
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    int port = 8080;
    if (const char *p = getenv("PORT")) port = atoi(p);
    cout << "Serving MNIST web demo on http://localhost:" << port << endl;
    if (!svr.listen("0.0.0.0", port)) {
        cout << "Failed to listen on port " << port << endl;
        return 1;
    }

    return 0;
}
