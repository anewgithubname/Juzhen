/**
 * @file demo_zeroflow_web.cu
 * @brief Web demo of the zero-flow phenomenon (Wang, Wang, Liu & Suzuki,
 *        "Zero-Flow Encoders", ICML 2026, arXiv:2602.00797).
 *
 * A rectified flow trained with INDEPENDENT coupling between a source and a
 * target distribution has velocity field v_{t=0.5}(z) = 0 everywhere if and
 * only if the two distributions are identical (Theorem 3.1); more generally
 * v_t = -v_{1-t} (antisymmetry, Theorem 3.2).
 *
 * The page (examples/web/zeroflow.html) lets you pick a source and a target
 * from a menu of 2-D toy datasets, trains a small velocity-field MLP on the
 * fly, then streams back an animation of particles transported from t=0 to
 * t=1, coloured by speed. When source == target, everything freezes (turns
 * blue) at t = 0.5 — the zero-flow criterion made visible.
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

#include "../ml/layer.hpp"
#include "../cpp/juzhen.hpp"
#include "../external/httplib.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace Juzhen;

#ifdef CUDA
#define FLOAT CUDAfloat
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
#else
#define FLOAT float
#endif

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

// ── 2-D toy datasets, all roughly inside [-3.5, 3.5]^2 ──────────────────────
static const char* DATASETS[] = {"gaussian", "mixture", "moons", "ring", "spiral"};

static Matrix<float> sample_dataset(const string& name, int n) {
    normal_distribution<float> g(0.0f, 1.0f);
    uniform_real_distribution<float> u(0.0f, 1.0f);
    Matrix<float> X("data", 2, n);
    X.zeros();
    for (int i = 0; i < n; ++i) {
        float x = 0, y = 0;
        if (name == "mixture") {
            const float c = (u(global_rand_gen) < 0.5f) ? -1.8f : 1.8f;
            x = c + 0.5f * g(global_rand_gen);
            y = c + 0.5f * g(global_rand_gen);
        } else if (name == "moons") {
            const float th = 3.14159265f * u(global_rand_gen);
            if (u(global_rand_gen) < 0.5f) {
                x =  2.0f * cosf(th) - 1.0f;
                y =  2.0f * sinf(th) - 0.6f;
            } else {
                x = -2.0f * cosf(th) + 1.0f;
                y = -2.0f * sinf(th) + 0.6f;
            }
            x += 0.15f * g(global_rand_gen);
            y += 0.15f * g(global_rand_gen);
        } else if (name == "ring") {
            const float th = 6.2831853f * u(global_rand_gen);
            const float r = 2.2f + 0.15f * g(global_rand_gen);
            x = r * cosf(th);
            y = r * sinf(th);
        } else if (name == "spiral") {
            const float s = u(global_rand_gen);
            const float th = 4.0f * 3.14159265f * s;
            const float r = 0.4f + 2.6f * s;
            x = r * cosf(th) + 0.1f * g(global_rand_gen);
            y = r * sinf(th) + 0.1f * g(global_rand_gen);
        } else {  // "gaussian"
            x = 0.8f * g(global_rand_gen);
            y = 0.8f * g(global_rand_gen);
        }
        X.elem(0, i) = x;
        X.elem(1, i) = y;
    }
    return X;
}

static bool valid_dataset(const string& name) {
    for (const char* d : DATASETS)
        if (name == d) return true;
    return false;
}

// compact float formatting for JSON payloads
static void put_f(ostringstream& ss, float v) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%.3f", v);
    ss << buf;
}

int compute() {
#ifdef CUDA
    GPUSampler sampler(7);
#endif
    global_rand_gen.seed(7);

    // one request at a time: the matrix library is not thread-safe
    auto flow_mutex = make_shared<mutex>();

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        ifstream f(PROJECT_DIR + string("/examples/web/zeroflow.html"));
        if (!f) {
            res.status = 500;
            res.set_content("examples/web/zeroflow.html not found", "text/plain");
            return;
        }
        stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    // POST /train?source=mixture&target=mixture&iters=4000
    // Trains v_t(x) = argmin E||X' - X - u(X_t, t)||^2 with X_t = tX' + (1-t)X
    // and INDEPENDENT X ~ source, X' ~ target (eq. 5 of the paper), then
    // integrates M particles from t=0 to 1 and streams NDJSON:
    //   {"phase":"train","iter":..,"loss":..}
    //   {"phase":"grid","x":[..],"y":[..]}
    //   {"phase":"frame","t":..,"mean_v":..,"pts":[x,y,s,..],"vec":[u,v,..]}
    //   {"phase":"done"}
    svr.Post("/train", [&](const httplib::Request& req, httplib::Response& res) {
        string src = req.get_param_value("source");
        string tgt = req.get_param_value("target");
        int iters = 4000;
        if (req.has_param("iters")) iters = atoi(req.get_param_value("iters").c_str());
        iters = max(200, min(iters, 20000));
        if (!valid_dataset(src) || !valid_dataset(tgt)) {
            res.status = 400;
            res.set_content("{\"error\":\"unknown dataset\"}", "application/json");
            return;
        }

        res.set_chunked_content_provider(
            "application/x-ndjson",
            [&, src, tgt, iters](size_t, httplib::DataSink& sink) {
                lock_guard<mutex> lock(*flow_mutex);
                auto emit = [&](const string& line) {
                    return sink.write(line.c_str(), line.size());
                };

                const int d = 2, batchsize = 256;
                const float lr = 1e-3f;

                // velocity-field MLP: (x, y, t) -> (vx, vy)
                ReluLayer<FLOAT> L0(128, d + 1, batchsize),
                                 L1(128, 128, batchsize),
                                 L2(128, 128, batchsize);
                LinearLayer<FLOAT> L3(d, 128, batchsize);
                list<Layer<FLOAT>*> net({&L3, &L2, &L1, &L0});
                for (auto* l : net) {
                    l->adamWstate().alpha = lr;
                    l->adambstate().alpha = lr;
                }

                auto vs = [](vector<MatrixView<FLOAT>> v) { return vstack(v); };
                auto ones21 = Matrix<FLOAT>::ones(d, 1);

                // ── rectified-flow training with independent coupling ─────
                for (int i = 0; i < iters; ++i) {
                    auto X0 = Matrix<FLOAT>(sample_dataset(src, batchsize));
                    auto X1 = Matrix<FLOAT>(sample_dataset(tgt, batchsize));
                    auto t = Matrix<FLOAT>::rand(1, batchsize);
                    auto Xt = hadmd(X0, ones21 * (1 - t)) + hadmd(X1, ones21 * t);
                    auto inp = vs({Xt, t});
                    LossLayer<FLOAT> ls(batchsize, X1 - X0);
                    net.push_front(&ls);
                    float loss = item(forward(net, inp));
                    backprop(net, inp);
                    net.pop_front();

                    if (i % 200 == 0 || i == iters - 1) {
                        ostringstream ss;
                        ss << "{\"phase\":\"train\",\"iter\":" << i
                           << ",\"total\":" << iters << ",\"loss\":";
                        put_f(ss, loss);
                        ss << "}\n";
                        if (!emit(ss.str())) return true;   // client gone
                    }
                }

                // ── animation: particles + a velocity-field grid ──────────
                const int M = 300;          // particles
                const int K = 50;           // frames (t = 0 .. 1)
                const int G = 12;           // G x G field grid
                const float LO = -3.5f, HI = 3.5f;

                // grid positions (fixed; sent once)
                Matrix<float> grid_h("grid", 2, G * G);
                grid_h.zeros();
                {
                    ostringstream ss;
                    ss << "{\"phase\":\"grid\",\"x\":[";
                    for (int a = 0; a < G; ++a)
                        for (int b = 0; b < G; ++b) {
                            const int j = a * G + b;
                            grid_h.elem(0, j) = LO + (HI - LO) * (a + 0.5f) / G;
                            grid_h.elem(1, j) = LO + (HI - LO) * (b + 0.5f) / G;
                            if (j) ss << ",";
                            put_f(ss, grid_h.elem(0, j));
                        }
                    ss << "],\"y\":[";
                    for (int j = 0; j < G * G; ++j) {
                        if (j) ss << ",";
                        put_f(ss, grid_h.elem(1, j));
                    }
                    ss << "]}\n";
                    if (!emit(ss.str())) return true;
                }
                auto grid = Matrix<FLOAT>(grid_h);

                auto Z = Matrix<FLOAT>(sample_dataset(src, M));
                const float dt = 1.0f / K;
                for (int k = 0; k <= K; ++k) {
                    const float tv = (float)k / K;
                    auto t_row = Matrix<FLOAT>::ones(1, M) * tv;
                    auto V = forward(net, vs({Z, t_row}));
                    auto Zh = as_host(Z);
                    auto Vh = as_host(V);

                    auto tg = Matrix<FLOAT>::ones(1, G * G) * tv;
                    auto VG = as_host(forward(net, vs({grid, tg})));

                    float mean_v = 0.0f;
                    ostringstream ss;
                    ss << "{\"phase\":\"frame\",\"t\":";
                    put_f(ss, tv);
                    ss << ",\"pts\":[";
                    for (int j = 0; j < M; ++j) {
                        const float sp = sqrtf(Vh.elem(0, j) * Vh.elem(0, j) +
                                               Vh.elem(1, j) * Vh.elem(1, j));
                        mean_v += sp;
                        if (j) ss << ",";
                        put_f(ss, Zh.elem(0, j)); ss << ",";
                        put_f(ss, Zh.elem(1, j)); ss << ",";
                        put_f(ss, sp);
                    }
                    mean_v /= M;
                    ss << "],\"vec\":[";
                    for (int j = 0; j < G * G; ++j) {
                        if (j) ss << ",";
                        put_f(ss, VG.elem(0, j)); ss << ",";
                        put_f(ss, VG.elem(1, j));
                    }
                    ss << "],\"mean_v\":";
                    put_f(ss, mean_v);
                    ss << "}\n";
                    if (!emit(ss.str())) return true;

                    if (k < K) Z = Z + V * dt;   // Euler step
                }

                emit("{\"phase\":\"done\"}\n");
                sink.done();
                return true;
            });
    });

    int port = 8096;
    if (const char* p = getenv("PORT")) port = atoi(p);
    cout << "Serving zero-flow web demo on http://localhost:" << port << endl;
    if (!svr.listen("0.0.0.0", port)) {
        cout << "Failed to listen on port " << port << endl;
        return 1;
    }

    return 0;
}
