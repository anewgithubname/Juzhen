/**
 * Logistic Regression using MLP with one hidden layer. 
 * Author: Song Liu (song.liu@bristol.ca.uk)
 *  Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#include "juzhen.hpp"
#include <math.h>
#define MatrixI Matrix<int> 
#define MatrixF Matrix<float> 
#define PROFILING

// convert label Y matrix (1 X n) to one-hot encoding. 
MatrixF one_hot(const MatrixI &Y, int k){
    MatrixF Y_one_hot("One_hot", k, Y.num_col());
    Y_one_hot.zeros();

    for(int i = 0; i < Y.num_col(); i++){
        Y_one_hot.elem(Y.elem(0, i), i) = 1.0;
    }

    return Y_one_hot;
}

// Logistic Regression
int main(){ MemoryDeleter<float> md1; MemoryDeleter<int> md2; //will release the memory for us when the function exits
    // load data
    int n = 60000, d = 28*28, k = 10;
    MatrixF X("X",d,n); 
    X.read("X.matrix"); X = std::move(X.T());
    cout << "size of X: " << X.num_row() << " " << X.num_col() << endl;

    MatrixI labels("labels",1,n); 
    labels.read("Y.matrix"); labels = std::move(labels.T());
    cout << "size of labels: " << labels.num_row() << " " << labels.num_col() << endl;

    MatrixF Y = one_hot(labels, k);
    cout << "size of Y: " << Y.num_row() << " " << Y.num_col() << endl;

    //initialize parameters
    int m = 28*28*3;
    MatrixF W1("W1",m,d); // first layer coefficient, m x d
    W1.randn(0,.1);
    MatrixF W2("W2",k,m); // second layer coefficient, k x m
    W2.randn(0,.1);
    MatrixF b1("b1",m,1); // first layer bias, m x 1
    b1.randn(0,.1);
    MatrixF b2("b2",k,1); // second layer bias k x 1
    b2.randn(0,.1);

    MatrixF O_k1("ones", k, 1); O_k1.ones();

    // set up batch size
    int nb = 128; int n_batch = n/nb;
    MatrixF O_1nb("ones", 1, nb); O_1nb.ones();

    auto t1 = Clock::now(); auto t2 = Clock::now();
    MatrixF O_1n("ones", 1, n); O_1n.ones();
    MatrixF && f = W2*tanh(W1*X+b1*O_1n) + b2*O_1n;
    cout <<"obj " << -sum(sum(hadmd(Y,f),0) -log(sum(exp(f),0)),1)/n << endl;
    
    #ifdef PROFILING
    Profiler p1("batching"), p2("computing"), p3("updating");
    #endif
    // training, run gradient descent
    for (int i = 0; i <= 10000; i++){            
        // select batches 
#ifdef PROFILING 
        p1.start(); 
#endif
        int batch_id = (i%n_batch);
        MatrixF && X_i = X.columns(nb*batch_id, nb*(batch_id+1));
        MatrixF && Y_i = Y.columns(nb*batch_id, nb*(batch_id+1));
#ifdef PROFILING
        p1.end(); 
#endif
        
#ifdef PROFILING 
        p2.start();
#endif
        // W1X_i + b1, saving to a temp var
        auto f1 = W1*X_i+b1*O_1nb;
        // activation function
        auto tanhf1 = tanh(f1);
        // partial derivative of tanh(W1 X + b1)
        auto d_tanh_f1 = d_tanh(f1);
        // forward evaluation
        auto f = W2*tanh(std::move(f1)) + b2*O_1nb;

        auto invZ = O_k1*(1.0f/sum(exp(f),0));
        // softmax of f
        auto pred = hadmd(exp(f),invZ);

        // computing W1 gradient using chain rule
        auto g_W1 = - hadmd(d_tanh_f1,W2.T()*Y_i)*X_i.T()/nb;
                   g_W1 +=  hadmd(d_tanh_f1,W2.T()*pred)*X_i.T()/nb;
        // computing W2 gradient using chain rule
        auto g_W2 = - Y_i*(tanhf1).T() /nb;
                   g_W2 +=  pred*(tanhf1).T()/nb;
        // computing b1 gradient using chain rule
        auto g_b1 = - hadmd(d_tanh_f1,W2.T()*Y_i)*O_1nb.T()/nb;
                   g_b1 +=  hadmd(d_tanh_f1,W2.T()*pred)*O_1nb.T()/nb;
        // computing b2 gradient using chain rule
        auto g_b2 = - Y_i*O_1nb.T()/nb + pred*O_1nb.T()/nb;
#ifdef PROFILING
        p2.end();
#endif

#ifdef PROFILING
        p3.start();
#endif
        // gradient descent, g_W1, g_W2, g_b1, g_b2 are sacrificed for speed.
        W1 -= .01f*std::move(g_W1);
        W2 -= .01f*std::move(g_W2);
        b1 -= .01f*std::move(g_b1);
        b2 -= .01f*std::move(g_b2);
#ifdef PROFILING
        p3.end();
#endif


        if (i % 500 == 0){
            // print out training status
            cout << "Iteration: " << i << endl;
            MatrixF O_1n("ones", 1, n); O_1n.ones();
            MatrixF && f = W2*tanh(W1*X+b1*O_1n) + b2*O_1n;
            cout <<"obj " << -sum(sum(hadmd(Y,f),0) -log(sum(exp(f),0)),1)/n << endl;
            t2 = Clock::now();
            cout << time_in_ms(t1,t2) << " ms" << endl;
            cout<<endl; 
            t1 = Clock::now();
        }
    }

    // load test data
    int nt = 10000;
    MatrixF Xt("Xt",d,nt); 
    Xt.read("T.matrix"); Xt = std::move(Xt.T());
    cout<<"size of Xt: "<<Xt.num_row()<<" "<<Xt.num_col()<<endl;

    MatrixI labels_t("labels_t",1,nt); 
    labels_t.read("YT.matrix"); labels_t = std::move(labels_t.T());
    cout<<"size of labels_t: "<<labels_t.num_row()<<" "<<labels_t.num_col()<<endl;

    MatrixF O_1nt("ones", 1, nt); O_1nt.ones();
    // prediction
    MatrixF && ft = exp(W2*tanh(W1*Xt+b1*O_1nt) + b2*O_1nt);
    MatrixI pred("pred",1,nt);

    // compute test accuracy
    int err = 0;
    for(int i = 0; i < nt; i++){
        double max = 0;
        for (int j = 0; j < k; j++){
            if (ft.elem(j,i) > max){
                max = ft.elem(j,i);
                pred.elem(0,i) = j;
            }
        }
        if (pred.elem(0,i) != labels_t.elem(0,i)){
            // cout << "Prediction error: " << pred.elem(0,i) << " " << labels_t.elem(0,i) << endl;
            err++;
        }
    }
    cout << "testing error: "<< (double)err/nt << endl;
}
