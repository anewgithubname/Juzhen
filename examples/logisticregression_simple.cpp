/**
 * A simple (<= 100 lines) logistic regression binary classifier. 
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
// #define NO_CBLAS
#include "../cpp/juzhen.hpp"
using namespace std;
int main(){ MemoryDeleter<float> md1; //will release the memory for us when the function exits
    //creating dataset
    int n = 5000, d = 2;
    Matrix<float> Xpos("X+",d,n); 
    Xpos.randn(); Xpos -= 2.0f;

    Matrix<float> XNeg("X-",d,n); 
    XNeg.randn(); XNeg += 0.0f;

    Matrix<float> theta("theta",d,1); 
    theta.zeros();
    float b = 1.0;
    
    { cout<<"training..."<<endl; 
        //training
        for (int i = 0; i <= 1000; i++){
            //computing gradient
            auto fPos = theta.T()*Xpos+b;
            auto fNeg = -(theta.T()*XNeg+b);

            auto pt1 = -exp(-fPos)/(exp(-fPos) + 1.0f);
            auto pt2 = Xpos*pt1.T()/n;

            auto nt1 = -exp(-fNeg)/(exp(-fNeg) + 1.0f);
            auto nt2 = -XNeg*nt1.T()/n;

            auto g = pt2 + nt2;

            //gradient descent
            theta -= 1.0f*g;
            b -= 1.0f*(sum(pt1,1)/n - sum(nt1,1)/n).elem(0,0);

            //printing progress
            if (i % 100 == 0){
                cout << "Iteration: " << i << endl;
                cout << "norm of gradient (g): "<<g.norm() << endl;
                cout<<theta.T()<<endl;
                cout<<"bias (b): "<<b<<endl;
                cout << endl;
            }
        }
    }

    int nt = 1000;
    //testing
    {cout<< "testing: " << endl;
        Matrix<float> XposTe("X+Te",d,nt); 
        XposTe.randn(); XposTe -= 2.0f;

        Matrix<float> XNegTe("X-Te",d,nt); 
        XNegTe.randn(); XNegTe += 0.0f;

        auto XTe = hstack(vector<Matrix<float>>{XposTe, XNegTe});
        auto pred = theta.T()*XTe+b;
        int FP = 0, FN = 0;
        for(int i = 0; i < 2*nt; i++){
            if(pred.elem(0,i) <0 && i < nt)
                FN++;
            else if(pred.elem(0,i) >0 && i >= nt)
                FP++;
        }
        cout << "False positives: " << FP << endl;
        cout << "False negatives: " << FN << endl;
    }
}