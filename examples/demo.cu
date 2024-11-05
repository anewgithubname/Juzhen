/**
 * @file demo.cu
 * @brief test basic functionality of the library
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

#include "../cpp/juzhen.hpp"
using namespace std;
#define MatRef const Matrix<float> & 
#define cuMatrix Matrix<CUDAfloat>

void cpu_matrixaccess(){
    {
        cout << "basic hstack" << endl;
        Matrix<float> A = {"A", {{1,2,3},{4,5,6}}};
        Matrix<float> B = {"B", {{7,8},{9,8},{7,6}}};
    
        cout << hstack<float>({A.T(),B,A.T(),B,A.T(),B}) <<endl << endl;
    }
    {
        cout << "big vstack" << endl;
        Matrix<float> &&A = {"A", 5000,5000}; A.zeros();
        Matrix<float> &&B = {"B", 5000,5000}; B.zeros();
        auto t1 = Clock::now();
        cout << vstack<float>({A.T(),B,A.T(),B,A.T(),B}).num_col() << endl;
        auto t2 = Clock::now();
        cout << "time: " << time_in_ms(t1,t2) << "ms." << endl <<endl;
    }

    {
        cout<< "index generation "<<endl;
        auto idx = seq(0, 10);
        for(size_t &i : idx) cout<<i<<" ";
        cout << endl; 

        idx = shuffle(0,10);
        for(size_t &i : idx) cout<<i<<" ";
        cout << endl <<endl; 
    }

    {
        cout << "slicing" << endl;
        Matrix<float> &&A = {"A", 5000,5000}; A.zeros();
        Matrix<float> &&B = {"B", 5000,5000}; B.zeros();

        auto t1 = Clock::now();
        cout << A.slice(0,A.num_row(),123,123+234).num_col()<<endl;
        auto t2 = Clock::now();
        cout << "time: " << time_in_ms(t1, t2) << " ms" << endl <<endl;
        
        cout << "random accessing" << endl;
        t1 = Clock::now();
        cout << A.slice(seq(A.num_row()),shuffle(123,123+234)).num_col()<<endl;
        t2 = Clock::now();
        cout << "time: " << time_in_ms(t1, t2) << " ms" << endl<<endl;
    }

}

void cpu_arithmetics(){
    auto A = Matrix<float>::randn(800, 800);
    auto B = Matrix<float>::randn(800, 800);

    cout<< "timer started..."<<endl;
    auto t1 = Clock::now();
    Matrix<float> && C = (((A+B).T()*100.0f*(A+B)/5.0+1.0f)*2.0f+2.0f).inv()/4.0;
    auto t2 = Clock::now();
    cout << "Time taken: " << time_in_ms(t1, t2) <<" ms" << endl;

    write("/tmp/C.matrix", A);
}

void cpu_slowinv(){
    
    auto A = Matrix<float>::randn(500, 1000);
    A = A*A.T()/10000.0;

    Matrix<float> Astar("Astar", 500, 500);
    Astar.zeros();
    for (int i = 0; i < 10; i++)
    {
        auto t1 = Clock::now();
        Astar -= .1f*(Astar * (A * A) - A);
        auto t2 = Clock::now();
        cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
 
        cout << Astar.norm() << endl;
    }
}

void cpu_dre(){
    cout<<"generating data..."<<endl;
    int d = 500, n = 5000;
    auto Xp = Matrix<float>::randn(n,d);
    // Xp.read("/tmp/Xp.matrix");
    auto Xq = Matrix<float>::randn(n, d);
    // Xq.read("/tmp/Xq.matrix");

    Matrix<float> theta("theta", d, 1);
    theta.zeros();

    cout<<"timer started..."<<endl;
    auto t1 = Clock::now();
    for (int i = 0; i < 200; i++)
    {
        MatRef g1 = sum(Xp,0)/(double)n;
        MatRef N = sum(exp(Xq*theta),0)/(double)n;
        MatRef g2 = exp(Xq*theta).T()*Xq/N.elem(0,0)/(double)n;
        MatRef g = - g1 + g2;
        theta +=  - 0.1f*g.T();
        cout<< theta.norm()<<endl;
    }
    auto t2 = Clock::now();
    cout<<theta.norm()<<endl;
    cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
}

#ifndef CPU_ONLY
void cuda_basic(){
    {
        cuMatrix A(Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(Matrix<float>("B",{{1,4},{2,5},{3,6}}));

        cout <<hstack({A,B.T()}).to_host()<<endl;

        cout <<vstack({A, B.T()}).to_host()<<endl;

        auto t1 = Clock::now();
        cuMatrix AA = cuMatrix::randn(5000, 10000);
        cuMatrix BB = cuMatrix::randn(10000, 5000);
        auto t2 = Clock::now();
        cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
        cout << vstack({AA,BB.T(),AA,BB.T(),AA,BB.T()}).num_row()<<endl;
    }
    {
        cuMatrix A(Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(Matrix<float>("B",{{1,2,3},{4,5,6}}));
        B = B.T();
        cout<< A.to_host() <<endl; cout<< B.to_host() <<endl;
        // cout<<(A*B*BT*BT2*B.T()*A.T()).to_host()<<endl;
        cout << (A -= B.T()).to_host() <<endl;
        cout<<  (2*(A*3+B.T()/2+1)).to_host()<<endl;
        cout <<B.to_host()<<endl;
        cout << sum(B.T(),0).to_host() <<endl;
        cout << log(exp(B)).to_host() <<endl;
    }

    {
        printf("testing hadmd \n");
        cuMatrix A(Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cuMatrix A1(Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(Matrix<float>("B",{{1,2,3},{4,5,6}}));
        cout << hadmd(A1, B).to_host() <<endl;
        cout << hadmd(A.T(), B).to_host() <<endl;
        cout << hadmd(A, B.T()).to_host() <<endl;
    }

    {
        printf("elem div \n");
        cuMatrix A(Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cout << (1/A).to_host() <<endl;
        cout << (1/A.T()).to_host() <<endl;
    }

    {
        printf("elem div2 \n");
        cuMatrix A(Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cuMatrix A1(Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(Matrix<float>("B",{{1,2,3},{4,5,6}}));
        cout << (A1/B).to_host() <<endl;
        cout << (A/B.T()).to_host() <<endl;
        cout << (A.T()/B).to_host() <<endl;
    }

    {
        printf("tanh \n");
        cuMatrix A(Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cout << (tanh(A)).to_host() <<endl;
        cout << (d_tanh(A.T())).to_host() <<endl;
    }
}

void cuda_slowinv(){

    cout << "generating data..."<<endl;
    int d = 500, n = 1000;
    auto A = cuMatrix::randn(d,n);
    A = A*A.T()/(float)n;

    cuMatrix Astar("Astar", d, d);

    cout<<"allocating"<<endl;
    cout<<"process started"<<endl;
    for (int i = 0; i < 10; i++)
    {
        auto t1 = Clock::now();
        Astar -= .1f*(Astar * (A * A) - A);
        auto t2 = Clock::now();
        cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
        cout << Astar.norm() << endl;
     }

}

void cuda_dre(){

    cout<<"generating data..."<<endl;
    int d = 500, n = 5000;
    auto Xp = cuMatrix::randn(n, d);
    auto Xq = cuMatrix::randn(n, d);

    cuMatrix theta("theta", d, 1);

    cout<<"timer started..."<<endl;
    auto t1 = Clock::now();

    cuMatrix cuTheta_old("theta_old", d, 1); 
    for (int i = 0; i < 200; i++)
    {
        cuMatrix && g1 = sum(Xp,0)/(float)n;
        cuMatrix && N = sum(exp(Xq*theta),0)/(float)n;
        cuMatrix && g2 = exp(Xq*theta).T()*Xq/N.norm()/(float)n;
        cuMatrix && g = g2 - g1;
        theta +=  - 0.1*g.T();
        cout<< (cuTheta_old - theta).norm() <<endl;
        cuTheta_old = theta*1;
    }
    auto t2 = Clock::now();

    cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;

}
#endif

int compute(){ 
    global_rand_gen.seed(0);
    
    //spdlog::set_level(spdlog::level::debug);
    //std::cout << __cplusplus << " " << HAS_CONCEPTS << std::endl;

    printf("cpu_matrixaccess\n");
    cpu_matrixaccess();

    printf("\ncpu_arithmetics\n");
    cpu_arithmetics();

    printf("\ncpu_slowinv\n");
    cpu_slowinv();

    printf("\ncpu_dre\n");
    cpu_dre();

#ifndef CPU_ONLY
    GPUSampler sampler(1);
     printf("\ncuda_basic\n");
     cuda_basic();

     printf("\ncuda_slowinv\n");
     cuda_slowinv();
    
    printf("\ncuda_dre\n");
    cuda_dre();
#endif
    return 0;
}
