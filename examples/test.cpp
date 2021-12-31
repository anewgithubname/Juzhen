#include "juzhen.hpp"
using namespace std;
#define MatRef const Matrix<float> & 
void cpu_matrixaccess(){
    {
        cout << "basic hstack" << endl;
        Matrix<float> A = {"A", {{1,2,3},{4,5,6}}};
        Matrix<float> B = {"B", {{7,8},{9,8},{7,6}}};
    
        cout << hstack(std::vector<Matrix<float>>({A.T(),B,A.T(),B,A.T(),B})) <<endl << endl;
    }
    {
        cout << "big vstack" << endl;
        Matrix<float> &&A = {"A", 5000,5000}; A.zeros();
        Matrix<float> &&B = {"B", 5000,5000}; B.zeros();
        vector<Matrix<float>> matrices = {A.T(),B,A.T(),B,A.T(),B};
        auto t1 = Clock::now();
        cout << vstack(matrices).num_col() << endl;
        auto t2 = Clock::now();
        cout << "time: " << time_in_ms(t1,t2) << "ms." << endl <<endl;
    }

    {
        cout<< "index generation "<<endl;
        auto idx = seq(0, 10);
        for(int &i : idx) cout<<i<<" ";
        cout << endl; 

        idx = shuffle(0,10);
        for(int &i : idx) cout<<i<<" ";
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
      // srand(time(NULL));
    Matrix<float> A("A", 8000, 8000);
    A.randn();
    A.write("/tmp/A.matrix");

    Matrix<float> B("B", 8000, 8000);
    B.randn();
    B.write("/tmp/B.matrix");
    cout<< "timer started..."<<endl;
    auto t1 = Clock::now();
    // Matrix<float> C = A.T()*2.0;
    Matrix<float> && C = (((A+B).T()*100.0f*(A+B)/5.0+1.0f)*2.0f+2.0f).inv()/4.0;
    // Matrix<float> C = (A*A.T()/8000.0 + B*B.T()/8000.0).inv()-4.0;
    auto t2 = Clock::now();
    cout << "Time taken: " << time_in_ms(t1, t2) <<" ms" << endl;

    C.write("/tmp/C.matrix");
}

void cpu_slowinv(){
    
    Matrix<float> A("A", 5000, 10000);
    A.randn();
    A = A*A.T()/10000.0;
    // A.read("/tmp/AA.matrix");

    Matrix<float> Astar("Astar", 5000, 5000);
    Astar.zeros();
    // Astar.read("/tmp/Astar.matrix");
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
    int d = 5000, n = 100000;
    Matrix<float> Xp("Xp", n, d);
    // Xp.read("/tmp/Xp.matrix");
    Xp.randn();
    Matrix<float> Xq("Xq", n, d);
    // Xq.read("/tmp/Xq.matrix");
    Xq.randn();

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
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return; 
    }
    // // cuMatrix cuC = cuA*cuB;
    // auto t1 = Clock::now();
    // Matrix<float> C;
    // for(int i = 0; i < 10; i++)
    // {
    //     Matrix<float> A("A", 3000,3001,0);
    //     A.randn();
    //     Matrix<float> B("B", 3000,3001,0);
    //     B.randn();
    //     cuMatrix cuA(handle, A);
    //     cuMatrix cuB(handle, B);
    //     C = cuA.T()*cuB.T()*cuA*cuB.T()*cuA*cuB.T();
    // }
    // auto t2 = Clock::now();
    // cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
    // cout << C.num_row() << " " << C.num_col() << endl;

    // cout << ((cuMatrix &)C).to_host() <<endl;
    {
        cuMatrix A(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(handle, Matrix<float>("B",{{1,4},{2,5},{3,6}}));

        cout <<hstack(vector<cuMatrix>({A,B.T()})).to_host()<<endl;

        auto matrices = {A, B.T()};
        cout <<vstack(matrices).to_host()<<endl;

        auto t1 = Clock::now();
        cuMatrix AA(handle, "AA", 5000,10000); AA.randn();
        cuMatrix BB(handle, "BB", 10000,5000); BB.randn();
        auto t2 = Clock::now();
        cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
        cout << vstack({AA,BB.T(),AA,BB.T(),AA,BB.T()}).num_row()<<endl;
    }
    {
        cuMatrix A(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(handle, Matrix<float>("B",{{1,2,3},{4,5,6}}));
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
        cuMatrix A(handle, Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cuMatrix A1(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(handle, Matrix<float>("B",{{1,2,3},{4,5,6}}));
        cout << hadmd(A1, B).to_host() <<endl;
        cout << hadmd(A.T(), B).to_host() <<endl;
        cout << hadmd(A, B.T()).to_host() <<endl;
    }

    {
        printf("elem div \n");
        cuMatrix A(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cout << (1/A).to_host() <<endl;
        cout << (1/A.T()).to_host() <<endl;
    }

    {
        printf("elem div2 \n");
        cuMatrix A(handle, Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cuMatrix A1(handle, Matrix<float>("A",{{1,2,3},{4,5,6}}));
        cuMatrix B(handle, Matrix<float>("B",{{1,2,3},{4,5,6}}));
        cout << (A1/B).to_host() <<endl;
        cout << (A/B.T()).to_host() <<endl;
        cout << (A.T()/B).to_host() <<endl;
    }

    {
        printf("tanh \n");
        cuMatrix A(handle, Matrix<float>("A",{{1,4},{2,5},{3,6}}));
        cout << (tanh(A)).to_host() <<endl;
        cout << (d_tanh(A.T())).to_host() <<endl;
    }
    cublasDestroy(handle);
}

void cuda_slowinv(){
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return; 
    }

    cout << "generating data..."<<endl;
    int d = 5000, n=10000;
    cuMatrix A(handle, "A", d, n);
    A.randn();
    A = A*A.T()/(float)n;
    // A.read("/tmp/AA.matrix");
    // cout<<A<<endl;

    cuMatrix Astar(handle, "Astar", d, d);
    // Astar.zeros();
    // Astar.read("/tmp/Astar.matrix");
    // cout<<Astar<<endl;

    cout<<"allocating"<<endl;
    // cuMatrix cuA(handle, A); cuMatrix cuAstar(handle, Astar);
    cout<<"process started"<<endl;
    for (int i = 0; i < 10; i++)
    {
        auto t1 = Clock::now();
        Astar -= .1f*(Astar * (A * A) - A);
        auto t2 = Clock::now();
        cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
        cout << Astar.norm() << endl;
     }

    // cout<<A<<endl;
    cublasDestroy(handle);
}

void cuda_dre(){
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return; 
    }

    cout<<"generating data..."<<endl;
    int d = 5000, n = 50000;
    cuMatrix Xp(handle, "Xp", n, d);
    Xp.randn();
    // Xp.read("/tmp/Xp.matrix");

    cuMatrix Xq(handle, "Xq", n, d);
    Xq.randn();
    // Xq.read("/tmp/Xq.matrix");

    cuMatrix theta(handle, "theta", d, 1);
    // theta.zeros();

    // cuMatrix cuXp(handle, Xp); cuMatrix cuXq(handle, Xq);
    // cuMatrix cuTheta(handle, theta);
    cout<<"timer started..."<<endl;
    auto t1 = Clock::now();

    cuMatrix cuTheta_old(handle, "theta_old", d, 1); 
    for (int i = 0; i < 200; i++)
    {
        cuMatrix && g1 = sum(Xp,0)/(float)n;
        cuMatrix && N = sum(exp(Xq*theta),0)/(float)n;
        cuMatrix && g2 = exp(Xq*theta).T()*Xq/N.norm()/(float)n;
        cuMatrix && g = - g1 + g2;
        theta +=  - 0.1*g.T();
        cout<< (cuTheta_old - theta).norm() <<endl;
        cuTheta_old = theta*1;
    }
    auto t2 = Clock::now();
    // cout<<cuTheta.to_host().sub(0,10,0,1) <<endl;
    // idxlist rowidx = shuffle(123,123+234);
    // cout << (cuTheta.to_host().sub(rowidx,seq(cuTheta.num_col())) 
    //         - cuTheta.sub(rowidx,seq(cuTheta.num_col()))).norm()<<endl;
    cout << "Time taken: " << time_in_ms(t1, t2) << " ms" << endl;
    
    cublasDestroy(handle);
}
#endif

int main(){ 
#ifndef CPU_ONLY
    GPUMemoryDeleter md1; 
#endif
    MemoryDeleter<float> md2;
    printf("cpu_matrixaccess\n");
    cpu_matrixaccess();

    printf("\ncpu_arithmetics\n");     
    cpu_arithmetics();

    printf("\ncpu_slowinv\n");
    cpu_slowinv();

    printf("\ncpu_dre\n");
    cpu_dre();

#ifndef CPU_ONLY
    printf("\ncuda_basic\n");
    cuda_basic();

    printf("\ncuda_slowinv\n");
    cuda_slowinv();
    
    printf("\ncuda_dre\n");
    cuda_dre();
#endif
    return 0;
}