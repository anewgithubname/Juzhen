#include "../ml/assignment.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
using namespace Juzhen;
int compute() {
    std::mt19937 rng(123);
    LinearAssignmentSolver solver;
    double checksum=0;
    for(int n:{32,64,128,256,512}) for(const std::string kind:{"random","geometric","ties","identical_rows"}) {
        // Distinct instances, rather than repeatedly solving a single matrix.
        std::vector<Matrix<float>> inputs;
        for(int k=0;k<12;++k) {
            Matrix<float> a("cost",n,n);
            std::vector<float> x(n),y(n);
            for(int i=0;i<n;++i) { x[i]=float(rng()%65536)/65536; y[i]=float(rng()%65536)/65536; }
            for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
                float v=float(rng()%65536)/65536;
                if(kind=="geometric") v=(x[i]-y[j])*(x[i]-y[j]);
                else if(kind=="ties") v=float(rng()%8);
                else if(kind=="identical_rows") v=float(j);
                a.elem(i,j)=v;
            }
            inputs.push_back(std::move(a));
        }
        for(int k=0;k<2;++k) checksum+=solver.solve(inputs[k]).cost;
        std::vector<double> times;
        for(int k=2;k<12;++k) {
            auto start=std::chrono::steady_clock::now();
            auto r=solver.solve(inputs[k]);
            times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            checksum+=r.cost;
        }
        std::sort(times.begin(),times.end());
        std::cout << std::fixed << std::setprecision(6) << "RESULT n=" << n << " family=" << kind
                  << " median_ms=" << (times[4]+times[5])/2 << " mean_ms="
                  << std::accumulate(times.begin(),times.end(),0.0)/times.size()
                  << " max_ms=" << times.back() << " samples=10\n";
    }
    std::cout << "checksum=" << checksum << std::endl;
    return std::isfinite(checksum) ? 0 : 1;
}
