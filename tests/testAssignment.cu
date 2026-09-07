#include "../ml/assignment.hpp"
#include <functional>
#include <iostream>
#include <random>
using namespace Juzhen;

static double exhaustive(const Matrix<float>& a, bool maximize) {
    if(a.num_row()>a.num_col()) return exhaustive(a.T(),maximize);
    double best=maximize ? -INFINITY : INFINITY;
    std::vector<bool> used(a.num_col());
    std::function<void(size_t,double)> visit=[&](size_t row,double cost) {
        if(row==a.num_row()) { best=maximize ? std::max(best,cost) : std::min(best,cost); return; }
        for(size_t col=0;col<a.num_col();++col) if(!used[col]) {
            used[col]=true; visit(row+1,cost+a.elem(row,col)); used[col]=false;
        }
    };
    visit(0,0); return best;
}
static void validate(const Matrix<float>& a,const AssignmentResult& r) {
    size_t count=0; double cost=0;
    for(size_t i=0;i<a.num_row();++i) if(r.row_to_col[i]>=0) {
        int j=r.row_to_col[i];
        if(size_t(j)>=a.num_col() || r.col_to_row[j]!=int(i)) throw std::runtime_error("Invalid matching");
        ++count; cost+=a.elem(i,j);
    }
    if(count!=std::min(a.num_row(),a.num_col()) || cost!=r.cost) throw std::runtime_error("Invalid cost/cardinality");
}
int compute() {
    LinearAssignmentSolver solver;
    std::mt19937 rng(42);
    int cases=0;
    for(int n=1;n<=7;++n) for(int m=1;m<=7;++m) for(int trial=0;trial<8;++trial) {
        Matrix<float> a("cost",n,m);
        for(int i=0;i<n;++i) for(int j=0;j<m;++j) a.elem(i,j)=(int(rng()%33)-16)*0.25f;
        for(bool maximize:{false,true}) {
            auto result=solver.solve(a,maximize); validate(a,result);
            if(result.cost!=exhaustive(a,maximize)) throw std::runtime_error("Not optimal");
            auto t=solver.solve(a.T(),maximize); validate(a.T(),t);
            if(t.cost!=result.cost) throw std::runtime_error("Transpose mismatch");
            ++cases;
        }
    }
    for(int n:{32,128,512}) {
        auto a=Matrix<float>::ones(n,n);
        for(int i=0;i<n;++i) a.elem(i,i)=-1;
        auto r=solver.solve(a); validate(a,r);
        if(r.cost!=-n) throw std::runtime_error("Large planted optimum failed");
        a.zeros(); r=solver.solve(a); validate(a,r);
        if(r.cost!=0) throw std::runtime_error("Tie case failed");
    }
    Matrix<float> empty("empty",0,3);
    auto e=solver.solve(empty); validate(empty,e);
    Matrix<float> invalid("invalid",1,1);
    for(float bad:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
        invalid.elem(0,0)=bad; bool rejected=false;
        try { solver.solve(invalid); } catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("Non-finite cost accepted");
    }
    std::cout << "PASS: " << cases << " exhaustive min/max cases plus transpose, workspace reuse, large/tied/empty/invalid cases\n";
    return 0;
}
