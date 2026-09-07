#pragma once
#include "../cpp/juzhen.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Juzhen {
struct AssignmentResult {
    // -1 marks unmatched vertices of a rectangular problem.
    std::vector<int> row_to_col, col_to_row;
    double cost = 0;
};

// Exact dense linear assignment via shortest augmenting paths (Hungarian
// primal-dual formulation). O(min(rows,cols)^2 * max(rows,cols)) time.
// CPU only: pass a Matrix<float>, or explicitly call gpu_matrix.to_host().
// Retains scratch storage between calls; one solver per concurrent thread.
class LinearAssignmentSolver {
    std::vector<double> costs_, u_, v_, slack_;
    std::vector<int> owner_, previous_;
    std::vector<unsigned char> visited_;
public:
    AssignmentResult solve(const Matrix<float>& input, bool maximize = false) {
        const size_t rows=input.num_row(), cols=input.num_col();
        if (rows >= size_t(std::numeric_limits<int>::max()) ||
            cols >= size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Assignment dimensions exceed integer index range");
        AssignmentResult result;
        result.row_to_col.assign(rows,-1); result.col_to_row.assign(cols,-1);
        if (!rows || !cols) return result;
        const bool transposed=rows>cols;
        const int n=static_cast<int>(std::min(rows,cols));
        const int m=static_cast<int>(std::max(rows,cols));
        if (size_t(n)>costs_.max_size()/size_t(m)) throw std::length_error("Assignment matrix too large");
        costs_.resize(size_t(n)*m);
        // Materialize once in the row-contiguous layout used by the search.
        // elem() respects Juzhen's lazy transpose flag.
        for (int i=0;i<n;++i) for (int j=0;j<m;++j) {
            double c=transposed ? input.elem(j,i) : input.elem(i,j);
            if (!std::isfinite(c)) throw std::invalid_argument("Assignment requires finite costs");
            costs_[size_t(i)*m+j]=maximize ? -c : c;
        }
        u_.assign(n+1,0); v_.assign(m+1,0); owner_.assign(m+1,0);
        slack_.resize(m+1); previous_.resize(m+1); visited_.resize(m+1);
        for (int row=1;row<=n;++row) {
            owner_[0]=row;
            std::fill(slack_.begin(),slack_.end(),std::numeric_limits<double>::infinity());
            std::fill(visited_.begin(),visited_.end(),0);
            int column=0;
            do {
                visited_[column]=1;
                int current=owner_[column], next=0;
                double delta=std::numeric_limits<double>::infinity();
                const double* values=costs_.data()+size_t(current-1)*m;
                for (int j=1;j<=m;++j) if (!visited_[j]) {
                    double reduced=values[j-1]-u_[current]-v_[j];
                    if (reduced<slack_[j]) { slack_[j]=reduced; previous_[j]=column; }
                    // Prefer a free column on ties to avoid unnecessary paths.
                    if (slack_[j]<delta || (slack_[j]==delta && owner_[j]==0)) {
                        delta=slack_[j]; next=j;
                    }
                }
                for (int j=0;j<=m;++j) {
                    if (visited_[j]) { u_[owner_[j]]+=delta; v_[j]-=delta; }
                    else slack_[j]-=delta;
                }
                column=next;
            } while (owner_[column]!=0);
            do {
                int predecessor=previous_[column];
                owner_[column]=owner_[predecessor]; column=predecessor;
            } while (column!=0);
        }
        for (int j=1;j<=m;++j) if (owner_[j]) {
            int row=transposed ? j-1 : owner_[j]-1;
            int col=transposed ? owner_[j]-1 : j-1;
            result.row_to_col[row]=col; result.col_to_row[col]=row;
            result.cost+=input.elem(row,col);
        }
        return result;
    }
};
} // namespace Juzhen
