/*==============================================================================
 *     File: amd.cpp
 *  Created: 2025-04-18 08:53
 *   Author: Bernie Roesler
 *
 *  Description: Implementations for AMD ordering.
 *
 *============================================================================*/

#include <algorithm>  // min, max
#include <cmath>      // sqrt
#include <numeric>    // iota
#include <stdexcept>

#include "amd.h"
#include "csc.h"

namespace cs {


// Helper functions for AMD ordering
static inline csint flip(csint i) {  return -i - 2; }

static csint wclear(csint mark, csint lemax, std::vector<csint>& w, csint N)
{
    if (mark < 2 || (mark + lemax < 0)) {
        for (csint k = 0; k < N; k++) {
            if (w[k] != 0) {
                w[k] = 1;
            }
        }
        mark = 2;
    }
    return mark;  // at this point, w [0..N-1] < mark holds
}


// Compute the C matrix for AMD ordering
CSCMatrix build_graph(const CSCMatrix& A, const AMDOrder order, csint dense)
{
    CSCMatrix C;

    auto [M, N] = A.shape();
    bool values = false;  // symbolic transposes
    CSCMatrix AT = A.transpose(values);

    switch (order) {
        case AMDOrder::Natural:
            // NOTE should never get here since amd returns early
            C = A;  // natural ordering (no permutation)
            break;
        case AMDOrder::APlusAT:
            if (M != N) {
                throw std::runtime_error("Matrix must be square for APlusAT!");
            }
            C = A + AT;
            break;
        case AMDOrder::ATANoDenseRows: {
            // Drop dense columns from AT
            csint q = 0;
            for (csint j = 0; j < M; j++) {
                csint p = AT.p_[j];  // column j of AT starts here
                AT.p_[j] = q;        // new column j starts here
                if (AT.p_[j+1] - p > dense) {
                    continue;        // skip dense col j
                }
                // Copy non-dense entries
                for (; p < AT.p_[j+1]; p++) {
                    AT.i_[q++] = AT.i_[p];
                }
            }
            AT.p_[M] = q;     // finalize AT
            C = AT * AT.transpose(values);  // C = A^T * A, without dense rows
            break;
        }
        case AMDOrder::ATA:
            C = A.transpose(values) * A;
            break;
        default:
            throw std::runtime_error("Invalid AMD order specified!");
    }

    // Drop diagonal entries from C (no self-edges in the graph)
    C.fkeep([] (csint i, csint j, double v) { return i != j; });

    csint cnz = C.nnz();
    csint t = cnz + cnz / 5 + 2 * N;  // elbow room for C
    C.realloc(t);

    return C;
}


std::vector<csint> amd(const CSCMatrix& A, const AMDOrder order)
{
    std::vector<csint> p;  // the output vector

    if (order == AMDOrder::Natural) {
        // Natural ordering (no permutation)
        p.resize(A.N_);
        std::iota(p.begin(), p.end(), 0);  // identity permutation
        return p;
    }

    auto [M, N] = A.shape();

    // --- Construct C matrix --------------------------------------------------
    // Fine dense threshold
    csint dense = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(N)));
    dense = std::min(N - 2, dense);

    // Create the working matrix, subsequent operations are done in-place
    CSCMatrix C = build_graph(A, order, dense);

    // --- Allocate result + workspaces ----------------------------------------
    p = std::vector<csint>(N + 1);

    std::vector<csint> len(N + 1);
    for (csint k = 0; k < N; k++) {
        len[k] = C.p_[k+1] - C.p_[k];  // length of adjacency list
    }
    len[N] = 0;

    std::vector<csint> head(N + 1, -1),   // head of the degree list
                       next(N + 1, -1),   // next node in the degree list
                       last(N + 1, -1),   // previous node in the degree list
                       hhead(N + 1, -1),  // head of the hash list
                       nv(N + 1, 1),      // node i is just 1 node
                       w(N + 1, 1),       // node i is alive
                       elen(N + 1),       // Ek of node i is empty
                       degree {len};      // degree of node i

    csint mark = wclear(0, 0, w, N);  // clear w
    elen[N] = -2;  // N is a dead element
    C.p_[N] = -1;  // N is a root of assembly tree
    w[N] = 0;      // N is a dead element

    // --- Initialize degree lists ---------------------------------------------
    csint nel = 0;  // number of empty nodes
    for (csint i = 0; i < N; i++) {
        csint d = degree[i];
        if (d == 0) {
            elen[i] = -2;  // node i is empty
            nel++;
            C.p_[i] = -1;  // i is a root of the assembly tree
            w[i] = 0;
        } else if (d > dense) {  // node i is dense
            nv[i] = 0;     // absorb i into element n
            elen[i] = -1;  // node i is dead
            nel++;
            C.p_[i] = flip(N);
            nv[N]++;
        } else {
            if (head[d] != -1) {
                last[head[d]] = i;
            }
            next[i] = head[d];  // put node i in degree list d
            head[d] = i;
        }
    }

    // --- Main Loop -----------------------------------------------------------
    // while (nel < N)
    // {
    // }

    return p;
}



}  // namespace cs

/*==============================================================================
 *============================================================================*/
