/*==============================================================================
 *     File: fillreducing.cpp
 *  Created: 2025-04-18 08:53
 *   Author: Bernie Roesler
 *
 *  Description: Implementations for AMD ordering.
 *
 *============================================================================*/

#include <algorithm>  // min, max
#include <cmath>      // sqrt
#include <numeric>    // iota, accumulate
#include <optional>
#include <ranges>     // reverse
#include <stdexcept>
#include <tuple>      // tie
#include <vector>

#include "types.h"
#include "fillreducing.h"
#include "csc.h"
#include "cholesky.h"  // tdfs
#include "solve.h"     // dfs
#include "utils.h"     // randperm

namespace cs {


// Helper functions for AMD ordering

/** Flip the sign of an integer
 * 
 * This function flips the sign of an integer `i` and returns the result.
 * It is used to mark elements as dead in the AMD algorithm.
 *
 * @param i  the integer to flip
 *
 * @return the flipped integer
 */
static inline csint flip(csint i) {  return -i - 2; }


/** Clear the workspace
 * 
 * This function clears the workspace `w` and returns the updated mark.
 * If `mark` is less than 2 or if `mark + lemax` is less than 0, it clears
 * the workspace and sets `mark` to 2.
 *
 * @param mark  the current mark
 * @param lemax  the maximum length of the workspace
 * @param w  the workspace vector
 * @param N  the size of the matrix
 *
 * @return the updated mark
 */
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
            // NOTE should never get here since amd returns early, but if it
            // does, return the original matrix without values
            C = CSCMatrix {{}, A.indices(), A.indptr(), A.shape()};
            break;
        case AMDOrder::APlusAT:
            if (M != N) {
                throw std::runtime_error("Matrix must be square for APlusAT!");
            }
            C = A + AT;
            break;
        case AMDOrder::ATANoDenseRows: {
            // Drop dense columns from AT (i.e., rows from A)
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

            AT.p_[M] = q;   // finalize AT
            AT.realloc(q);  // resize AT to remove dense rows

            C = AT * AT.transpose(values);  // C = A^T * A, without dense rows
            break;
        }
        case AMDOrder::ATA:
            C = AT * A;
            break;
        default:
            throw std::runtime_error("Invalid AMD order specified!");
    }

    // Drop diagonal entries from C (no self-edges in the graph)
    C.fkeep([] (csint i, csint j, [[maybe_unused]] double v) { return i != j; });

    return C;
}


std::vector<csint> amd(const CSCMatrix& A, const AMDOrder order)
{
    std::vector<csint> P;  // the output vector

    if (order == AMDOrder::Natural) {
        // Natural ordering (no permutation)
        P.resize(A.N_);
        std::iota(P.begin(), P.end(), 0);  // identity permutation
        return P;
    }

    auto [M, N] = A.shape();

    // --- Construct C matrix --------------------------------------------------
    // Find dense threshold
    csint dense = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(N)));
    dense = std::min(N - 2, dense);

    // Create the working matrix, subsequent operations are done in-place
    CSCMatrix C = build_graph(A, order, dense);

    // Allocate elbow room for C
    csint cnz = C.nnz();
    csint t = cnz + cnz / 5 + 2 * N;
    C.realloc(t);

    // --- Allocate result + workspaces ----------------------------------------
    std::vector<csint> len(N + 1);

    for (csint k = 0; k < N; k++) {
        len[k] = C.p_[k+1] - C.p_[k];  // length of adjacency list
    }

    len[N] = 0;

    std::vector<csint> head(N+1, -1),   // head of the degree list
                       next(N+1, -1),   // next node in the degree list
                       last(N+1, -1),   // previous node in the degree list
                       hhead(N+1, -1),  // head of the hash list
                       nv(N+1, 1),      // node i is just 1 node
                       w(N+1, 1),       // node i is alive
                       elen(N+1),       // Ek of node i is empty
                       degree {len};    // degree of node i

    csint lemax = 0;
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
    csint mindeg = 0;  // track minimum degree over all iterations

    while (nel < N) {  // while selecting pivots
        // --- Select node of minimum approximate degree -----------------------
        csint k = -1;
        while (mindeg < N) {
            k = head[mindeg];
            if (k != -1) {
                break;
            }
            mindeg++;
        }

        if (next[k] != -1) {
            last[next[k]] = -1;
        }

        head[mindeg] = next[k];  // remove k from degree list
        csint elenk = elen[k];   // elenk = |Ek|
        csint nvk = nv[k];       // # of nodes k represents
        nel += nvk;              // nv[k] nodes of A eliminated

        // --- Garbage collection ----------------------------------------------
        if (elenk > 0 && cnz + mindeg >= C.nzmax()) {
            for (csint j = 0; j < N; j++) {
                csint p = C.p_[j];
                if (p >= 0) {           // j is a live node or element
                    C.p_[j] = C.i_[p];  // save first entry of object
                    C.i_[p] = flip(j);  // first entry is now flip(j)
                }
            }

            csint p = 0,
                  q = 0;
            while (p < cnz) {               // scan all of memory
                csint j = flip(C.i_[p++]);  // found object j
                if (j >= 0) {
                    C.i_[q] = C.p_[j];      // restore first entry of object
                    C.p_[j] = q++;          // new pointer to object j
                    for (csint k3 = 0; k3 < len[j] - 1; k3++) {
                        C.i_[q++] = C.i_[p++];
                    }
                }
            }

            cnz = q;  // C.i_[cnz...C.nzmax()-1] now free
        }

        // --- Construct new element -------------------------------------------
        csint dk = 0;
        nv[k] = -nvk;  // flag k as in Lk
        csint p = C.p_[k];
        csint pk1 = (elenk == 0) ? p : cnz;  // do in place if elen[k] == 0
        csint pk2 = pk1;

        for (csint k1 = 1; k1 <= elenk + 1; k1++) {
            csint e, pj, ln;

            if (k1 > elenk) {
                e = k;                // search the nodes in k
                pj = p;               // list of nodes starts at C.i_[pj]
                ln = len[k] - elenk;  // length of list of nodes in k
            } else {
                e = C.i_[p++];        // search the nodes in e
                pj = C.p_[e];
                ln = len[e];          // length of list of nodes in e
            }
            
            for (csint k2 = 1; k2 <= ln; k2++) {
                csint i = C.i_[pj++];
                csint nvi = nv[i];
                if (nvi <= 0) {
                    continue;                 // node i dead, or seen
                }
                dk += nvi;                    // degree[Lk] += size of node i
                nv[i] = -nvi;                 // negate nv[i] to denote i in Lk
                C.i_[pk2++] = i;              // place i in Lk

                if (next[i] != -1) {
                    last[next[i]] = last[i];
                }

                if (last[i] != -1) {          // remove i from degree list
                    next[last[i]] = next[i];
                } else {
                    head[degree[i]] = next[i];
                }
            }  // for k2

            if (e != k) {
                C.p_[e] = flip(k);  // absorb e into k
                w[e] = 0;           // e is now a dead element
            }
        }  // for k1

        if (elenk != 0) {
            cnz = pk2;              // C.i_[cnz...C.nzmax()] is free
        }

        degree[k] = dk;             // external degree of k - |Lk\i|
        C.p_[k] = pk1;              // element k is in C.i_[pk1..pk2-1]
        len[k] = pk2 - pk1;         // length of adjacency list of element k
        elen[k] = -2;               // k is now an element

        // --- Find set differences --------------------------------------------
        mark = wclear(mark, lemax, w, N);  // clear w if necessary

        for (csint pk = pk1; pk < pk2; pk++) {   // scan 1 : find |Le \ Lk|
            csint i = C.i_[pk];
            csint eln = elen[i];
            if (eln <= 0) {
                continue;                 // skip if elen[i] empty
            }
            csint nvi = -nv[i];          // nv[i] was negated
            csint wnvi = mark - nvi;
            for (csint p = C.p_[i]; p <= C.p_[i] + eln - 1; p++) {  // scan Ei
                csint e = C.i_[p];
                if (w[e] >= mark) {
                    w[e] -= nvi;              // decrement |Le \ Lk|
                } else if (w[e] != 0) {       // ensure e is a live element
                    w[e] = degree[e] + wnvi;  // 1st time e seen in scan 1
                }
            }
        }  // scan1

        // --- Degree Update ---------------------------------------------------
        for (csint pk = pk1; pk < pk2; pk++) {  // scan2: degree update
            csint i = C.i_[pk];
            csint p1 = C.p_[i];
            csint p2 = p1 + elen[i] - 1;
            csint pn = p1;

            csint h = 0;
            csint d = 0;
            for (csint p = p1; p <= p2; p++) {  // scan Ei
                csint e = C.i_[p];
                if (w[e] != 0) {  // e is an unabsorbed element
                    csint dext = w[e] - mark;  // dext = |Le \ Lk|
                    if (dext > 0) {
                        d += dext;          // sum up the set differences
                        C.i_[pn++] = e;     // keep e in Ei
                        h += e;             // compute the hash of node i
                    } else {
                        C.p_[e] = flip(k);  // aggressive absorb. e -> k
                        w[e] = 0;           // e is a dead element
                    }
                }
            }

            elen[i] = pn - p1 + 1;  // elen[i] = |Ei|
            csint p3 = pn;
            csint p4 = p1 + len[i];

            for (csint p = p2 + 1; p < p4; p++) {  // prune edges in Ai
                csint j = C.i_[p];
                csint nvj = nv[j];
                if (nvj <= 0) {
                    continue;    // node j dead or in Lk
                }
                d += nvj;        // degree(i) += |j|
                C.i_[pn++] = j;  // place j in node list of i
                h += j;          // compute hash for node i
            }

            if (d == 0) {                    // check for mass elimination
                C.p_[i] = flip(k);           // absorb i into k
                csint nvi = -nv[i];          // restore nv[i]
                dk -= nvi;                   // |Lk| -= |i|
                nvk += nvi;                  // |k| += nv[i]
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1;                // node i is dead
            } else {
                degree[i] = std::min(degree[i], d);  // update degree(i)
                C.i_[pn] = C.i_[p3];         // move first node to end
                C.i_[p3] = C.i_[p1];         // move 1st el. to end of Ei
                C.i_[p1] = k;                // add k as 1st element of Ei
                len[i] = pn - p1 + 1;        // new len of adj. list of node i
                h = ((h < 0) ? -h : h) % N;  // finalize hash of i
                next[i] = hhead[h];          // place i in hash bucket
                hhead[h] = i;
                last[i] = h;                 // save hash of i in last[i]
            }
        }  // scan2

        degree[k] = dk;  // finalize |Lk|
        lemax = std::max(lemax, dk);
        mark = wclear(mark + lemax, lemax, w, N);  // clear w

        // --- Supernode detection ---------------------------------------------
        for (csint pk = pk1; pk < pk2; pk++) {
            csint i = C.i_[pk];
            if (nv[i] >= 0) {
                continue;  // skip if i is dead
            }
            csint h = last[i];  // scan hash bucket of node i
            i = hhead[h];
            hhead[h] = -1;  // hash bucket will be empty
            while (i != -1 && next[i] != -1) {
                csint ln = len[i];
                csint eln = elen[i];

                for (csint p = C.p_[i] + 1; p <= C.p_[i] + ln - 1; p++) {
                    w[C.i_[p]] = mark;  // mark the nodes in i
                }

                csint jlast = i;
                csint j = next[i];

                while (j != -1) {  // compare i with all j
                    bool ok = (len[j] == ln) && (elen[j] == eln);

                    for (csint p = C.p_[j] + 1; ok && p <= C.p_[j] + ln - 1; p++) {
                        if (w[C.i_[p]] != mark) {
                            ok = false;  // compare i and j
                        }
                    }

                    if (ok) {
                        C.p_[j] = flip(i);  // absorb j into i
                        nv[i] += nv[j];
                        nv[j] = 0;
                        elen[j] = -1;       // node j is dead
                        j = next[j];        // delete j from hash bucket
                        next[jlast] = j;
                    } else {
                        jlast = j;          // j and i are different
                        j = next[j];
                    }
                }

                i = next[i];
                mark++;
            }
        }  // supernode detection

        // --- Finalize new element -------------------------------------------
        p = pk1;
        for (csint pk = pk1; pk < pk2; pk++) {  // finalize Lk
            csint i = C.i_[pk];
            csint nvi = -nv[i];
            if (nvi <= 0) {
                continue;                    // skip if i is dead
            }
            nv[i] = nvi;                     // restore nv[i]
            csint d = degree[i] + dk - nvi;  // compute external degree(i)
            d = std::min(d, N - nel - nvi);
            if (head[d] != -1) {
                last[head[d]] = i;
            }
            next[i] = head[d];               // put i back in degree list
            last[i] = -1;
            head[d] = i;
            mindeg = std::min(mindeg, d);    // find new minimum degree
            degree[i] = d;
            C.i_[p++] = i;                   // place i in Lk
        }

        nv[k] = nvk;  // # nodes absorbed into k

        if ((len[k] = p - pk1) == 0) {  // length of adj list of element k
            C.p_[k] = -1;  // k is a root of the tree
            w[k] = 0;      // k is now a dead element
        }

        if (elenk != 0) {
            cnz = p;  // free unuzed space in Lk
        }
    }  // while selecting pivots

    // --- Postordering --------------------------------------------------------
    for (csint i = 0; i < N; i++) {
        C.p_[i] = flip(C.p_[i]);  // fix assembly tree
    }

    std::fill(head.begin(), head.end(), -1);

    // Place unordered nodes in lists
    for (csint j = N; j >= 0; j--) {
        if (nv[j] > 0) {
            continue;  // skip if j is an element
        }
        next[j] = head[C.p_[j]];  // place j in list of its parent
        head[C.p_[j]] = j;
    }

    // Place elements in lists
    for (csint e = N; e >= 0; e--) {
        if (nv[e] <= 0) {
            continue;  // skip unless e is an element
        }
        if (C.p_[e] != -1) {
            next[e] = head[C.p_[e]];  // place e in list of its parent
            head[C.p_[e]] = e;
        }
    }

    // Postorder the assembly tree
    P.reserve(N + 1);
    for (csint i = 0; i <= N; i++) {
        if (C.p_[i] == -1) {
            tdfs(i, head, next, P);
        }
    }

    // Only return the first N elements of P
    return std::vector<csint>(P.cbegin(), P.cbegin() + N);
}


// TODO rewrite with explicit stacks created in the function
void augment(
    csint k,
    const CSCMatrix& A,
    std::vector<csint>& jmatch,
    std::vector<csint>& cheap,
    std::vector<csint>& w,
    std::vector<csint>& js,
    std::vector<csint>& is,
    std::vector<csint>& ps
)
{
    bool found = false;
    csint head = 0;
    js[0] = k;  // start with just node k in jstack

    while (head >= 0) {
        // --- Start (or continue) depth-first-search at node j ----------------
        csint j = js[head];      // get j from top of jstack
        if (w[j] != k) {         // 1st time j visited for kth path
            w[j] = k;            // mark j as visited for kth path
            csint i = -1;

            csint p;
            for (p = cheap[j]; p < A.p_[j+1]; p++) {
                i = A.i_[p];     // try a cheap assignment (i,j)
                found = (jmatch[i] == -1);
                if (found) {
                    break;
                }
            }

            cheap[j] = p;        // start here next time j is traversed

            if (found) {
                is[head] = i;    // column j matched with row i
                break;           // end of augmenting path
            }

            ps[head] = A.p_[j];  // no cheap match: start dfs for j
        }

        // --- Depth-first-search of neighbors of j ----------------------------
        csint p;
        for (p = ps[head]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];        // consider row i
            if (w[jmatch[i]] == k) {
                continue;             // skip jmatch [i] if marked
            }
            ps[head] = p + 1;         // pause dfs of node j
            is[head] = i;             // i will be matched with j if found
            js[++head] = jmatch[i];   // start dfs at column jmatch [i]
            break;
        }

        if (p == A.p_[j+1]) {
            head--;                   // node j is done; pop from stack
        }
    }

    if (found) {
        for (csint p = head; p >= 0; p--) {
            jmatch[is[p]] = js[p];    // augment the match
        }
    }
}


// -----------------------------------------------------------------------------
//         Maximum matching
// -----------------------------------------------------------------------------
namespace detail {

bool augment_r(
    csint k,
    const CSCMatrix& A,
    std::vector<csint>& jmatch,
    std::vector<csint>& cheap,
    std::vector<csint>& w,
    csint j
)
{
    bool found = false;

    // --- Start depth-first-search at node j -------------------------------
    w[j] = k;  // mark j as visited for kth path

    csint p = -1,
          i = -1;

    for (p = cheap[j]; p < A.p_[j+1] && !found; p++) {
        i = A.i_[p];  // try a cheap assignment (i,j)
        found = (jmatch[i] == -1);
    }

    cheap[j] = p;  // start here next time j is traversed

    // --- Depth-first-search of neighbors of j -----------------------------
    for (p = A.p_[j]; p < A.p_[j+1] && !found; p++) {
        i = A.i_[p];  // consider row i

        if (w[jmatch[i]] == k) {
            continue;  // skip jmatch[i] if marked
        }

        // Recursively search for an augmenting path
        found = augment_r(k, A, jmatch, cheap, w, jmatch[i]);
    }

    if (found) {
        jmatch[i] = j;  // augment jmatch if path found
    }

    return found;
}


MaxMatch maxtrans_r(const CSCMatrix& A, [[maybe_unused]] csint seed)
{
    auto [M, N] = A.shape();

    MaxMatch jimatch(M, N, -1);  // allocate result
    auto& [jmatch, imatch] = jimatch;  // reference to jmatch and imatch

    std::vector<csint> w(N, -1),  // mark all nodes as unvisited
                       cheap(A.p_);  // cheap assignment

    for (csint k = 0; k < N; k++) {
        augment_r(k, A, jmatch, cheap, w, k);
    }
    
    // imatch is the inverse of jmatch
    for (csint i = 0; i < M; i++) {
        if (jmatch[i] >= 0) {
            imatch[jmatch[i]] = i;
        }
    }

    return jimatch;
}

}  // namespace detail


MaxMatch maxtrans(const CSCMatrix& A, csint seed)
{
    auto [M, N] = A.shape();

    MaxMatch jimatch(M, N, -1);  // allocate result

    // count non-empty rows and columns
    std::vector<csint> w(M); // workspace
    csint k = 0;
    csint n2 = 0;

    for (csint j = 0; j < N; j++) {
        n2 += (A.p_[j] < A.p_[j+1]);
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            w[A.i_[p]] = 1;
            k += (j == A.i_[p]);  // count entries already on diagonal
        }
    }

    if (k == std::min(M, N)) {  // quick return if diagonal zero-free
        std::iota(jimatch.jmatch.begin(), jimatch.jmatch.begin() + k, 0);
        std::iota(jimatch.imatch.begin(), jimatch.imatch.begin() + k, 0);
        return jimatch;
    }

    csint m2 = std::accumulate(w.cbegin(), w.cend(), 0);  // count non-empty rows

    // transpose if needed
    const CSCMatrix C = (m2 < n2) ? A.transpose(false) : A;
    M = C.M_;
    N = C.N_;

    // If we transposed, we need to swap the imatch and jmatch vectors
    std::vector<csint>& jmatch = (m2 < n2) ? jimatch.imatch : jimatch.jmatch;
    std::vector<csint>& imatch = (m2 < n2) ? jimatch.jmatch : jimatch.imatch;

    // Allocate workspaces
    w.resize(N);
    std::fill(w.begin(), w.end(), -1);  // mark all nodes as unvisited

    std::vector<csint> cheap(C.p_),  // cheap assignment
                       is(N),        // row indices stack
                       js(N),        // col indices stack
                       ps(N);        // pause stack for DFS in augment

    // randperm can help with worst-case behavior O(|A|N), see Davis, p 118.
    std::vector<csint> q = randperm(N, seed);  // random permutation of columns

    // augment the path, starting at column q[k]
    for (csint k = 0; k < N; k++) {
        augment(q[k], C, jmatch, cheap, w, js, is, ps);
    }

    std::fill(imatch.begin(), imatch.end(), -1);  // find row match
    for (csint i = 0; i < M; i++) {
        if (jmatch[i] >= 0) {
            imatch[jmatch[i]] = i;
        }
    }

    return jimatch;
}


// -----------------------------------------------------------------------------
//         Strongly-connected components
// -----------------------------------------------------------------------------
SCCResult scc(const CSCMatrix& A)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    SCCResult D(M);  // allocate result

    const CSCMatrix AT = A.transpose(false);  // symbolic transpose

    std::vector<char> marked(N, false);  // mark visited nodes
    std::vector<csint> xi, pstack, rstack;  // stacks for DFS
    xi.reserve(N);
    pstack.reserve(N);
    rstack.reserve(N);

    // ----- DFS through all of A
    for (csint i = 0; i < N; i++) {
        if (!marked[i]) {
            dfs(A, i, marked, xi, pstack, rstack);
        }
    }

    // ----- DFS through A^T
    std::fill(marked.begin(), marked.end(), false);  // clear marks

    // get i in reverse order of finish time
    for (const auto& i : std::views::reverse(xi)) {
        if (!marked[i]) {
            D.r.push_back(N - D.p.size());  // node i is the start of a block
            dfs(AT, i, marked, D.p, pstack, rstack);
        }
    }

    D.r.push_back(0);  // first block starts at zero

    // reverse the order of the blocks and nodes since dfs returns in reverse
    std::reverse(D.r.begin(), D.r.end());
    std::reverse(D.p.begin(), D.p.end());

    D.Nb = D.r.size() - 1;  // number of strongly connected components

    // ----- Sort each block in natural order
    // Number each node by its block number
    std::vector<csint> Blk(N);
    for (csint b = 0; b < D.Nb; b++) {
        for (csint k = D.r[b]; k < D.r[b+1]; k++) {
            Blk[D.p[k]] = b;
        }
    }

    // Sort the indices of each block
    std::vector<csint> rcopy = D.r;  // pointers to start of blocks
    for (csint i = 0; i < N; i++) {
        D.p[rcopy[Blk[i]]++] = i;
    }

    return D;
}


// --- Dulmage-Mendelsohn Permutation ----------------------------------------
// TODO rewrite with explicit queues created in the function?
void bfs(
    const CSCMatrix& A,
    csint N,
    std::vector<csint>& wi,
    std::vector<csint>& wj,
    std::vector<csint>& queue,
    const std::vector<csint>& imatch,
    const std::vector<csint>& jmatch,
    csint mark
)
{
    csint head = 0;
    csint tail = 0;

    // Place all unmatched nodes in queue
    for (csint j = 0; j < N; j++) {
        if (imatch[j] < 0) {    // skip j if matched
            wj[j] = 0;          // j in set C0 (R0 if transpose)
            queue[tail++] = j;  // place unmatched col j in queue
        }
    }

    if (tail == 0) {
        return;  // no unmatched nodes
    }

    const CSCMatrix& C = (mark == 1) ? A : A.transpose(false);

    // BFS loop
    while (head < tail) {
        csint j = queue[head++];  // get j from front of queue

        for (csint p = C.p_[j]; p < C.p_[j+1]; p++) {
            csint i = C.i_[p];     // consider row i

            if (wi[i] >= 0) {
                continue;          // skip if i is marked
            }

            wi[i] = mark;          // i in set R1 (C3 if transpose)
            csint j2 = jmatch[i];  // transverse alternating path to j2

            if (wj[j2] >= 0) {
                continue;          // skip if j2 is marked
            }

            wj[j2] = mark;         // j2 in set C1 (R3 if transpose)
            queue[tail++] = j2;    // add j2 to queue
        }
    }

    return;
}


static void matched(
    csint N,
    const std::vector<csint>& wj,
    const std::vector<csint>& imatch,
    std::vector<csint>& p,
    std::vector<csint>& q,
    std::array<csint, 5>& cc,
    std::array<csint, 5>& rr,
    csint set,
    csint mark
)
{
    csint kc = cc[set],
          kr = rr[set-1];
    for (csint j = 0; j < N; j++) {
        if (wj[j] == mark) {  // skip if j is not in C set
            p[kr++] = imatch[j];
            q[kc++] = j;
        }
    }
    cc[set+1] = kc;
    rr[set] = kr;
}


static void unmatched(
    csint M,
    const std::vector<csint>& wi,
    std::vector<csint>& p,
    std::array<csint, 5>& rr,
    csint set
)
{
    csint kr = rr[set];
    for (csint i = 0; i < M; i++) {
        if (wi[i] == 0) {
            p[kr++] = i;
        }
    }
    rr[set+1] = kr;
}


static void gather_scatter(
    std::vector<csint>& source,
    std::vector<csint>& temp,
    const std::vector<csint>& ps,
    csint nc,
    csint offset
)
{
    for (csint k = 0; k < nc; k++) {
        temp[k] = source[ps[k] + offset];
    }
    std::copy(temp.cbegin(), temp.cbegin() + nc, source.begin() + offset);
}


// Dulmage-Mendelsohn Permutation
DMPermResult dmperm(const CSCMatrix& A, csint seed)
{
    // --- Maximum Matching ----------------------------------------------------
    auto [M, N] = A.shape();

    DMPermResult D(M, N);  // allocate result

    auto [jmatch, imatch] = maxtrans(A, seed);  // maximum matching

    // --- Coarse Decomposition ------------------------------------------------
    // CSparse uses wi = D.r and wj = D.s as workspace
    std::vector<csint> wi(M + 6, -1),  // unmark all row and columns for bfs
                       wj(N + 6, -1);

    bfs(A, N, wi, wj, D.q, imatch, jmatch, 1);  // find C1, R1 from C0
    bfs(A, M, wj, wi, D.p, jmatch, imatch, 3);  // find C3, R3 from R0

    unmatched(N, wj, D.q, D.cc, 0);  // unmatched set C0
    matched(N, wj, imatch, D.p, D.q, D.cc, D.rr, 1,  1);  // set R1 and C1
    matched(N, wj, imatch, D.p, D.q, D.cc, D.rr, 2, -1);  // set R2 and C2
    matched(N, wj, imatch, D.p, D.q, D.cc, D.rr, 3,  3);  // set R3 and C3
    unmatched(M, wi, D.p, D.rr, 3);  // unmatched set R0

    // --- Fine decomposition --------------------------------------------------
    std::vector<csint> p_inv = inv_permute(D.p);

    // C = A(p, q) will hold A(R2, C2)
    CSCMatrix C = A.permute(p_inv, D.q, false);

    // delete cols C0, C1, and C3 from C
    csint nc = D.cc[3] - D.cc[2];

    if (D.cc[2] > 0) {
        for (csint j = D.cc[2]; j <= D.cc[3]; j++) {
            C.p_[j - D.cc[2]] = C.p_[j];
        }
    }

    C.N_ = nc;  // update cols

    // Delete rows R0, R1, and R3 from C
    if (D.rr[2] - D.rr[1] < M) {
        C.fkeep(
            [D](csint i, [[maybe_unused]] csint j, [[maybe_unused]] double v) {
                return i >= D.rr[1] && i < D.rr[2];  // true if row i is in R2
            }
        );

        csint cnz = C.p_[nc];

        if (D.rr[1] > 0) {
            for (csint k = 0; k < cnz; k++) {
                C.i_[k] -= D.rr[1];
            }
        }
    }

    C.M_ = nc;  // update rows

    // Find strongly connected components
    SCCResult strong_cc = scc(C);

    // --- Combine coarse and fine decompositions ------------------------------
    // C(scc.p, scc.p) is the permuted matrix
    // kth block is scc.r[k]..r[k+1]-1
    // scc.Nb is the number of blocks of A(R2, C2)
    std::vector<csint>& ps = strong_cc.p;
    std::vector<csint>& rs = strong_cc.r;
    gather_scatter(D.q, wj, ps, nc, D.cc[2]);
    gather_scatter(D.p, wi, ps, nc, D.rr[1]);

    // Create the fine block partitions
    csint nb1 = strong_cc.Nb;
    csint nb2 = 0;

    D.r[0] = 0;
    D.s[0] = 0;

    // Leading coarse block A(R1, [C0 C1])
    if (D.cc[2] > 0) {
        nb2++;
    }

    // Coarse block A(R2, C2)
    for (csint k = 0; k< nb1; k++) {
        D.r[nb2] = rs[k] + D.rr[1];  // A(R2, C2) splits into nb1 fine blocks
        D.s[nb2] = rs[k] + D.cc[2];
        nb2++;
    }

    // Trailing coarse block A([R3 R0], C3)
    if (D.rr[2] < M) {
        D.r[nb2] = D.rr[2];
        D.s[nb2] = D.cc[3];
        nb2++;
    }

    D.r[nb2] = M;
    D.s[nb2] = N;
    D.Nb = nb2;

    // Reallocate the result
    D.r.resize(D.Nb + 1);
    D.s.resize(D.Nb + 1);
    D.r.shrink_to_fit();
    D.s.shrink_to_fit();

    return D;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
