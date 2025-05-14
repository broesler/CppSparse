/*==============================================================================
 *     File: qr.cpp
 *  Created: 2025-02-11 12:52
 *   Author: Bernie Roesler
 *
 *  Description: Implements QR decomposition using Householder reflections and
 *    Givens rotations.
 *
 *============================================================================*/

#include <numeric>  // accumulate
#include <ranges>   // views::reverse, span
#include <vector>
#include <optional>

#include "cholesky.h"  // etree, post
#include "qr.h"
#include "utils.h"

namespace cs {


inline auto sign(double x) { return std::copysign(1.0, x); }


Householder house(std::span<const double> x)
{
    double beta, s, sigma = 0.0;
    std::vector<double> v(x.begin(), x.end());  // copy x into v

    // sigma is the sum of squares of all elements *except* the first
    for (csint i = 1; i < v.size(); i++) {
        sigma += v[i] * v[i];
    }

    if (sigma == 0) {  // x is already a multiple of e1
        //---------- LAPACK DLARFG algorithm
        s = v[0];  // H is the identity, so Hx = x
        beta = 0;  // LAPACK DLARFG: H is just the identity
        v[0] = 1;
        //---------- Davis book code (cs_house)
        // s = std::fabs(v[0]);         // s = |x(0)|
        // beta = (v[0] <= 0) ? 2 : 0;  // make direction positive if x(0) < 0
        // v[0] = 1;
    } else {
        //---------- LAPACK DLARFG algorithm
        // consistent with scipy.linalg.qr(mode='raw') and MATLAB
        // LAPACK uses the notation: tau := beta, beta := s
        double norm_x = std::sqrt(v[0] * v[0] + sigma);
        double alpha = v[0];
        s = -sign(alpha) * norm_x;
        beta = (s - alpha) / s;

        v[0] = 1;
        for (csint i = 1; i < v.size(); i++) {
            v[i] /= (alpha - s);
        }

        //---------- Davis book code (cs_house)
        // s = std::sqrt(v[0] * v[0] + sigma);  // s = norm(x)
        // v[0] = (v[0] <= 0) ? (v[0] - s) : (-sigma / (v[0] + s));
        // beta = -1 / (s * v[0]);  // Davis book code

        // Scale to be self-consistent with v[0] = 1.
        // Matches cs_qr when we normalize V and beta after the call.
        // double v0 = v[0];  // cache value before we change it to 1.0
        // beta *= v0 * v0;   // works with Davis book code + v[0] = 1 scaling

        //---------- Golub & Van Loan (Algorithm 5.1.1) (3 or 4ed)
        // Gives same result as the beta from Davis book, scaled by v0**2.
        // beta = 2 * (v0 * v0) / (v0 * v0 + sigma);

        // normalize to v[0] == 1
        // for (auto& vi : v) {
        //     vi /= v0;
        // }
    }

    return {v, beta, s};
}


std::vector<double>& happly(
    const CSCMatrix& V,
	csint j,
	double beta,
	std::vector<double>& x
)
{
    double tau = 0.0;

    // tau = v^T x
    for (csint p = V.p_[j]; p < V.p_[j+1]; p++) {
        tau += V.v_[p] * x[V.i_[p]];
    }

    tau *= beta;  // tau = beta * v^T x

    // Hx = x - v*tau
    for (csint p = V.p_[j]; p < V.p_[j+1]; p++) {
        x[V.i_[p]] -= V.v_[p] * tau;
    }

    return x;
}


std::vector<csint> find_leftmost(const CSCMatrix& A)
{
    std::vector<csint> leftmost(A.M_, -1);

    for (csint k = A.N_ - 1; k >= 0; k--) {
        for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
            leftmost[A.i_[p]] = k;  // leftmost[i] = min(find(A(i, :)))
        }
    }

    return leftmost;
}


void vcount(const CSCMatrix& A, SymbolicQR& S)
{
    assert(S.leftmost.size() == A.M_);
    assert(S.parent.size() == A.N_);

    auto [M, N] = A.shape();
    std::vector<csint> next(M),      // the next row index
                       head(N, -1),  // the first row index in each column
                       tail(N, -1),  // the last row index in each column
                       nque(N);      // the number of rows in each column

    S.p_inv.assign(M + N, -1);  // initialize permutation vector

    // Initialize the linked lists for each row with their leftmost index
    for (csint i = M-1; i >= 0; i--) {  // scan rows in reverse order
        csint k = S.leftmost[i];
        if (k != -1) {                  // row i is not empty
            if (nque[k]++ == 0) {
                tail[k] = i;            // first row in queue k
            }
            next[i] = head[k];          // put i at head of queue k
            head[k] = i;
        }
    }

    S.vnz = 0;
    S.m2 = M;

    // List k contains all rows that belong to V(:, k)
    csint k;  // declare outside loop for final row permutation
    for (k = 0; k < N; k++) {          // find row permutation and nnz(V)
        csint i = head[k];             // remove row i from queue k
        S.vnz++;                       // count V(k, k) as nonzero
        if (i < 0) {
            i = S.m2++;                // add a fictitious row
        }
        S.p_inv[i] = k;                // associate row i with V(:, k)
        if (--nque[k] <= 0) {          // skip if V(k+1:m, k) is empty
            continue;
        }
        S.vnz += nque[k];              // nque[k] is nnz(V(k+1:m, k))
        csint pa = S.parent[k];
        if (pa != -1) {                // move all rows to parent of k
            if (nque[pa] == 0) {
                tail[pa] = tail[k];
            }
            next[tail[k]] = head[pa];
            head[pa] = next[i];
            nque[pa] += nque[k];
        }
    }

    for (csint i = 0; i < M; i++) {    // assign any unordered rows to last k
        if (S.p_inv[i] < 0) {
            S.p_inv[i] = k++;
        }
    }

    S.p_inv.resize(S.m2);     // resize to the actual number of rows
    S.p_inv.shrink_to_fit();  // shrink to fit
}


SymbolicQR sqr(const CSCMatrix& A, AMDOrder order, bool use_postorder)
{
    auto [M, N] = A.shape();
    CSCMatrix C = A;

    if (M < N) {
        C = A.slice(0, M, 0, M);  // slice to (M, M)
        N = M;
    }

    CSCMatrix Ac;
    if (use_postorder) {
        Ac = C;  // copy original (sliced) matrix for postordering
    }

    SymbolicQR S;          // allocate result
    std::vector<csint> q;  // column permutation vector

    if (order == AMDOrder::Natural) {
        q.resize(N);
        std::iota(q.begin(), q.end(), 0);  // identity permutation
    } else {
        q = amd(C, order);                 // order = ATA for QR
    }

    // Find pattern of Cholesky factor of A.T @ A
    bool values = false,  // don't copy values
         CTC = true;      // do take the etree/counts of A^T A

    C = C.permute_cols(q, values);

    S.parent = etree(C, CTC);  // etree of C^T C, C = A[:, q]
    std::vector<csint> postorder = post(S.parent);

    // Exercise 5.5 combine the postordering
    if (use_postorder) {
        q = pvec(postorder, q);          // combine the permutations
        C = Ac.permute_cols(q, values);  // apply combined permutation
        S.parent = etree(C, CTC);        // recompute etree
        postorder = post(S.parent);      // recompute postorder
    }

    S.q = q;  // store the column permutation

    // column counts of the Cholesky factor of C^T C
    std::vector<csint> cp = counts(C, S.parent, postorder, CTC);
    S.rnz = std::accumulate(cp.begin(), cp.end(), 0);

    S.leftmost = find_leftmost(C);
    vcount(C, S);  // compute p_inv, vnz, m2
    assert(S.vnz >= 0 && S.rnz >= 0);  // overflow guard

    return S;
}


QRResult qr(const CSCMatrix& A, const SymbolicQR& S)
{
    auto [M, N] = A.shape();
    csint M2 = S.m2;  // if M < N, M2 = M

    csint Nv = std::min(M, N);

    // Allocate result matrices
    CSCMatrix V({M2, Nv}, S.vnz);  // Householder vectors
    CSCMatrix R({M2, Nv}, S.rnz);  // R factor
    std::vector<double> beta(Nv);  // scaling factors

    // Allocate workspaces
    std::vector<double> x(M2);      // dense vector
    std::vector<csint>  w(M2, -1),  // workspace for pattern of V[:, k]
                        s, t;       // stacks for pattern of R[:, k]
    s.reserve(Nv);
    t.reserve(Nv);

    // Compute the V and R column by column
    csint vnz = 0,
          rnz = 0;

    for (csint k = 0; k < Nv; k++) {
        R.p_[k] = rnz;    // R[:, k] starts here
        V.p_[k] = vnz;    // V[:, k] starts here
        csint p1 = vnz;   // save start of V(:, k)
        w[k] = k;         // add V(k, k) to pattern of V
        V.i_[vnz++] = k;  // V(k, k) is non-zero

        t.clear();
        csint col = S.q[k];  // permuted column of A

        // find R[:, k] pattern
        for (csint p = A.p_[col]; p < A.p_[col+1]; p++) {
            csint i = S.leftmost[A.i_[p]];  // i = min(find(A(i, q)))

            s.clear();
            while (w[i] != k) {  // traverse up to k
                s.push_back(i);
                w[i] = k;
                i = S.parent[i];
            }

            // Push path onto "output" stack
            std::copy(s.rbegin(), s.rend(), std::back_inserter(t));

            i = S.p_inv[A.i_[p]];     // i = permuted row of A(:, col)
            x[i] = A.v_[p];           // x(i) = A(:, col)

            if (i > k && w[i] < k) {  // pattern of V(:, k) = x(k+1:m)
                V.i_[vnz++] = i;      // add i to pattern of V(:, k)
                w[i] = k;
            }
        }

        // for each i in pattern of R[:, k] (R(i, k) is non-zero)
        for (csint i : t | std::views::reverse) {
            x = happly(V, i, beta[i], x);  // apply (V(i), Beta(i)) to x
            R.i_[rnz] = i;                 // R(i, k) = x(i)
            R.v_[rnz++] = x[i];
            x[i] = 0;
            if (S.parent[i] == k) {
                // Scatter the non-zero pattern without changing the values
                vnz = V.scatter(i, 0, w, std::nullopt, k, V, vnz, false);
            }
        }

        // gather V(:, k) = x
        for (csint p = p1; p < vnz; p++) {
            V.v_[p] = x[V.i_[p]];
            x[V.i_[p]] = 0;  // clear x
        }

        // [v, beta, s] = house(x) == house(V[p1:vnz, k])
        Householder h = house(std::span(V.v_).subspan(p1, vnz - p1));
        std::copy(h.v.begin(), h.v.end(), V.v_.begin() + p1);
        beta[k] = h.beta;
        R.i_[rnz] = k;      // R(k, k) = -sign(x[0]) * norm(x)
        R.v_[rnz++] = h.s;
    }

    R.p_[Nv] = rnz;  // finalize R
    V.p_[Nv] = vnz;  // finalize V

    std::vector<csint> q = S.q;

    if (M < N) {
        // Compute the remaining columns of R: append Q^T A[:, M:] to R
        R = hstack(R, apply_qtleft(V, beta, S.p_inv, A.slice(0, M, M, N)));
        // Append the remaining columns of A onto q
        for (csint k = M; k < N; k++) {
            q.push_back(k);
        }
    }

    return {V, beta, R, S.p_inv, q};
}


// Exercise 5.1
QRResult symbolic_qr(const CSCMatrix& A, const SymbolicQR& S)
{
    auto [M, N] = A.shape();
    csint M2 = S.m2;

    // Exercise 5.2: handle M < N
    csint Nv = std::min(M, N);

    // Allocate result matrices with no values
    bool values = false;
    CSCMatrix V({M2, Nv}, S.vnz, values);   // Householder vectors
    CSCMatrix R({M2, Nv}, S.rnz, values);   // R factor

    // Allocate workspaces
    std::vector<csint> w(M2, -1),  // workspace for pattern of V[:, k]
                       s, t;       // stacks for pattern of R[:, k]
    s.reserve(Nv);
    t.reserve(Nv);

    // Compute V and R
    csint vnz = 0,
          rnz = 0;

    for (csint k = 0; k < Nv; k++) {
        R.p_[k] = rnz;    // R[:, k] starts here
        V.p_[k] = vnz;    // V[:, k] starts here
        w[k] = k;         // add V(k, k) to pattern of V
        V.i_[vnz++] = k;  // V(k, k) is non-zero

        t.clear();
        csint col = S.q[k];  // permuted column of A
        // find R[:, k] pattern
        for (csint p = A.p_[col]; p < A.p_[col+1]; p++) {
            csint i = S.leftmost[A.i_[p]];  // i = min(find(A(i, q)))

            s.clear();
            while (w[i] != k) {  // traverse up to k
                s.push_back(i);
                w[i] = k;
                i = S.parent[i];
            }

            // Push path onto "output" stack
            std::copy(s.rbegin(), s.rend(), std::back_inserter(t));

            i = S.p_inv[A.i_[p]];     // i = permuted row of A(:, col)

            if (i > k && w[i] < k) {  // pattern of V(:, k)
                V.i_[vnz++] = i;      // add i to pattern of V(:, k)
                w[i] = k;
            }
        }

        // for each i in pattern of R[:, k] (R(i, k) is non-zero)
        for (csint i : t | std::views::reverse) {
            R.i_[rnz++] = i;  // R(i, k)
            if (S.parent[i] == k) {
                // Scatter the non-zero pattern without changing the values
                vnz = V.scatter(i, 0, w, std::nullopt, k, V, vnz, values);
            }
        }

        R.i_[rnz++] = k;  // R(k, k)
    }

    R.p_[Nv] = rnz;  // finalize R
    V.p_[Nv] = vnz;  // finalize V

    return {V, {}, R, S.p_inv, S.q};
}


// Exercise 5.3
void reqr(const CSCMatrix& A, const SymbolicQR& S, QRResult& res)
{
    auto [M, N] = A.shape();
    csint M2 = S.m2;

    csint Nv = std::min(M, N);

    // Check that results have been allocated
    CSCMatrix& V = res.V;
    CSCMatrix& R = res.R;
    std::vector<double>& beta = res.beta;

    if (V.indices().empty() || R.indices().empty()) {
        throw std::runtime_error("V and R patterns have not been computed!");
    }

    // Allocate values in the result matrices
    V.v_.resize(V.nnz());
    beta.resize(V.N_);
    R.v_.resize(R.nnz());

    // Allocate workspaces
    std::vector<double> x(M2);  // dense vector

    // Compute V and R
    for (csint k = 0; k < Nv; k++) {
        csint col = res.q[k];  // permuted column of A

        // R[:, k] pattern known. Scatter A[:, col] into x
        for (csint p = A.p_[col]; p < A.p_[col+1]; p++) {
            csint i = res.p_inv[A.i_[p]];  // i = permuted row of A(:, col)
            x[i] = A.v_[p];                // x(i) = A(:, col)
        }

        // for each i in pattern of R[:, k] (R(i, k) is non-zero)
        for (csint p = R.p_[k]; p < R.p_[k+1] - 1; p++) {
            csint i = R.i_[p];             // R(i, k)
            x = happly(V, i, beta[i], x);  // apply (V(i), Beta(i)) to x
            R.v_[p] = x[i];                // R(i, k) = x(i)
            x[i] = 0;
        }

        // gather V(:, k) = x
        for (csint p = V.p_[k]; p < V.p_[k+1]; p++) {
            V.v_[p] = x[V.i_[p]];
            x[V.i_[p]] = 0;  // clear x
        }

        // [v, beta, s] = house(x) == house(V[:, k])
        auto V_k = std::span(V.v_).subspan(V.p_[k], V.p_[k+1] - V.p_[k]);
        Householder h = house(V_k);
        std::copy(h.v.begin(), h.v.end(), V.v_.begin() + V.p_[k]);
        beta[k] = h.beta;
        R.v_[R.p_[k+1] - 1] = h.s;  // R(k, k) = -sign(x[0]) * norm(x)
    }

    if (M < N) {
        // Compute the remaining columns of R
        R = hstack(R, apply_qtleft(V, beta, res.p_inv, A.slice(0, M, M, N)));
        // Append the remaining columns of A onto q
        for (csint k = M; k < N; k++) {
            res.q.push_back(k);
        }
    }
}


std::vector<double> apply_qleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    const std::vector<csint>& p_inv,
    const std::vector<double>& y
)
{
    auto [M2, N] = V.shape();

    std::vector<double> Px = y;

    // Px is size N, but happly expects size M2
    if (M2 > N) {
        Px.insert(Px.end(), M2 - N, 0.0);  // pad with zeros
    }

    for (csint j = N - 1; j >= 0; j--) {
        Px = happly(V, j, beta[j], Px);
    }

    std::vector<double> x = pvec(p_inv, Px);

    return x;
}


std::vector<double> apply_qtleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    const std::vector<csint>& p_inv,
    const std::vector<double>& y
)
{
    auto [M2, N] = V.shape();

    std::vector<double> x = y;

    csint M = x.size();
    if (M2 > M) {
        x.insert(x.end(), M2 - M, 0.0);  // pad with zeros
    }

    x = ipvec(p_inv, x);  // x = Py

    for (csint j = 0; j < N; j++) {
        x = happly(V, j, beta[j], x);
    }

    return x;
}


CSCMatrix apply_qtleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    const std::vector<csint>& p_inv,
    const CSCMatrix& Y
)
{
    auto [M2, N] = V.shape();
    auto [M, NY] = Y.shape();

    CSCMatrix X = Y;  // copy Y into X, work in-place
    CSCMatrix C({M, NY}, 2 * V.nnz());  // allocate C for the result

    if (M2 > M) {
        X.add_empty_bottom(M2 - M);
    }

    // NOTE p_inv is passed along to apply_qtleft(V, beta, p_inv, x)
    // X = X.permute_rows(p_inv);  // apply p_inv to Y

    std::vector<double> x(M);
    csint nz = 0;

    // Apply the Householder reflectors to each column of Y
    for (csint k = 0; k < NY; k++) {
        if (nz + M > C.nzmax()) {
            C.realloc(2 * C.nzmax() + M);
        }

        C.p_[k] = nz;  // column j of C starts here

        // Scatter X(:, k) into x
        for (csint p = X.p_[k]; p < X.p_[k+1]; p++) {
            x[X.i_[p]] = X.v_[p];
        }

        // Apply Householder reflection to x
        x = apply_qtleft(V, beta, p_inv, x);

        // Gather x into X(:, k)
        for (csint i = 0; i < M; i++) {
            if (x[i] != 0) {
                C.i_[nz] = i;
                C.v_[nz++] = x[i];
                x[i] = 0.0;  // clear x
            }
        }
    }

    C.p_[NY] = nz;  // finalize C
    C.realloc();

    return C;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
