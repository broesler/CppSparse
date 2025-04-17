%===============================================================================
%     File: chol_demo.m
%  Created: 2025-01-22 11:50
%   Author: Bernie Roesler
%
%  Description: Set up example matrix for testing Cholesky-related functions.
%
%===============================================================================

% N = 11;
% rows = [1:N, 1 + [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]];
% cols = [1:N, 1 + [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]];
% vals = ones(size(rows));

% % Values for the lower triangle
% L = sparse(rows, cols, vals, N, N);

% % Create the symmetric matrix A, and add to diagonal to ensure positive definite
% diag_A = max(sum(L + L' - 2*diag(diag(L))));
% A = full(L + triu(L', 1) + diag_A*eye(N));  % use full for easier display
% A = sparse(A);

A = davis_example_chol();
N = size(A, 1);

% Get the elimination tree
[parent, post] = etree(A);

% Compute the Cholesky factor
R = chol(A, 'lower');
Rp = chol(A(post, post), 'lower');

assert (nnz(R) == nnz(Rp));  % post-ordering does not change nnz

% Compute the row counts of the post-ordered Cholesky factor
col_counts = sum(Rp != 0);
row_counts = sum(Rp != 0, 2)';

[count, h, parent_, post_, Rs] = symbfact(sparse(A));

assert(parent == parent_');
assert(post == post_');

% Create update vector
k = randi(N);
idx = find(R(:, k));
w = zeros(N, 1);
w(idx) = rand(length(idx), 1);

% Update the Cholesky factor
A_up = A + w*w';

% Use built-in update function. 
% NOTE that cholupdate expects the *upper* triangular Cholesky factor.
R_up = cholupdate(R', w, '+')';

assert(norm(R_up * R_up' - A_up) < 1e-14)

% ---------- Compute the incomplete Cholesky factorization
% The local drop tolerance at step j of the factorization is:
%   norm(A(j:end, j), 1) * droptol.
droptol = 0.01;
options = struct('type', 'ict', 'droptol', droptol);

L = ichol(A, options);  % lower by default
Lf = chol(A, 'lower');  % upper by default

% Compute the 1-norm of each A(j:end, j), and show a comparison with the
% actual values of the full L to see which elements would be dropped. Compare
% this to an absolute drop tolerance like in our implementation.
col_norms = zeros(N, 1);
for k = 1:N
    col_norms(k) = norm(A(k:end, k), 1);
end

% Compute the drop tolerance for each column
col_droptols = col_norms * droptol;

% Compare the drop tolerance to the actual values of L (before scaling by pivot)
Lf_expectdrop = sparse(Lf);  % pre-allocate
for k = 1:N
    col = Lf(k:end, k);
    cond = abs(col * col(1)) < col_droptols(k);
    Lf_expectdrop(k:end, k) = col .* cond;
end

disp('Expected drops in Lf:');
disp(full(Lf_expectdrop))
fprintf('nnz(Lf_expectdrop): %d\n', nnz(Lf_expectdrop));  % == 7

% Get the actual entries where Lf is non-zero, but L is zero
% (i.e. the entries that were dropped)
Lf_isdropped = Lf .* (L == 0);

% disp('Entries of Lf that are dropped:');
% disp(full(Lf_isdropped));
fprintf('nnz(Lf_isdropped): %d\n', nnz(Lf_isdropped));  % == 6

% NOTE 
% For droptol = 0.01,
% In column 10, the drop tolerance as computed from A(10:end, 10) is 0.2,
% so the value L(10, 11) ~ 0.19 should get dropped by ichol, but doesn't?
% 
% Similar results for other droptol values. We expect to drop many more elements
% than actually get dropped, despite applying the stated drop tolerance from the
% MATLAB documentation.
%
% disp(nnz(Lf_expectdrop - Lf_isdropped))
%
% Octave checks 
%   std::abs (w_data[jrow]) < (droptol * cols_norm[k])      (line 358)
% *before* scaling by the pivot on  lines 391-393:
%     // Once elements are dropped and compensation of column sums are done,
%     // scale the elements by the pivot.
%     data_l[total_len] = std::sqrt (data_l[total_len]);
%     for (jj = total_len + 1; jj < (total_len + w_len); jj++)
%         data_l[jj] /= data_l[total_len];
% See permalink: <https://github.com/gnu-octave/octave/blob/ab54952a577d012156cbb6f3288be69c20183d86/libinterp/corefcn/__ichol__.cc#L358>
%
% We need to update our filter to re-scale the elements by the diagonal of L.
%
% TODO plot a graph of nnz vs. droptol for each filter to show difference.

% check that L*L' is a good approximation to A
drop_small = @(x) x .* (abs(x) > 1e-14);

LLT = L * L';
AmLLT = spfun(drop_small, A - LLT);  % maintain shape of matrix with spfun

fprintf('nnz(AmLLT): %d\n', nnz(AmLLT))  % 6 for droptol = 1e-2

% Test norm only on pattern of A
LLT_Anz = LLT .* spones(A);

Anz_norm = norm(A - LLT_Anz, 'fro') / norm(A, 'fro');
assert(Anz_norm < 1e-15)

% Test norm on entire matrix
test_norm = norm(AmLLT, 'fro') / norm(A, 'fro');
assert(test_norm < droptol)

% TODO graph not implemented in octave
% G = graph(A);
% Gp = graph(R + R');
% plot(G);

% Get column counts of A^T A
c = symbfact(A, 'col');

% c =     [7   6   8   8   7   6   5   4   3   2   1]  % chol(ATA)
% count = [3   3   4   3   3   4   4   3   3   2   1]  % chol(A)

%===============================================================================
%===============================================================================
