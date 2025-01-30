%===============================================================================
%     File: chol_demo.m
%  Created: 2025-01-22 11:50
%   Author: Bernie Roesler
%
%  Description: Set up example matrix for testing Cholesky-related functions.
%
%===============================================================================

N = 11;
rows = [1:N, 1 + [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]];
cols = [1:N, 1 + [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]];
vals = ones(size(rows));

% Values for the lower triangle
L = sparse(rows, cols, vals, N, N);

% Create the symmetric matrix A, and add to diagonal to ensure positive definite
diag_A = max(sum(L + L' - 2*diag(diag(L))));
A = full(L + triu(L', 1) + diag_A*eye(N));  % use full for easier display

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

% TODO fails in octave
% G = graph(A);
% Gp = graph(R + R');
% plot(G);

%===============================================================================
%===============================================================================
