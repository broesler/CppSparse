function A = davis_example_chol()
% DAVIS_EXAMPLE_CHOL Creates an 11x11 sparse, symmetric, positive definite
% matrix from Davis p 39, Figure 4.2.
%
% Example matrix from Davis, "Direct Methods for Sparse Linear Systems", p. 39,
% Figure 4.2.
%
% Inputs:
%   None
% Outputs:
%   A - (sparse) matrix:
%       [ 11    0    0    0    0    1    1    0    0    0    0;
%          0   12    1    0    0    0    0    1    0    0    0;
%          0    1   13    0    0    0    0    0    0    1    1;
%          0    0    0   14    0    1    0    0    0    1    0;
%          0    0    0    0   15    0    0    1    0    0    1;
%          1    0    0    1    0   16    0    0    1    1    0;
%          1    0    0    0    0    0   17    0    0    0    1;
%          0    1    0    0    1    0    0   18    0    1    1;
%          0    0    0    0    0    1    0    0   19    0    0;
%          0    0    1    1    0    1    0    1    0   20    1;
%          0    0    1    0    1    0    1    1    0    1   21 ]
%
% Example:
%  A = davis_example_chol();
%  disp(full(A));
%  assert(A == A')
%
%===============================================================================
%     File: davis_example_chol.m
%  Created: 2025-04-15 14:31
%   Author: Bernie Roesler
%===============================================================================

N = 11;  % total number of rows and columns

% Only off-diagonal elements
rows = 1 + [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10];
cols = 1 + [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9];
vals = ones(length(rows), 1);

% Values for the lower triangle
L = sparse(rows, cols, vals, N, N);

% Create the symmetric matrix A
A = L + L';

% Set the diagonal to ensure positive definiteness
for i = 1:N
    A(i, i) = i + 9;
end

end
%===============================================================================
%===============================================================================
