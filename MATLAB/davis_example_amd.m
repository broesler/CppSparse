function A = davis_example_amd()
% DAVIS_EXAMPLE_AMD Creates an 10x10 sparse, symmetric, positive definite
% matrix from Davis p 7.3, p 101.
%
% Example matrix from Davis, "Direct Methods for Sparse Linear Systems", p. 101,
% Figure 7.1.
%
% Inputs:
%   None
% Outputs:
%   A - (sparse) matrix:
%       [ 10    0    0    1    0    1    0    0    0    0;
%          0   11    0    0    1    1    0    0    1    0;
%          0    0   12    0    1    1    1    0    0    0;
%          1    0    0   13    0    0    1    1    0    0;
%          0    1    1    0   14    0    1    0    1    0;
%          1    1    1    0    0   15    0    0    0    0;
%          0    0    1    1    1    0   16    1    1    1;
%          0    0    0    1    0    0    1   17    1    1;
%          0    1    0    0    1    0    1    1   18    1;
%          0    0    0    0    0    0    1    1    1   19 ]


%
% Example:
%  A = davis_example_amd();
%  disp(full(A));
%  assert(A == A')
%
%===============================================================================
%     File: davis_example_amd.m
%  Created: 2025-04-20 20:00
%   Author: Bernie Roesler
%===============================================================================

N = 10;  % total number of rows and columns

% Only off-diagonal elements
rows = 1 + [0, 3, 5, 1, 4, 5, 8, 2, 4, 5, 6, 3, 6, 7, 4, 6, 8, 5, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9];
cols = 1 + [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9];
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
