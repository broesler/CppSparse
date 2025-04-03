function A = davis_example_qr()
% davis_example_qr Creates a sparse matrix from Davis p 74, Figure 5.1.
%
% Example matrix from Davis, "Direct Methods for Sparse Linear Systems", p. 74,
% Figure 5.1.
%
% Inputs:
%   None
% Outputs:
%   A - (sparse) matrix:
%       [ 1   0   0   1   0   0   1   0;
%         0   2   1   0   0   0   1   0;
%         0   0   3   1   0   0   0   0;
%         1   0   0   4   0   0   1   0;
%         0   0   0   0   5   1   0   0;
%         0   0   0   0   1   6   0   1;
%         0   1   1   0   0   0   7   1;
%         0   0   0   0   1   1   1   0 ]
%
% Example:
%  A = davis_example_qr();
%  disp(full(A));
%
%===============================================================================
%     File: davis_example_qr.m
%  Created: 2025-04-03 09:16
%   Author: Bernie Roesler
%===============================================================================

% Davis QR example Figure 5.1, p 74.
% Define the test matrix A (See Davis, Figure 5.1, p 74)
i = 1 + [0, 1, 2, 3, 4, 5, 6, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6];
j = 1 + [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7];

% Label the diagonal elements 1..7, skipping the 8th
v = ones(size(i));
v(1:7) = [1:7];

A = sparse(i, j, v, 8, 8);

end
%===============================================================================
%===============================================================================
