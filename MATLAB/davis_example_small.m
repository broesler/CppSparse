function A = davis_example_small()
% DAVIS_EXAMPLE_SMALL Creates a sparse matrix from Davis pp 7-8, Eqn (2.1).
%
% Example matrix from Davis, "Direct Methods for Sparse Linear Systems", p. 7-8,
% Eqn (2.1).
%
% Inputs:
%   None
% Outputs:
%   A - (sparse) matrix:
%       [[4.5, 0. , 3.2, 0. ],
%        [3.1, 2.9, 0. , 0.9],
%        [0. , 1.7, 3. , 0. ],
%        [3.5, 0.4, 0. , 1. ]]
%
% Example:
%  A = davis_example_small();
%  disp(full(A));
%
%===============================================================================
%     File: davis_example_small.m
%  Created: 2025-02-05 11:43
%   Author: Bernie Roesler
%===============================================================================

% See Davis pp 7-8, Eqn (2.1)
i = [2,    1,    3,    0,    1,    3,    3,    1,    0,    2] + 1;
j = [2,    0,    3,    2,    1,    0,    1,    3,    0,    1] + 1;
v = [3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7];

A = sparse(i, j, v, 4, 4);

end
%===============================================================================
%===============================================================================
