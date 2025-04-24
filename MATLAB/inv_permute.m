function p_inv = inv_permute(p)
% INV_PERMUTE compute the inverse permutation of a vector.
%   p_inv = inv_permute(p) computes the inverse permutation of the vector p.
%===============================================================================
%     File: inv_permute.m
%  Created: 2025-04-23 20:18
%   Author: Bernie Roesler
%===============================================================================

for i = 1:length(p)
    p_inv(p(i)) = i;
end

%===============================================================================
%===============================================================================
