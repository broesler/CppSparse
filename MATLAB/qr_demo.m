%===============================================================================
%     File: qr_demo.m
%  Created: 2025-02-11 16:56
%   Author: Bernie Roesler
%
%  Description: Test QR decomposition algorithms.
%
%===============================================================================

tol = 1e-14;

% ---------- See Strang "Linear Algebra", p. 203
%   The Strang example has a worked QR decomposition for comparison.
% A = [[1 1 2]
%      [0 0 1]
%      [1 0 0]];
% [V_r, Beta_r, R_r] = qr_right(A)
% [V_l, Beta_l, R_l] = qr_left(A)
% [Q, R] = qr(A);

% ---------- Matrix from Davis Figure 5.1, p 74.
N = 8;
rows = 1 + [0, 1, 2, 3, 4, 5, 6, 7, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6];
cols = 1 + [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7];
vals = [1:N-1, 0, ones(1, size(rows, 2) - N)];

A = sparse(rows, cols, vals, N, N);

% ---------- Take the example from Davis Eqn (2.1)
% A = davis_example();

%-------------------------------------------------------------------------------
%        Compute the QR decomposition
%-------------------------------------------------------------------------------
[V_r, Beta_r, R_r] = qr_right(A);
[V_l, Beta_l, R_l] = qr_left(A);

% Compute the QR decomposition using the CSparse implementation
% NOTE that the CSparse implementation computes a row permutation:
%   QR = PA
[V, Beta, p, R] = cs_qr(A);

% V is not scaled to be 1 on the diagonal, like in scipy.linalg.qr, so scale it
% manually.
%
% Divide each column of V by its diagonal element, and scale Beta by the square
% of the diagonal element.
for k = 1:size(V_r, 2)
    V_rkk = V_r(k, k);
    V_r(:, k) /= V_rkk;
    Beta_r(k) *= V_rkk^2;

    V_lkk = V_l(k, k);
    V_l(:, k) /= V_lkk;
    Beta_l(k) *= V_lkk^2;

    Vkk = V(k, k);
    V(:, k) /= Vkk;
    Beta(k) *= Vkk^2;
end

% V_r and V_l now match the V from scipy.linalg.qr(..., mode'raw').
assert(norm(abs(V_r - V_l)) < tol);
assert(norm(abs(Beta_r - Beta_l)) < tol);
assert(norm(abs(R_r - R_l)) < tol);

% NOTE that V != V_r, because of the row permutation applied
% R == -R_r *except* for the last element R(N, N) == R_r(N, N)

% Compute the Q matrix by applying the Householder vectors to the identity
Q_r = cs_qright(V_r, Beta_r, [], speye(size(V_r, 1)));
Q_l = cs_qleft(V_l, Beta_l, [], speye(size(V_l, 1)));

% Make sure to include the row permutation p!
Q = cs_qright(V, Beta, p, speye(size(V, 1)));

% Compute the QR decomposition using the built-in MATLAB function
[Q_, R_] = qr(A);

% Compare to the built-in MATLAB QR decomposition
assert(norm(abs(Q_r - Q_l')) < tol);
assert(norm(abs(Q_r * R_r - A)) < tol);
assert(norm(abs(Q_l' * R_l - A)) < tol);
assert(norm(abs(Q * R - A)) < tol);
assert(norm(abs(Q_ * R_ - A)) < tol);  % built-in


%===============================================================================
%===============================================================================
