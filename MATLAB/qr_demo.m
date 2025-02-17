%===============================================================================
%     File: qr_demo.m
%  Created: 2025-02-11 16:56
%   Author: Bernie Roesler
%
%  Description: Test QR decomposition algorithms.
%
%===============================================================================

% % See Strang "Linear Algebra", p. 203
% A = [[1 1 2]
%      [0 0 1]
%      [1 0 0]];

% [V_r, Beta_r, R_r] = qr_right(A)
% [V_l, Beta_l, R_l] = qr_left(A)

% [Q, R] = qr(A);

A = davis_example();

% Compute the QR decomposition two different ways
[V_r, Beta_r, R_r] = qr_right(A);
[V_l, Beta_l, R_l] = qr_left(A);  % FIXME this function is incorrect

% V_r and V_l are the same,
% Beta_r and Beta_l are the same
% R_r and R_l are different, but only on the diagonal:
%  R(1, 1) differs by a sign. The rest have the same sign but different values.

% Compute the Q matrix by applying the Householder vectors to the identity
Q_r = cs_qright(V_r, Beta_r, [], speye(size(V_r, 1)));
Q_l = cs_qright(V_l, Beta_l, [], speye(size(V_l, 1)));

[Q_, R_] = qr(A);

% Q_r and Q_l are the same
% Q_ differs from Q_r and Q

% Compare to the built-in MATLAB QR decomposition
norm(abs(Q_r * R_r - A), 2)
norm(abs(Q_l * R_l - A), 2)  % FIXME fails
norm(abs(Q_ * R_ - A), 2)


%===============================================================================
%===============================================================================
