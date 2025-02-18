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

% TODO make these edits in my branch of CSparse and remove the files from this
% CSparse++ repo.
% Compute the QR decomposition two different ways
[V_r, Beta_r, R_r] = qr_right(A);
[V_l, Beta_l, R_l] = qr_left(A);  % FIXME R_l diagonal is not correct

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
end

% V_r and V_l now matche the V from scipy.linalg.qr(..., mode'raw').

% Compute the Q matrix by applying the Householder vectors to the identity
Q_r = cs_qright(V_r, Beta_r, [], speye(size(V_r, 1)));
Q_l = cs_qleft(V_l, Beta_l, [], speye(size(V_l, 1)));

[Q_, R_] = qr(A);

% Compare to the built-in MATLAB QR decomposition
norm(abs(Q_r - Q_l'), 2)
norm(abs(Q_r * R_r - A), 2)
norm(abs(Q_l' * R_l - A), 2)  % FIXME fails
norm(abs(Q_ * R_ - A), 2)


%===============================================================================
%===============================================================================
