%===============================================================================
%     File: qr_rankdef.m
%  Created: 2025-04-30 14:59
%   Author: Bernie Roesler
%
%  Description: Test rank-deficient QR decomposition.
%
%===============================================================================

clear;

A = davis_example_qr();

k = 4;

% Create a rank-deficient matrix
% Ar = A(:, 1:5);
Ar = A;

[M, N] = size(Ar);

Ar(k, :) = 0;
% Ar(:, k) = 0;

[V, Beta, p, R] = cs_qr(Ar);

% Can't remove additional rows of V and R here, because otherwise p is invalid

Q = cs_qright(V, Beta, p, eye(size(V, 1)));

QR = (Q * R)(1:M, 1:N);

assert(norm(Ar - QR) < 1e-10, 'QR decomposition failed');

disp('size(A):')
disp(size(A))
disp('size(V):')
disp(size(V))
disp('size(R):')
disp(size(R))
disp('size(Q):')
disp(size(Q))

disp('Ar = ')
disp(full(Ar))

disp('QR = ')
disp(QR)

%===============================================================================
%===============================================================================
