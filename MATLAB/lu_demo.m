%===============================================================================
%     File: lu_demo.m
%  Created: 2025-04-24 20:41
%   Author: Bernie Roesler
%
%  Description: Test LU decomposition and solve Ax = b to compare MATLAB
%    behavior with python and cs_lu.
%
%===============================================================================

clear; close all;

A = davis_example_qr();
N = size(A, 1);

for i = 1:N
    A(i, i) += 10;
end

expect = [1:N]';

p = 1 + [5, 1, 7, 0, 2, 6, 4, 3]';  % arbitrary permutation
p_inv = inv_permute(p);

Ap = A(p, :);

%% Permuted Solve
b = A * expect;
bp = b(p);

assert(norm(bp - Ap * expect) < 1e-14, 'Permutation failed');

[L, U, P] = lu(full(Ap));
Q = eye(N);

assert(norm(L * U - P * Ap) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = P * bp;
y = L \ Pb;
QTx = U \ y;
x = Q * QTx;

assert(norm(x - expect) < 1e-14, 'LU solve failed');

%% Repeat with vector form of permutation
[L, U, pv] = lu(full(Ap), 'vector');

assert(norm(L * U - Ap(pv, :)) < 1e-14, 'LU decomposition failed');

[pv_, ~] = find(P');

assert(norm(pv - pv_) < 1e-14, 'LU permutation failed');


%% Test solution with column permutation
[L, U, P, Q] = lu(A);

assert(norm(L * U - P * A * Q) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = P * b;
y = L \ Pb;
QTx = U \ y;
x = Q * QTx;

assert(norm(x - expect) < 1e-14, 'LU solve failed');

%% Repeat with vector form of permutation
[L, U, pv, qv] = lu(A, 'vector');

assert(norm(L * U - A(pv, qv)) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = b(pv);
y = L \ Pb;
QTx = U \ y;
x = QTx(inv_permute(qv));  % == Q * QTx requires inverse of qv

assert(norm(x - expect) < 1e-14, 'LU solve failed');


%% Test A' x = b
bt = A' * expect;

[L, U, P] = lu(full(Ap));
Q = eye(N);

% Solve A' x = bt
QTb = Q' * bt;
y = U' \ QTb;
Px = L' \ y;
xt = P' * Px;

% Solution for permuted rows of A:
% A' x = b
% => A -> PA  (permute rows of A)
% Now we are solving:
% (PA)' xp = b
% A' P' xp = b
% x = P' xp.

% permute x by P' == p_inv to match Ap'
x = xt(p_inv);  % == P' x

assert(norm(x - expect) < 1e-14, 'LU transpose solve failed');


%% Test Column permutations from cs_lu
% nargout == 4 && nargin == 2 -> order = 1 (APlusAT)
[L, U, p_, q_] = cs_lu(A, 1.0);

printf('APlusAT:\n  p = %s\n  q = %s\n', ...
       mat2str(p_ - 1), mat2str(q_ - 1));

% nargout == 4 && nargin == 1 -> order = 2 (ATANoDenseRows)
[L, U, p_, q_] = cs_lu(A);

printf('ATANoDenseRows:\n  p = %s\n  q = %s\n', ...
       mat2str(p_ - 1), mat2str(q_ - 1));

%===============================================================================
%===============================================================================
