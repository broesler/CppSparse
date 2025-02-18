%===============================================================================
%     File: householder_experiment.m
%  Created: 2025-02-12 16:00
%   Author: Bernie Roesler
%
%  Description: Experiment on p 79 of Davis.
%
%===============================================================================

clear; close all;

COLAMD = false;  % if true, use colamd ordering, else use natural ordering

load west0479
A = west0479;

if COLAMD
    q = colamd(A);
else
    q = 1:size(A, 2);
end

[Q, R] = qr(A(:,q));

[V, beta, p, R2] = cs_qr(A(:, q));
Q2 = cs_qright(V, beta, p, speye(size(V, 1)));

fprintf('nnz(A) = %d\n', nnz(A));
fprintf('nnz(Q) = %d\n', nnz(Q));
fprintf('nnz(R) = %d\n', nnz(R));
fprintf('nnz(V) = %d\n', nnz(V));

%-------------------------------------------------------------------------------
%        Plots
%-------------------------------------------------------------------------------
fig = figure(1); hold on
set(fig, 'Position', [300, 200, 900, 640]);

fig_title = 'QR factorization of A';
if COLAMD
    fig_title = [fig_title, ' (colamd ordering)'];
else
    fig_title = [fig_title, ' (natural ordering)'];
end
title(fig_title);

subplot(2, 2, 1); hold on
title('A');
spy(A);

subplot(2, 2, 2); hold on
title('V');
spy(V); 

subplot(2, 2, 3); hold on
title('Q');
spy(Q); 

subplot(2, 2, 4); hold on
title('R');
spy(R); 

%===============================================================================
%===============================================================================
