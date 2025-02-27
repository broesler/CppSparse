%===============================================================================
%     File: householder_experiment.m
%  Created: 2025-02-12 16:00
%   Author: Bernie Roesler
%
%  Description: Experiment on p 79 of Davis.
%
%===============================================================================

clear; close all;

SAVE_FIGS = false;
fig_path = '../plots/';

COLAMD = true;  % if true, use colamd ordering, else use natural ordering

load west0479
A = west0479;

if COLAMD
    % Permute the columns by approximate minimum degree ordering
    q = colamd(A);
    Aq = A(:, q);
else
    Aq = A;
end

[Q, R] = qr(Aq);

[V, beta, p, R2] = cs_qr(Aq);
Q2 = cs_qright(V, beta, p, speye(size(V, 1)));

% Compare "P" return with pre-computed colamd ordering
% NOTE for a *sparse* A, the column permutation is selected to minimize fill-in
% in the QR decomposition. For a *dense* A, the column permutation is selected
% such that the diagonal elements of R are non-increasing.
% For this A matrix, we actually get *fewer* nonzeros in R with the built-in QR
% permutation than we do by pre-computing colamd!
[Qp, Rp, pp] = qr(A, 'vector');
Ap = A(:, pp);

[Vp, ~, ~] = cs_qr(Ap);
% Q2 = cs_qright(V, beta, p, speye(size(V, 1)));

fprintf('nnz(A) = %d\n', nnz(A));
fprintf('nnz(Q) = %d\n', nnz(Q));
fprintf('nnz(R) = %d\n', nnz(R));
fprintf('nnz(V) = %d\n', nnz(V));
fprintf('nnz(Rp) = %d\n', nnz(Rp));
fprintf('nnz(Vp) = %d\n', nnz(Vp));

% With natural ordering:
% nnz(A) = 1888
% nnz(Q) = 148170
% nnz(R) = 59442
% nnz(V) = 40959
%
% With colamd ordering:
% nnz(A) = 1888
% nnz(Q) = 38764
% nnz(R) = 7599
% nnz(V) = 3909
% nnz(Rp) = 6833  # with built-in QR permutation
% nnz(Vp) = 3946

assert(norm(Aq - Q*R, 'fro') < 1e-9);
assert(norm(Aq - Q2*R2, 'fro') < 1e-9);


%-------------------------------------------------------------------------------
%        Plot the original and pre-permuted matrices
%-------------------------------------------------------------------------------
fig = figure(1); hold on
set(fig, 'Position', [300, 200, 900, 380]);

if COLAMD
    fig_title = 'COLAMD Ordering';
    tag = 'COLAMD_';
else
    fig_title = 'Natural Ordering';
    tag = 'NATURAL_';
end

% Manually add title text
text(0.5, 0.95, fig_title, ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

subplot(1, 2, 1); hold on
title('A');
spy(A);
axis equal

subplot(1, 2, 2); hold on
title('A(:, q)');
spy(Aq); 
axis equal

if SAVE_FIGS
    saveastight(fig, [fig_path 'west0479_' tag 'A_MATLAB.pdf']);
end


%-------------------------------------------------------------------------------
%        Plot the QR decomposition with pre-permuted matrix
%-------------------------------------------------------------------------------
fig = figure(2); hold on
set(fig, 'Position', [300, 200, 900, 380]);

% Manually add title text
text(0.5, 0.95, fig_title, ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

subplot(1, 2, 1); hold on
title('Q');
spy(Q); 
axis equal

subplot(1, 2, 2); hold on
title('V + R');
spy(V + R); 
axis equal

if SAVE_FIGS
    saveastight(fig, [fig_path 'west0479_' tag 'QR_MATLAB.pdf']);
end


%-------------------------------------------------------------------------------
%        Plot thet qr-permuted matrices
%-------------------------------------------------------------------------------
fig = figure(3); hold on
set(fig, 'Position', [300, 200, 900, 380]);

% Manually add title text
text(0.5, 0.95, fig_title, ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

subplot(1, 2, 1); hold on
title('Qp');
spy(Qp); 
axis equal

subplot(1, 2, 2); hold on
title('Vp + Rp');
spy(Vp + Rp); 
axis equal

if SAVE_FIGS
    saveastight(fig, [fig_path 'west0479_' tag 'QRp_MATLAB.pdf']);
end

%===============================================================================
%===============================================================================
