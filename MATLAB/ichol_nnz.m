%===============================================================================
%     File: ichol_nnz.m
%  Created: 2025-04-27 19:59
%   Author: Bernie Roesler
%
%  Description: Compute the incomplete Cholesky factorization and test the
%    threshold dropping criterion in Octave.

% NOTE 
% For droptol = 0.01,
% In column 10, the drop tolerance as computed from A(10:end, 10) is 0.2,
% so the value L(10, 11) ~ 0.19 should get dropped by ichol, but doesn't?
% 
% Similar results for other droptol values. We expect to drop many more elements
% than actually get dropped, despite applying the stated drop tolerance from the
% MATLAB documentation.
%
% disp(nnz(L_expectdrop - L_isdropped))
%
% Octave checks 
%   std::abs (w_data[jrow]) < (droptol * cols_norm[k])      (line 358)
% *before* scaling by the pivot on  lines 391-393:
%     // Once elements are dropped and compensation of column sums are done,
%     // scale the elements by the pivot.
%     data_l[total_len] = std::sqrt (data_l[total_len]);
%     for (jj = total_len + 1; jj < (total_len + w_len); jj++)
%         data_l[jj] /= data_l[total_len];
% See permalink: <https://github.com/gnu-octave/octave/blob/ab54952a577d012156cbb6f3288be69c20183d86/libinterp/corefcn/__ichol__.cc#L358>
%
% We need to update our filter to re-scale the elements by the diagonal of L.
%
%===============================================================================

set(0, 'DefaultLineLineWidth', 1.15);
set(0, 'DefaultAxesFontSize', 14);

%% Create the A matrix
N = 11;  % total number of rows and columns

% Only off-diagonal elements
rows = 1 + [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10];
cols = 1 + [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9];
vals = ones(length(rows), 1);

% Values for the lower triangle
L = sparse(rows, cols, vals, N, N);

% Create the symmetric matrix A
A = L + L';

% Set the diagonal to ensure positive definiteness
for i = 1:N
    A(i, i) = i + 9;
end

clear L;

%% Compute the incomplete Cholesky factorization
% The local drop tolerance at step j of the factorization is:
%   norm(A(j:end, j), 1) * droptol.

% Try many droptols and see how many elements are dropped
% droptols = [0.01];  % testing
droptols = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1];

nnz_Ldrops = zeros(length(droptols), 1);
nnz_pivs = zeros(length(droptols), 1);
nnz_nopivs = zeros(length(droptols), 1);


for i = 1:length(droptols)
    droptol = droptols(i);
    options = struct('type', 'ict', 'droptol', droptol);

    L = ichol(A, options);  % incomplete (lower by default)
    Lf = chol(A, 'lower');  % full       (upper by default)

    % Compute the 1-norm of each A(j:end, j), and show a comparison with the
    % actual values of the full L to see which elements would be dropped. Compare
    % this to an absolute drop tolerance like in our implementation.
    col_norms = zeros(N, 1);
    for k = 1:N
        col_norms(k) = norm(A(k:end, k), 1);
    end

    % Compute the local drop tolerance for each column
    col_droptols = col_norms * droptol;

    % Compare the drop tolerance to the actual values of L
    % Ignore diagonals, they are never dropped
    L_expectdrop_piv = sparse(tril(Lf, -1));   % pre-allocate
    L_expectdrop_nopiv = sparse(tril(Lf, -1));

    for k = 1:N-1
        idx = k+1:N;     % rows to check, ignore diagonals
        piv = Lf(k, k);  % pivot
        col = Lf(idx, k);
        cond_piv = abs(col * piv) < col_droptols(k);  % scale by the pivot
        cond_nopiv = abs(col) < col_droptols(k);      % actual L(:, j)
        L_expectdrop_piv(idx, k) = col .* cond_piv;
        L_expectdrop_nopiv(idx, k) = col .* cond_nopiv;
    end

    % Get the actual entries where Lf is non-zero, but L is zero
    % (i.e. the entries that were dropped)
    L_isdropped = Lf .* (L == 0);

    % Store the results for plotting
    nnz_pivs(i) = nnz(L_expectdrop_piv);
    nnz_nopivs(i) = nnz(L_expectdrop_nopiv);
    nnz_Ldrops(i) = nnz(L_isdropped);
end

%-------------------------------------------------------------------------------
%        Plots
%-------------------------------------------------------------------------------
% Plot a graph of nnz vs. droptol for each filter to show difference.
figure(1); clf; hold on;

% plot(droptols, nnz(Lf) * ones(length(droptols), 1), 'k-', ...
%      'DisplayName', 'Full L');
plot(droptols, (nnz(Lf) - N) * ones(length(droptols), 1), 'k-.', ...
     'DisplayName', 'Full L (w/o diagonals)');

plot(droptols, nnz_Ldrops, 'kx-', 'LineWidth', 1.15, 'DisplayName', 'ichol');
plot(droptols, nnz_pivs,   'o--', 'LineWidth', 1.15, 'DisplayName', 'L with pivot');
plot(droptols, nnz_nopivs, 'o-.', 'LineWidth', 1.15, 'DisplayName', 'L without pivot');

set(gca, 'XScale', 'log');
xlabel('Drop tolerance');
ylabel('Number of non-zeros');
grid on
orient landscape

legend('Location', 'SouthEast');

% Set papersize
% set(gcf, 'papersize', [800 600]);
% set(gcf, 'paperposition', [10 10 780 580]);
% print(gcf, '../plots/ichol_nnz.pdf', '-dpdf', '-bestfit');

%===============================================================================
%===============================================================================
