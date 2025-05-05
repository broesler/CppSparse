%===============================================================================
%     File: colcolamd_compare.m
%  Created: 2025-05-05 16:18
%   Author: Bernie Roesler
%
%  Description: Compare the SuiteSparse COLCOLAMD with the CSparse COLAMD function.
%
%===============================================================================

clear;

warning off

SAVE_FIGS = true;

% Ns = [10, 20, 50, 100];
% Ns = [10, 20, 50, 100, 200, 500, 1000];
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
density = 0.02;
N_trials = 5;
N_samples = 10;

colamd_times = zeros(length(Ns), 1);
colamd2_times = zeros(length(Ns), 1);
cs_amd_times = zeros(length(Ns), 1);

colamd_lnzs = zeros(length(Ns), 1);
colamd2_lnzs = zeros(length(Ns), 1);
cs_amd_lnzs = zeros(length(Ns), 1);

% Create a random sparse matrix and time row indexing
for i = 1:length(Ns)
    N = Ns(i);
    disp(['N = ', num2str(N)]);

    A = sprand(N, N, density);

    % Specify bandwidth by eliminating elements outside bands
    [J, I] = meshgrid(1:N, 1:N);
    bw = round(N/10);
    is_within_band = abs(I - J) <= bw;
    A(~is_within_band) = 0;

    colamd_times(i) = timeit(@() colamd(A), N_trials, N_samples);
    colamd2_times(i) = timeit(@() colamd2(A), N_trials, N_samples);
    cs_amd_times(i) = timeit(@() cs_amd(A, 2), N_trials, N_samples);

    % Compare the number of non-zeros in the factorization
    q = colamd(A);
    [L, U, P] = lu(A(:, q));
    colamd_lnzs(i) = nnz(L) + nnz(U);

    q = colamd2(A);
    [L, U, P] = lu(A(:, q));
    colamd2_lnzs(i) = nnz(L) + nnz(U);

    q = cs_amd(A, 2);
    [L, U, P] = lu(A(:, q));
    cs_amd_lnzs(i) = nnz(L) + nnz(U);

    % Display a matrix
    if N == 1000
        figure(3); clf;

        subplot(1, 2, 1);
        spy(A);
        title('Matrix A');

        subplot(1, 2, 2);
        spy(P * A(:, q));
        title('COLAMD Permuted Matrix');

        orient landscape;

        if SAVE_FIGS
            saveas(3, '../plots/colamd_matrix.png');
        end
    end
end


%-------------------------------------------------------------------------------
%        Plot the results
%-------------------------------------------------------------------------------
figure(1); clf; hold on;
loglog(Ns, colamd_times, 'k.-');
loglog(Ns, colamd2_times, 'o-');
loglog(Ns, cs_amd_times, 'x-');

legend('Built-In COLAMD', 'COLAMD', 'CSparse COLAMD', 'Location', 'SouthEast');

grid on;
orient landscape;
title(sprintf('COLAMD Timing Comparison (density = %.2f)', density));
xlabel('Number of Columns');
ylabel('Time to Permute [s]');

if SAVE_FIGS
    saveas(1, '../plots/colamd_times.png');
end

figure(2); clf; hold on;
loglog(Ns, colamd_lnzs, 'k.-');
loglog(Ns, colamd2_lnzs, 'o-');
loglog(Ns, cs_amd_lnzs, 'x-');

legend('Built-In COLAMD', 'COLAMD', 'CSparse COLAMD', 'Location', 'SouthEast');

grid on;
orient landscape;
title(sprintf('COLAMD Quality Comparison (density = %.2f)', density));
xlabel('Number of Columns');
ylabel('Number of Non-zeros in Factorization');

if SAVE_FIGS
    saveas(2, '../plots/colamd_lnzs.png');
end

%===============================================================================
%===============================================================================
