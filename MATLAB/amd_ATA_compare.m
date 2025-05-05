%===============================================================================
%     File: amd_ATA_compare.m
%  Created: 2025-05-05 16:30
%   Author: Bernie Roesler
%
%  Description: Compare the SuiteSparse AMD with the CSparse AMD function.
%
%===============================================================================

clear;

SAVE_FIGS = false;

% Ns = [10, 20, 50, 100];
Ns = [10, 20, 50, 100, 200, 500, 1000];
% Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
density = 0.02;
N_trials = 5;
N_samples = 10;

amd_times = zeros(length(Ns), 1);
amd2_times = zeros(length(Ns), 1);
cs_amd_times = zeros(length(Ns), 1);

amd_lnzs = zeros(length(Ns), 1);
amd2_lnzs = zeros(length(Ns), 1);
cs_amd_lnzs = zeros(length(Ns), 1);

% Create a random sparse matrix and time row indexing
for i = 1:length(Ns)
    N = Ns(i);
    disp(['N = ', num2str(N)]);

    % TODO try 2D or 3D Laplacian matrix
    A = sprand(N, N, density);

    % Specify bandwidth by eliminating elements outside bands
    [J, I] = meshgrid(1:N, 1:N);
    bw = round(N/10);
    is_within_band = abs(I - J) <= bw;
    A(~is_within_band) = 0;

    ATA = A' * A;
    f_amd = @() amd(ATA);
    f_amd2 = @() amd2(ATA);
    f_cs_amd = @() cs_amd(A, 3);

    amd_times(i) = timeit(f_amd, N_trials, N_samples);
    amd2_times(i) = timeit(f_amd2, N_trials, N_samples);
    cs_amd_times(i) = timeit(f_cs_amd, N_trials, N_samples);

    % Compare the number of non-zeros in the factorization
    q = f_amd();
    amd_lnzs(i) = sum(symbfact(A(:, q), 'col'));

    q = f_amd2();
    amd2_lnzs(i) = sum(symbfact(A(:, q), 'col'));

    q = f_cs_amd();
    cs_amd_lnzs(i) = sum(symbfact(A(:, q), 'col'));

    % Display a matrix
    if N == 1000
        figure(3); clf;

        subplot(1, 2, 1);
        spy(A);
        title('Matrix A');

        subplot(1, 2, 2);
        spy(A(q, q));
        title('AMD Permuted Matrix');

        orient landscape;

        if SAVE_FIGS
            saveas(3, '../plots/amd_ATA_matrix.png');
        end
    end
end


%-------------------------------------------------------------------------------
%        Plot the results
%-------------------------------------------------------------------------------
figure(1); clf; hold on;
loglog(Ns, amd_times, 'k.-');
loglog(Ns, amd2_times, 'o-');
loglog(Ns, cs_amd_times, 'x-');

legend('Built-In AMD', 'AMD', 'CSparse AMD', 'Location', 'SouthEast');

grid on;
orient landscape;
title(sprintf('AMD Timing Comparison (density = %.2f)', density));
xlabel('Number of Columns');
ylabel('Time to Permute [s]');

if SAVE_FIGS
    saveas(1, '../plots/amd_ATA_times.png');
end

figure(2); clf; hold on;
loglog(Ns, amd_lnzs, 'k.-');
loglog(Ns, amd2_lnzs, 'o-');
loglog(Ns, cs_amd_lnzs, 'x-');

legend('Built-In AMD', 'AMD', 'CSparse AMD', 'Location', 'SouthEast');

grid on;
orient landscape;
title(sprintf('AMD Quality Comparison (density = %.2f)', density));
xlabel('Number of Columns');
ylabel('Number of Non-zeros in Factorization');

if SAVE_FIGS
    saveas(2, '../plots/amd_ATA_lnzs.png');
end

%===============================================================================
%===============================================================================
