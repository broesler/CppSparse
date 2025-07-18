%===============================================================================
%     File: amd_compare.m
%  Created: 2025-05-05 15:08
%   Author: Bernie Roesler
%
%  Description: Solution to Davis, Exercise 7.1: Compare the built-in MATLAB AMD
%  (`amd`), the SuiteSparse AMD (`amd2`), and the CSparse AMD (`cs_amd`).
%
%  Compare:
%    - run time
%    - memory usage
%    - ordering quality (via number of non-zeros in the Cholesky factorization)
%
%===============================================================================

clear;

SAVE_FIGS = true;

% Ns = [10, 20, 50, 100];
Ns = [10, 20, 50, 100, 200, 500, 1000];
% Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
density = 0.02;
N_trials = 5;

amd_times = zeros(length(Ns), 1);
amd2_times = zeros(length(Ns), 1);
cs_amd_times = zeros(length(Ns), 1);

amd_lnzs = zeros(length(Ns), 1);
amd2_lnzs = zeros(length(Ns), 1);
cs_amd_lnzs = zeros(length(Ns), 1);

% Create a random sparse matrix and time row indexing
for k = 1:length(Ns)
    N = Ns(k);
    disp(['----- N = ', num2str(N)]);

    % Create a random symmetric matrix with specified bandwidth
    A = sprandsym(N, density);
    bw = round(N/10);
    [i, j, v] = find(A);
    keep = abs(i - j) <= bw;
    A = sparse(i(keep), j(keep), v(keep), N, N);

    amd_times(k) = timeit(@() amd(A), N_trials);
    amd2_times(k) = timeit(@() amd2(A), N_trials);
    cs_amd_times(k) = timeit(@() cs_amd(A), N_trials);

    fprintf( ...
        ['        AMD time: %.2e s\n', ...
         '       AMD2 time: %.2e s\n', ...
         'CSparse AMD time: %.2e s\n'], ...
        amd_times(k), amd2_times(k), cs_amd_times(k) ...
    );

    % Compare the number of non-zeros in the factorization
    p = amd(A);
    amd_lnzs(k) = sum(symbfact(A(p,p)));

    p = amd2(A);
    amd2_lnzs(k) = sum(symbfact(A(p,p)));

    p = cs_amd(A);
    cs_amd_lnzs(k) = sum(symbfact(A(p,p)));

    amd_titles = {'Built-In AMD', 'SuiteSparse AMD', 'CSparse AMD'};
    amd_funcs = { ...
        @(A) amd(A), ...
        @(A) amd2(A), ...
        @(A) cs_amd(A) ...
    };

    % Display a matrix
    if N == max(Ns)
        figure(3); clf;

        subplot(1, 4, 1);
        spy(A);
        title('Matrix A');
        axis equal;

        for s = 1:3
            subplot(1, 4, s + 1);
            p = amd_funcs{s}(A);
            spy(A(p, p));
            title(amd_titles{s});
            axis equal;
        end

        orient landscape;

        if SAVE_FIGS
            saveas(3, '../plots/amd_matrix.png');
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
    saveas(1, '../plots/amd_times.png');
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
    saveas(2, '../plots/amd_lnzs.png');
end

%===============================================================================
%===============================================================================
