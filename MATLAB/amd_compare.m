%===============================================================================
%     File: amd_compare.m
%  Created: 2025-05-05 15:08
%   Author: Bernie Roesler
%
%  Description: Compare the SuiteSparse AMD with the CSparse AMD function.
%
%===============================================================================

clear;

% Ns = [10, 20, 50, 100];
% Ns = [10, 20, 50, 100, 200, 500, 1000];
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
density = 0.1;
N_trials = 5;
N_samples = 10;

amd_times = zeros(length(Ns), 1);
amd2_times = zeros(length(Ns), 1);
cs_amd_times = zeros(length(Ns), 1);

% Create a random sparse matrix and time row indexing
for i = 1:length(Ns)
    N = Ns(i);
    disp(['N = ', num2str(N)]);

    % TODO repeat for a number of matrices at each size
    A = sprandsym(N, density);

    amd_times(i) = timeit(@() amd(A), N_trials, N_samples);
    amd2_times(i) = timeit(@() amd2(A), N_trials, N_samples);
    cs_amd_times(i) = timeit(@() cs_amd(A), N_trials, N_samples);

    % Compare the number of non-zeros in the factorization
    % p = amd(A);
    % lnz = sum(symbfact(A(p,p)))

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

% saveas(1, '../plots/amd_times.png');

%===============================================================================
%===============================================================================
