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

% Tags for exercises 7.1, 7.2, and 7.3
tags = {'amd', 'colamd', 'amd_ATA'};

% Ns = [10, 20, 50, 100];
Ns = [10, 20, 50, 100, 200, 500, 1000];
% Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
density = 0.02;
N_trials = 5;

% Plot titles for names of AMD functions
amd_titles = {'Built-In AMD', 'SuiteSparse AMD', 'CSparse AMD'};

% Run for Exercise 7.1, 7.2, and 7.3
for tag = tags
    tag = tag{1};  % Convert from cell to string
    disp(['---------- Running for tag: ', tag]);

    % Compare the number of non-zeros in the factorization
    if strcmp(tag, 'amd')
        amd_funcs = {...
            @(A) amd(A), ...
            @(A) amd2(A), ...
            @(A) cs_amd(A) ...
        };
        quality_func = @(A, p) sum(symbfact(A(p, p)));
    elseif strcmp(tag, 'colamd')
        amd_funcs = {...
            @(A) colamd(A), ...
            @(A) colamd2(A), ...
            @(A) cs_amd(A, 2) ...
        };
        quality_func = @colamd_quality;
    elseif strcmp(tag, 'amd_ATA')
        amd_funcs = {...
            @(A) amd(A' * A), ...
            @(A) amd2(A' * A), ...
            @(A) cs_amd(A, 3) ...
        };
        quality_func = @(A, p) sum(symbfact(A(:, p), 'col'));
    else
        error('Unknown tag: %s', tag);
    end

    runtimes = zeros(length(amd_funcs), length(Ns));
    lnzs = zeros(length(amd_funcs), length(Ns));

    % Create a random sparse matrix and time row indexing
    for k = 1:length(Ns)
        N = Ns(k);
        disp(['----- N = ', num2str(N)]);

        % Create a random (possibly symmetric) matrix with specified bandwidth
        if strcmp(tag, 'amd')
            A = sprandsym(N, density);
        else
            A = sprand(N, N, density);
        end

        % Ensure the diagonal is non-zero
        A = A + speye(N);

        bw = round(N/10);
        [i, j, v] = find(A);
        keep = abs(i - j) <= bw;
        A = sparse(i(keep), j(keep), v(keep), N, N);

        % Time the AMD functions
        for s = 1:length(amd_funcs)
            runtimes(s, k) = timeit(@() amd_funcs{s}(A), N_trials);
        end

        fprintf( ...
            ['        AMD time: %.2e s\n', ...
            '       AMD2 time: %.2e s\n', ...
            'CSparse AMD time: %.2e s\n'], ...
            runtimes(1, k), runtimes(2, k), runtimes(3, k) ...
        );

        for s = 1:length(amd_funcs)
            p = amd_funcs{s}(A);
            lnzs(s, k) = quality_func(A, p);
        end

        % Display a matrix
        if N == max(Ns)
            figure(3); clf;

            N_subplots = length(amd_funcs) + 1;

            subplot(1, N_subplots, 1);
            spy(A);
            title('Matrix A');
            axis equal;

            for s = 1:length(amd_funcs)
                subplot(1, N_subplots, s + 1);
                p = amd_funcs{s}(A);
                spy(A(p, p));
                title(amd_titles{s});
                axis equal;
            end

            orient landscape;

            if SAVE_FIGS
                filename = ['../plots/amd_matrix_' tag '.png'];
                fprintf('Saving figure to %s\n', filename);
                saveas(3, filename);
            end
        end
    end


    %---------------------------------------------------------------------------
    %        Plot the results
    %---------------------------------------------------------------------------
    linestyles = {'k.-', 'o-', 'x-'};

    figure(1); clf; hold on;
    loglog(Ns, runtimes, linestyles);

    legend('Built-In AMD', 'AMD', 'CSparse AMD', 'Location', 'SouthEast');

    grid on;
    orient landscape;
    title(sprintf('AMD Timing Comparison (density = %.2f)', density));
    xlabel('Number of Columns');
    ylabel('Time to Permute [s]');

    if SAVE_FIGS
        filename = ['../plots/amd_times_' tag '.png'];
        fprintf('Saving figure to %s\n', filename);
        saveas(1, filename);
    end


    figure(2); clf; hold on;
    loglog(Ns, lnzs, linestyles);

    legend('Built-In AMD', 'AMD', 'CSparse AMD', 'Location', 'SouthEast');

    grid on;
    orient landscape;
    title(sprintf('AMD Quality Comparison (density = %.2f)', density));
    xlabel('Number of Columns');
    ylabel('Number of Non-zeros in Factorization');

    if SAVE_FIGS
        filename = ['../plots/amd_lnzs_' tag '.png'];
        fprintf('Saving figure to %s\n', filename);
        saveas(2, filename);
    end

end  % for tag = tags


%-------------------------------------------------------------------------------
%        Helper Functions
%-------------------------------------------------------------------------------
function [lnzs] = colamd_quality(A, q)
    [L, U, ~] = lu(A(:, q));
    lnzs = nnz(L) + nnz(U);
end

%===============================================================================
%===============================================================================
