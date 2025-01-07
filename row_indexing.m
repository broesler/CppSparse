%===============================================================================
%     File: row_indexing.m
%  Created: 2025-01-07 15:16
%   Author: Bernie Roesler
%
%  Solution to Davis Exercise 2.30: Experment with row indexing of sparse
%  matrices in MATLAB. Does it use binary search or linear search? Does it take
%  advantage of special cases like A(1, :) or A(M, :)?
%
%===============================================================================

clear;

% Ms = [10, 20, 50, 100, 200, 500, 1000];
Ms = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
density = 0.1;
N_samples = 7;

times = zeros(length(Ms), 1);
log_times = zeros(length(Ms), 1);

figure(1); clf; hold on;

% Create a random sparse matrix and time row indexing
for M = Ms
    A = create_sparse_matrix(M, M, density);

    row_times = zeros(N_samples, M);

    % Time row indexing for each row and average
    for s = 1:N_samples
        for i = 1:M
            tic;
            row = A(i, :);
            row_times(s, i) = toc;
        end
    end

    times(M == Ms) = mean(mean(row_times));

    if M == 5000
        % Plot the distribution of times for each row
        % hist(mean(row_times, 1));

        % Plot the of times for each row
        scatter(1:M, mean(row_times, 1), 'marker', '.');
        xlabel('Row index');
        ylabel('Time to index row (s)');
    end

    % Compute the sum of the log of each column size
    % col_sizes = sum(A ~= 0, 2);
    % log_times(M == Ms) = mean(log(col_sizes));
end

%-------------------------------------------------------------------------------
%        Plot the results
%-------------------------------------------------------------------------------
figure(2); clf; hold on;
% loglog(Ms, log_times, 'x-');
loglog(Ms, times, 'o-');
loglog(Ms, Ms * times(1) / Ms(1), '.-');  % faux linear relationship

legend('Time to index row', 'Linear scaling');

grid on;
orient landscape;
xlabel('Matrix size M');
ylabel('Time to index row (s)');

saveas(1, './data/row_indexing_distribution.png');
saveas(2, './data/row_indexing_scaling.png');


%===============================================================================
%===============================================================================
