function typical_time = timeit(func_handle, num_trials, num_samples)
% TIMEIT Measures the typical execution time of a function.
%   TYPICAL_TIME = TIMEIT(FUNC_HANDLE) measures the typical time
%   (in seconds) to execute the function specified by the function handle
%   FUNC_HANDLE. FUNC_HANDLE should be a handle to a function that takes
%   no input arguments.
%===============================================================================
%     File: timeit.m
%  Created: 2025-05-05 15:33
%   Author: Bernie Roesler
%===============================================================================

if nargin < 1
    error('timeit: Not enough input arguments.');
end

if ~isa(func_handle, 'function_handle')
    error('timeit: First input must be a function handle.');
end

if nargin < 2
    num_trials = 7;
end

if nargin < 3
    % Determine the number of runs for a total run time of at least 0.2 seconds.
    [num_samples, total_time] = autorange(func_handle, 0.2);
end

% Run multiple timing trials
% Run the inner loop multiple times to get several independent measurements.
trial_times = zeros(1, num_trials);

for j = 1:num_trials
    tic;
    for i = 1:num_samples
        func_handle();
    end
    trial_times(j) = toc;
end

% --- Step 4: Calculate a representative statistic ---
% The median is often used to minimize the effect of outliers.
% Divide by the number of inner runs to get the time per single execution.
typical_time = min(trial_times) / num_samples;

%===============================================================================
%===============================================================================
