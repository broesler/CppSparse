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
    error('mock_timeit: Not enough input arguments.');
end

if ~isa(func_handle, 'function_handle')
    error('mock_timeit: First input must be a function handle.');
end

if nargin < 2
    num_trials = 10; % Default number of trials
end

if nargin < 3
    num_samples = 30; % Default number of samples
end

% --- Step 1: Warm-up runs ---
% Run the function a few times to allow for JIT compilation, caching, etc.
% This helps to get more representative timing results later.
num_warmup_runs = 5;
for i = 1:num_warmup_runs
    func_handle();
end

% --- Step 2: Determine an appropriate number of inner loop runs ---
% Run the function in a loop for a short duration to estimate how many
% runs are needed for a measurable time with tic/toc. Aim for at least
% a certain minimum time per inner loop execution.
min_inner_loop_time = 0.01; % Aim for at least 10 milliseconds per inner loop
estimated_time = 0;

while estimated_time < min_inner_loop_time
    tic;
    for i = 1:num_samples
        func_handle();
    end
    estimated_time = toc;

    if estimated_time < min_inner_loop_time
        % If the estimated time is too short, increase the number of runs
        % Increase exponentially to quickly reach a measurable time
        num_samples = num_samples * 2;
        % Add a safeguard to prevent infinite loops for very fast functions
        if num_samples > 1000000 % Arbitrary large limit
            warning('mock_timeit: Function is extremely fast, results may be less reliable.');
            break;
        end
    end
end

% --- Step 3: Run multiple timing trials (outer loop) ---
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
typical_time = median(trial_times) / num_samples;

%===============================================================================
%===============================================================================
