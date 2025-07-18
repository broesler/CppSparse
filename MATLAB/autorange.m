function [num_samples, total_time] = autorange(func_handle, target_time)
% AUTORANGE Automatically determines the number of inner loop runs for timing.
%   NUM_SAMPLES = AUTORANGE(FUNC_HANDLE) runs the function specified by
%   FUNC_HANDLE multiple times to determine an appropriate number of
%   inner loop runs such that the total execution time is at least
%   % TARGET_TIME seconds. The function returns the number of samples
%   % NUM_SAMPLES that should be used for timing.
%   If TARGET_TIME is not specified, it defaults to 0.2 seconds (200 ms).
%
%   [NUM_SAMPLES, TOTAL_TIME] = AUTORANGE(FUNC_HANDLE) also returns the
%   % total execution time TOTAL_TIME for the determined number of samples.
%
%   [NUM_SAMPLES, TOTAL_TIME] = AUTORANGE(FUNC_HANDLE, TARGET_TIME) specifies
%   % the target time in seconds.
%
%   The function is primarily intended for use with the TIMEIT function when the
%   running time of the function is not known in advance.
%
%   Example:
%   >>> func_handle = @() pause(0.01);  % Example function that pauses for 10 ms
%   >>> [num_samples, total_time] = autorange(func_handle, 0.2);
%   >>> disp(['Number of samples: ', num2str(num_samples)]);
%   >>> disp(['Total time: ', num2str(total_time), ' seconds']);
%
%===============================================================================
%     File: autorange.m
%  Created: 2025-07-18 10:23
%   Author: Bernie Roesler
%===============================================================================

if nargin < 2
    target_time = 0.2;  % default: 200 ms
end

% Run the function a few times to allow for JIT compilation, caching, etc.
num_warmup_runs = 3;
for i = 1:num_warmup_runs
    func_handle();
end

% Run the function in a loop for a short duration to estimate how many
% runs are needed for a measurable time with tic/toc.
MAX_SAMPLES = 1000000;  % arbitrary maximum samples
N = 1;        % start with 1 sample

while N < MAX_SAMPLES  % arbitrary maximum samples
    tic;
    for i = 1:N
        func_handle();
    end
    total_time = toc;

    if total_time >= target_time
        break;
    end

    % If the estimated time is too short, double the number of samples
    N = N * 2;
end

num_samples = round_up(N);  % make it a "nice" number

end


function R = round_up(N)
% ROUND_UP Rounds a number up to the nearest "nice" number.
%
%    R = ROUND_UP(N) rounds the number N up to the nearest multiple of 
%    {1, 2, 5} x 10^k.

if N < 2
    R = 1;
    return;
end

pow10 = 10^floor(log10(N));

for k = [1, 2, 5, 10]
    R = k * pow10;
    if R >= N
        return;
    end
end

R = 10 * pow10;  % if nothing else works, return the next power of ten

end

%===============================================================================
%===============================================================================
