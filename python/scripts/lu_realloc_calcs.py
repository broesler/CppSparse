#!/usr/bin/env python3
# =============================================================================
#     File: lu_realloc_calcs.py
#  Created: 2025-04-26 13:51
#   Author: Bernie Roesler
#
"""
Scratch calculations to determine the threshold for realloc failure.
"""
# =============================================================================

N = 100
k = 3  # arbitrary column index

nnz = N  # number of non-zero elements in the matrix (choose to be N)

max_request = 2 * nnz + N
print('max request: ', max_request)

for lower in [True, False]:
    print('--- lower: ' if lower else '--- upper: ')

    # Compute the expected threshold for realloc failure
    min_request = (nnz + N - k) if lower else (nnz + k + 1)

    # 3 scenarios:
    #   min_request < max_request < threshold    (pass on first request)
    #   min_request < threshold   < max_request  (pass with multiple requests)
    #   threshold   < min_request < max_request  (fail with multiple requests)

    print('min request: ', min_request)

    # Print requests between max and min
    requests = []
    request = max_request
    while (request > min_request):
        requests.append(request)
        request = int(request + min_request) // 2

    print('requests: ', requests)

# -----------------------------------------------------------------------------
#         Expect results:
# -----------------------------------------------------------------------------
# for N == 100:
# max request:  300
# --- lower:
# min request:  197
# requests:  [300, 248, 222, 209, 203, 200, 198]
# --- upper:
# min request:  104
# requests:  [300, 202, 153, 128, 116, 110, 107, 105]


# =============================================================================
# =============================================================================
