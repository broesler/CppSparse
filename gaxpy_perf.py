#!/usr/bin/env python3
# =============================================================================
#     File: gaxpy_perf.py
#  Created: 2025-01-07 12:49
#   Author: Bernie Roesler
#
"""
Plot the gaxpy performance data.
"""
# =============================================================================

import json
import matplotlib.pyplot as plt
import numpy as np

with open('gaxpy_perf.json', 'r') as f:
    data = json.load(f)

Ns = data['Ns']  # N values
del data['Ns']
times = data     # all other values are the times
del data

# Plot the data
fig, ax = plt.subplots(num=1, clear=True)
for key in times:
    ax.plot(Ns, times[key]['mean'], marker='.', label=key)
    ax.errorbar(Ns, times[key]['mean'],
                yerr=times[key]['std_dev'], fmt='none')

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid('both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')

plt.show()

# =============================================================================
# =============================================================================
