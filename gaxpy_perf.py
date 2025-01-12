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

SAVE_FIG = True

filestem = 'gaxpy_perf'
# filestem = 'gatxpy_perf'

with open(f"./plots/{filestem}.json", 'r') as f:
    data = json.load(f)

density = data['density']
del data['density']
Ns = data['Ns']  # N values
del data['Ns']
times = data     # all other values are the times
del data

# Plot the data
fig, ax = plt.subplots(num=1, clear=True)
for i, (key, val) in enumerate(times.items()):
    ax.errorbar(Ns, val['mean'],
                yerr=val['std_dev'], ecolor=f"C{i}", fmt='.-', label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')
ax.set_title(f"{filestem.split('_')[0]}, density {density}")

plt.show()

if SAVE_FIG:
    fig.savefig(f"./plots/{filestem}.png")

# =============================================================================
# =============================================================================
