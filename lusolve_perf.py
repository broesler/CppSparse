#!/usr/bin/env python3
# =============================================================================
#     File: lusolve_perf.py
#  Created: 2025-01-11 10:57
#   Author: Bernie Roesler
#
"""
Plot the lusolve performance data.
"""
# =============================================================================

import json
import matplotlib.pyplot as plt
import numpy as np

SAVE_FIG = True

filestem = 'lusolve_perf'

with open(f"./plots/{filestem}.json", 'r') as f:
    data = json.load(f)

# Hack key names from other script
N = int(data['density'])
del data['density']

densities = np.r_[data['Ns']] / 1000
del data['Ns']

times = data     # all other values are the times
del data

# Plot the data
fig, axs = plt.subplots(num=1, nrows=2, sharex=True, clear=True)
fig.suptitle(f"{filestem.split('_')[0]}, N = {N}")

ax = axs[0]
for i, (key, val) in enumerate(times.items()):
    ax.errorbar(densities, val['mean'],
                yerr=val['std_dev'], ecolor=f"C{i}", fmt='.-', label=key)

# ax.set_xscale('log')
# ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_ylabel('Time (s)')

# Plot the difference between the two methods
ax = axs[1]
for i, k in enumerate(['l', 'u']):
    key = f"{k}solve"
    opt_key = key + '_opt'

    mean = np.r_[times[key]['mean']] 
    opt_mean = np.r_[times[opt_key]['mean']]
    rel_diff = (mean - opt_mean) / mean

    var = np.r_[times[key]['std_dev']]**2 
    opt_var = np.r_[times[opt_key]['std_dev']]**2

    ax.errorbar(densities, rel_diff, yerr=np.sqrt(var + opt_var),
                ecolor=f"C{i}", fmt='.-', label=key)

ax.grid(which='both')
ax.legend()

ax.set_xlabel('Density of Matrix and RHS vector')
ax.set_ylabel('Time Ratio of Original to Optimized')

plt.show()

if SAVE_FIG:
    fig.savefig(f"./plots/{filestem}.png")

# =============================================================================
# =============================================================================
