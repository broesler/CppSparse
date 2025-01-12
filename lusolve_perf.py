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

density = data['density']
del data['density']

b_densities = np.r_[data['Ns']] / 1000
del data['Ns']

times = data     # all other values are the times
del data

# Plot the data
fig, ax = plt.subplots(num=1, clear=True)
for i, (key, val) in enumerate(times.items()):
    ax.errorbar(b_densities, val['mean'],
                yerr=val['std_dev'], ecolor=f"C{i}", fmt='.-', label=key)

ax.set_xscale('log')
# ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_xlabel('Density of RHS vector')
ax.set_ylabel('Time (s)')
ax.set_title(f"{filestem.split('_')[0]}, N = 2000, density {density}")

plt.show()

if SAVE_FIG:
    fig.savefig(f"./plots/{filestem}.png")

# =============================================================================
# =============================================================================
