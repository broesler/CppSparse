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
Ns = data['Ns']  # N values
del data['Ns']
times = data     # all other values are the times
del data

# Plot the data
fig, ax = plt.subplots(num=1, clear=True)
for key in times:
    ax.plot(Ns, times[key]['mean'], marker='o', label=key)
    ax.errorbar(Ns, times[key]['mean'],
                yerr=times[key]['std_dev'], fmt='none')

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid('both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')
ax.set_title(f"{filestem.split('_')[0]}, density {density}")

plt.show()

if SAVE_FIG:
    fig.savefig(f"./plots/{filestem}.png")

# =============================================================================
# =============================================================================
