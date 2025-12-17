#!/usr/bin/env python3
# =============================================================================
#     File: utils.py
#  Created: 2025-12-17 09:33
#   Author: Bernie Roesler
# =============================================================================

"""Utilities for scikit-sparse testing."""

import gc
import timeit
import tracemalloc

import numpy as np


def measure_perf(func, N_repeats=5, N_samples=None):
    """Measure time and memory usage of a function.

    Parameters
    ----------
    func : callable
        The function to measure.

    Returns
    -------
    time : float
        The minimum execution time in seconds.
    peak_mb : float
        The peak memory usage in megabytes.
    """
    # Measure timing (multiple runs)
    timer = timeit.Timer(func)
    if N_samples is None:
        N_samples, _ = timer.autorange()
    ts = timer.repeat(repeat=N_repeats, number=N_samples)
    ts = np.array(ts) / N_samples
    time = np.min(ts)

    # Measure memory usage (single pass)
    gc.collect()  # force garbage collection before measuring
    tracemalloc.start()

    try:
        func()
    except Exception:
        tracemalloc.stop()
        raise

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024**2)  # convert to MB

    return time, peak_mb


# =============================================================================
# =============================================================================
