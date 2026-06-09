#!/usr/bin/env python3
"""
Example diffusion analysis script.

This script reads one MSD file, fits the linear region, and computes a diffusion
coefficient from the Einstein relation:

    D = (1/6) d<|r(t)-r(0)|^2>/dt

The input file is expected to have at least two whitespace-delimited columns:

    time_ps   msd_A2

Additional columns are allowed and ignored.

Edit the USER SETTINGS section below for a different MSD file or fit window.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_FILE = Path("examples/diffusion/msd_example.dat")
OUTPUT_FILE = Path("results/diffusion/diffusion_example.csv")

SPECIES_LABEL = "Ar"
FIT_MIN_NS = 6.0
FIT_MAX_NS = 10.0

# =============================================================================
# CONSTANTS
# =============================================================================

A2_PER_PS_TO_M2_PER_S = 1.0e-8


def read_msd_file(path):
    """Read time in ps and MSD in A^2."""
    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: time_ps msd_A2")

    time_ps = data[:, 0].astype(float)
    msd_A2 = data[:, 1].astype(float)

    return time_ps, msd_A2


def fit_diffusion(time_ps, msd_A2):
    """Fit MSD versus time and compute D."""
    time_ns = time_ps / 1000.0
    fit_mask = (time_ns >= FIT_MIN_NS) & (time_ns <= FIT_MAX_NS)

    if np.count_nonzero(fit_mask) < 2:
        raise ValueError(
            f"The fit window {FIT_MIN_NS}-{FIT_MAX_NS} ns contains fewer than two points."
        )

    slope_A2_per_ps, intercept_A2 = np.polyfit(
        time_ps[fit_mask],
        msd_A2[fit_mask],
        deg=1,
    )

    diffusion_m2_s = (slope_A2_per_ps / 6.0) * A2_PER_PS_TO_M2_PER_S

    return {
        "species": SPECIES_LABEL,
        "input_file": str(INPUT_FILE),
        "fit_min_ns": FIT_MIN_NS,
        "fit_max_ns": FIT_MAX_NS,
        "slope_A2_per_ps": slope_A2_per_ps,
        "intercept_A2": intercept_A2,
        "D_m2_s": diffusion_m2_s,
        "D_1e8_m2_s": diffusion_m2_s / 1.0e-8,
    }


def main():
    time_ps, msd_A2 = read_msd_file(INPUT_FILE)
    result = fit_diffusion(time_ps, msd_A2)

    output = pd.DataFrame([result])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"Species:  {SPECIES_LABEL}")
    print(f"Input:    {INPUT_FILE}")
    print(f"Output:   {OUTPUT_FILE}")
    print()
    print(f"D = {result['D_m2_s']:.4e} m^2/s")
    print(f"D = {result['D_1e8_m2_s']:.3f} x 10^-8 m^2/s")


if __name__ == "__main__":
    main()
