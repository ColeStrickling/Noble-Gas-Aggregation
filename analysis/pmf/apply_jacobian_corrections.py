#!/usr/bin/env python3
"""
Example PMF post-processing script.

This script reads one WHAM output file, applies the appropriate geometric
Jacobian correction, shifts the dissociated tail to zero, and writes a processed
PMF file.

The WHAM input file is expected to have columns:

    coordinate_nm   free_kcal_mol   free_err_kcal_mol   prob   prob_err

Lines beginning with "#" are ignored.

Edit the USER SETTINGS section below for a different PMF.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_FILE = Path("examples/pmf/wham_output_example.dat")
OUTPUT_FILE = Path("results/pmf/processed_pmf_example.csv")

# Choose one: "two_body", "three_body", or "four_body"
GEOMETRY = "two_body"

TEMPERATURE_K = 1173.0

# PMF reference: subtract the average PMF over this distance range.
TAIL_MIN_A = 10.0
TAIL_MAX_A = 15.0

# =============================================================================
# CONSTANTS
# =============================================================================

R_KCAL_MOL_K = 0.00198720425864083
KCAL_TO_KJ = 4.184
EPS = 1.0e-12

JACOBIAN_POWER = {
    "two_body": 2,    # W(r) = F(r) + 2RT ln(r)
    "three_body": 1,  # W(l) = F(l) +  RT ln(l)
    "four_body": 0,   # W(s) = F(s)
}


def read_wham_file(path):
    """Read WHAM output."""
    columns = [
        "coordinate_nm",
        "free_kcal_mol",
        "free_err_kcal_mol",
        "prob",
        "prob_err",
    ]

    data = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        engine="python",
    )

    if data.empty:
        raise ValueError(f"No data were read from {path}")

    return data


def apply_jacobian_correction(coordinate_nm, free_kcal_mol, geometry):
    """Apply the geometry-specific Jacobian correction."""
    if geometry not in JACOBIAN_POWER:
        raise ValueError(
            f"Unknown GEOMETRY = {geometry}. "
            "Use 'two_body', 'three_body', or 'four_body'."
        )

    power = JACOBIAN_POWER[geometry]

    if power == 0:
        return free_kcal_mol.copy()

    coordinate_nm = np.clip(coordinate_nm, EPS, None)
    correction = power * R_KCAL_MOL_K * TEMPERATURE_K * np.log(coordinate_nm)

    return free_kcal_mol + correction


def shift_tail_to_zero(coordinate_A, pmf_kcal_mol):
    """Shift PMF so the average value in the tail window is zero."""
    tail_mask = (coordinate_A >= TAIL_MIN_A) & (coordinate_A <= TAIL_MAX_A)

    if np.any(tail_mask):
        tail_average = np.mean(pmf_kcal_mol[tail_mask])
    else:
        # Fallback if the selected tail window is outside the data range.
        n_tail = min(20, len(pmf_kcal_mol))
        tail_average = np.mean(pmf_kcal_mol[-n_tail:])

    return pmf_kcal_mol - tail_average


def main():
    data = read_wham_file(INPUT_FILE)

    coordinate_nm = data["coordinate_nm"].to_numpy(float)
    coordinate_A = 10.0 * coordinate_nm
    free_kcal_mol = data["free_kcal_mol"].to_numpy(float)

    pmf_kcal_mol = apply_jacobian_correction(
        coordinate_nm=coordinate_nm,
        free_kcal_mol=free_kcal_mol,
        geometry=GEOMETRY,
    )

    pmf_kcal_mol = shift_tail_to_zero(
        coordinate_A=coordinate_A,
        pmf_kcal_mol=pmf_kcal_mol,
    )

    output = pd.DataFrame({
        "coordinate_nm": coordinate_nm,
        "coordinate_A": coordinate_A,
        "wham_free_kcal_mol": free_kcal_mol,
        "pmf_kcal_mol": pmf_kcal_mol,
        "pmf_kJ_mol": pmf_kcal_mol * KCAL_TO_KJ,
        "free_err_kcal_mol": data["free_err_kcal_mol"].to_numpy(float),
        "prob": data["prob"].to_numpy(float),
        "prob_err": data["prob_err"].to_numpy(float),
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"Geometry: {GEOMETRY}")
    print(f"Input:    {INPUT_FILE}")
    print(f"Output:   {OUTPUT_FILE}")
    print(f"Rows:     {len(output)}")
    print()
    print(output.head())


if __name__ == "__main__":
    main()

