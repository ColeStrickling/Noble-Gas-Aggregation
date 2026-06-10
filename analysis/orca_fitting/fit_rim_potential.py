#!/usr/bin/env python3
"""
Fit a RIM-style salt--gas interaction potential to an ORCA energy scan.

This script is a cleaned-up version of the fitting workflow used for the
gas--ion interaction scans. It reads one ORCA scan output file, converts the
total energies to relative interaction energies, fits a RIM-style potential,
and writes the fitted parameters and fit curve.

The fitted functional form is written in the same convention used in the
manuscript:

    U(r) = A exp((sigma - r) / rho) - C / r^6 - D / r^8

During fitting, the final term is represented as +D_rep/r^8 because this is
often more convenient for enforcing short-range repulsion. The reported
manuscript-style value is therefore

    D_manuscript = -D_rep

To use this example:
    1. Put a real ORCA scan output file in examples/fitting/.
    2. Edit the USER SETTINGS section below.
    3. Run from the repository root:

       python analysis/fitting/fit_rim_potential.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# USER SETTINGS
# =============================================================================

# Example expected location. Replace with the real file you want to fit.
INPUT_FILE = Path("examples/fitting/E_out_K_ion_Ar_eps_1_8.txt")

# Output files.
OUTPUT_PARAMETERS = Path("results/fitting/fitted_rim_parameters.csv")
OUTPUT_CURVE = Path("results/fitting/fitted_rim_curve.csv")

# ORCA parsing settings.
# The notebook used 228 for E_out_K_ion_Ar_eps_1_8.txt.
N_HEADER_LINES = 228

# The notebook often read every other line. Use 0 if the energy is on the first
# line after the header, or 1 if it is on the second line after the header.
FIRST_DATA_LINE_OFFSET = 0

# In the notebook, the energy was read from line.split()[4].
ENERGY_COLUMN_INDEX = 4

# Energy correction/reference in Hartree.
# In the notebook example:
#     E_relative = (E_orca + 1125.938526) * 627.503
#     E_relative -= E_relative[-1]
REFERENCE_ENERGY_HARTREE = 1125.938526

# Distance grid settings.
# This matches the notebook pattern:
#     fine grid:   1.20, 1.21, ...
#     coarse grid: 6.15, 6.25, ...
N_COARSE_POINTS = 90
FINE_START_A = 1.20
FINE_STEP_A = 0.01
COARSE_START_A = 6.15
COARSE_STEP_A = 0.10

# Fitting region.
FIT_R_MAX_A = 9.0
FIT_ENERGY_MIN_KCAL_MOL = -40.0
FIT_ENERGY_MAX_KCAL_MOL = 500.0

# Weighting. Values below zero are emphasized to better fit the attractive well.
NEGATIVE_ENERGY_WEIGHT = 10.0

# Initial guess and bounds for:
#     A, sigma, rho, C, D_rep
#
# The fitted potential used internally is:
#     U(r) = A exp((sigma-r)/rho) - C/r^6 + D_rep/r^8
INITIAL_GUESS = np.array([7.8, 2.5, 0.2, -1000.0, 5000.0])
BOUNDS = [
    (1.1, 9.75),       # A
    (0.1, 10.0),       # sigma
    (0.1, 1.0),        # rho
    (-8000.0, 10000.0),# C
    (-60000.0, 50000.0)# D_rep
]


# =============================================================================
# CONSTANTS
# =============================================================================

HARTREE_TO_KCAL_MOL = 627.503
KCAL_TO_KJ = 4.184


def read_orca_scan_energies(
    path,
    n_header_lines,
    first_data_line_offset,
    energy_column_index,
):
    """Read total energies from an ORCA scan output file."""
    energies = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(n_header_lines):
            next(f, None)

        lines = f.readlines()

    for i in range(first_data_line_offset, len(lines), 2):
        parts = lines[i].split()

        if len(parts) <= energy_column_index:
            continue

        try:
            energies.append(float(parts[energy_column_index]))
        except ValueError:
            continue

    if not energies:
        raise ValueError(f"No energies were read from {path}")

    return np.array(energies, dtype=float)


def build_distance_grid(n_points):
    """Build the scan distance grid in Angstroms."""
    n_fine = n_points - N_COARSE_POINTS

    if n_fine <= 0:
        raise ValueError(
            "The number of energy points is smaller than N_COARSE_POINTS. "
            "Check N_COARSE_POINTS and the parsed ORCA energies."
        )

    fine = FINE_START_A + FINE_STEP_A * np.arange(n_fine)
    coarse = COARSE_START_A + COARSE_STEP_A * np.arange(N_COARSE_POINTS)

    distances = np.concatenate([fine, coarse])

    if len(distances) != n_points:
        raise RuntimeError("Distance-grid length does not match energy length.")

    return distances


def convert_to_relative_energy_kcal(energies_hartree):
    """Convert total energies to relative interaction energies in kcal/mol."""
    relative = (energies_hartree + REFERENCE_ENERGY_HARTREE) * HARTREE_TO_KCAL_MOL

    # Set the largest sampled distance as the zero-energy reference.
    relative -= relative[-1]

    return relative


def rim_fit_potential(r_A, theta):
    """
    Fitting form:
        U(r) = A exp((sigma-r)/rho) - C/r^6 + D_rep/r^8

    The manuscript RIM convention is:
        U(r) = A exp((sigma-r)/rho) - C/r^6 - D/r^8

    Therefore:
        D_manuscript = -D_rep
    """
    A, sigma, rho, C, D_rep = theta
    return A * np.exp((sigma - r_A) / rho) - C / r_A**6 + D_rep / r_A**8


def weighted_sse(theta, r_A, energy_kcal_mol):
    """Weighted sum of squared errors."""
    residuals = energy_kcal_mol - rim_fit_potential(r_A, theta)

    weights = np.ones_like(energy_kcal_mol, dtype=float)
    weights[energy_kcal_mol < 0.0] *= NEGATIVE_ENERGY_WEIGHT

    return np.sum((weights * residuals) ** 2)


def fit_potential(r_A, energy_kcal_mol):
    """Fit the RIM-style potential."""
    fit_mask = (
        (r_A <= FIT_R_MAX_A)
        & (energy_kcal_mol >= FIT_ENERGY_MIN_KCAL_MOL)
        & (energy_kcal_mol <= FIT_ENERGY_MAX_KCAL_MOL)
    )

    r_fit = r_A[fit_mask]
    e_fit = energy_kcal_mol[fit_mask]

    if len(r_fit) < 5:
        raise ValueError("Too few points remain after applying the fit filters.")

    result = minimize(
        weighted_sse,
        INITIAL_GUESS,
        args=(r_fit, e_fit),
        method="L-BFGS-B",
        bounds=BOUNDS,
        tol=1.0e-12,
    )

    theta = result.x
    e_pred = rim_fit_potential(r_fit, theta)
    residuals = e_fit - e_pred

    sse = float(np.sum(residuals**2))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_total = float(np.sum((e_fit - np.mean(e_fit))**2))
    r_squared = float(1.0 - sse / ss_total) if ss_total != 0.0 else np.nan

    metrics = {
        "success": bool(result.success),
        "message": result.message,
        "n_fit_points": len(r_fit),
        "SSE_kcal2_mol2": sse,
        "MSE_kcal2_mol2": mse,
        "RMSE_kcal_mol": rmse,
        "MAE_kcal_mol": mae,
        "R_squared": r_squared,
    }

    return theta, metrics, fit_mask


def write_outputs(r_A, energy_kcal_mol, theta, metrics):
    """Write fitted parameters and fitted curve."""
    A, sigma, rho, C, D_rep = theta
    D_manuscript = -D_rep

    parameter_row = {
        "A_kcal_mol": A,
        "sigma_A": sigma,
        "rho_A": rho,
        "C_kcal_mol_A6": C,
        "D_rep_kcal_mol_A8": D_rep,
        "D_manuscript_kcal_mol_A8": D_manuscript,
        "A_kJ_mol": A * KCAL_TO_KJ,
        "C_kJ_mol_A6": C * KCAL_TO_KJ,
        "D_manuscript_kJ_mol_A8": D_manuscript * KCAL_TO_KJ,
        **metrics,
    }

    curve = pd.DataFrame({
        "r_A": r_A,
        "qm_energy_kcal_mol": energy_kcal_mol,
        "fit_energy_kcal_mol": rim_fit_potential(r_A, theta),
        "qm_energy_kJ_mol": energy_kcal_mol * KCAL_TO_KJ,
        "fit_energy_kJ_mol": rim_fit_potential(r_A, theta) * KCAL_TO_KJ,
    })

    OUTPUT_PARAMETERS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CURVE.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([parameter_row]).to_csv(OUTPUT_PARAMETERS, index=False)
    curve.to_csv(OUTPUT_CURVE, index=False)

    return parameter_row


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Place a real ORCA scan output file at this path or edit INPUT_FILE."
        )

    energies_hartree = read_orca_scan_energies(
        path=INPUT_FILE,
        n_header_lines=N_HEADER_LINES,
        first_data_line_offset=FIRST_DATA_LINE_OFFSET,
        energy_column_index=ENERGY_COLUMN_INDEX,
    )

    r_A = build_distance_grid(len(energies_hartree))
    energy_kcal_mol = convert_to_relative_energy_kcal(energies_hartree)

    theta, metrics, fit_mask = fit_potential(r_A, energy_kcal_mol)
    parameter_row = write_outputs(r_A, energy_kcal_mol, theta, metrics)

    print("Fit complete.")
    print(f"Input:      {INPUT_FILE}")
    print(f"Parameters: {OUTPUT_PARAMETERS}")
    print(f"Fit curve:  {OUTPUT_CURVE}")
    print()
    print("Fitted parameters in manuscript convention:")
    print(f"A      = {parameter_row['A_kJ_mol']:.6f} kJ/mol")
    print(f"sigma  = {parameter_row['sigma_A']:.6f} A")
    print(f"rho    = {parameter_row['rho_A']:.6f} A")
    print(f"C      = {parameter_row['C_kJ_mol_A6']:.6f} kJ mol^-1 A^6")
    print(f"D      = {parameter_row['D_manuscript_kJ_mol_A8']:.6f} kJ mol^-1 A^8")
    print()
    print("Fit quality:")
    print(f"RMSE   = {metrics['RMSE_kcal_mol']:.6f} kcal/mol")
    print(f"MAE    = {metrics['MAE_kcal_mol']:.6f} kcal/mol")
    print(f"R^2    = {metrics['R_squared']:.6f}")


if __name__ == "__main__":
    main()
