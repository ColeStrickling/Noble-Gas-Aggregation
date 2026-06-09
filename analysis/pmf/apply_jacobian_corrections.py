#!/usr/bin/env python3
"""Apply geometric/Jacobian corrections to WHAM free-energy output.

Input format: whitespace-delimited file with coordinate in column 1 and free energy in column 2.
Default units are kJ/mol, using R = 0.008314462618 kJ/mol/K.
"""

import argparse
import numpy as np


R_KJ_MOL_K = 0.008314462618


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WHAM output file")
    parser.add_argument("--output", required=True, help="Corrected PMF output file")
    parser.add_argument("--geometry", required=True, choices=["two_body", "three_body", "four_body"])
    parser.add_argument("--temperature", type=float, default=1173.0)
    parser.add_argument("--coordinate-column", type=int, default=0)
    parser.add_argument("--free-energy-column", type=int, default=1)
    parser.add_argument("--shift-min", type=float, default=None, help="Minimum coordinate for zero-shift region")
    parser.add_argument("--shift-max", type=float, default=None, help="Maximum coordinate for zero-shift region")
    args = parser.parse_args()

    data = np.loadtxt(args.input, comments=["#", "@"])
    q = data[:, args.coordinate_column]
    F = data[:, args.free_energy_column]

    if np.any(q <= 0) and args.geometry in {"two_body", "three_body"}:
        raise ValueError("Coordinates must be positive for logarithmic Jacobian corrections.")

    kBT = R_KJ_MOL_K * args.temperature

    if args.geometry == "two_body":
        W = F + 2.0 * kBT * np.log(q)
    elif args.geometry == "three_body":
        W = F + kBT * np.log(q)
    else:
        W = F.copy()

    if args.shift_min is not None and args.shift_max is not None:
        mask = (q >= args.shift_min) & (q <= args.shift_max)
        if not np.any(mask):
            raise ValueError("No points found in requested shift range.")
        shift = np.nanmean(W[mask])
    else:
        shift = W[-1]

    W = W - shift

    header = "coordinate corrected_pmf_kJ_mol raw_free_energy_kJ_mol"
    np.savetxt(args.output, np.column_stack([q, W, F]), header=header)


if __name__ == "__main__":
    main()
