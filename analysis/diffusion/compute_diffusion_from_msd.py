#!/usr/bin/env python3
"""Fit MSD data and compute diffusion coefficients from the Einstein relation.

Expected input: whitespace- or comma-delimited table with columns including time and MSD.
Supported MSD units: nm2 or A2. Time is assumed to be ns.
"""

import argparse
import numpy as np


def load_table(path: str) -> np.ndarray:
    try:
        return np.genfromtxt(path, names=True, comments="#", delimiter=None)
    except Exception:
        return np.genfromtxt(path, names=True, comments="#", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--msd-column", default="msd_nm2")
    parser.add_argument("--fit-start", type=float, default=6.0)
    parser.add_argument("--fit-end", type=float, default=10.0)
    parser.add_argument("--msd-unit", choices=["nm2", "A2"], default="nm2")
    args = parser.parse_args()

    data = load_table(args.input)
    time = data[args.time_column]
    msd = data[args.msd_column]

    mask = (time >= args.fit_start) & (time <= args.fit_end)
    if mask.sum() < 2:
        raise ValueError("Need at least two MSD points in fitting interval.")

    slope, intercept = np.polyfit(time[mask], msd[mask], 1)

    # Convert slope to m^2/s.
    # nm^2/ns = 1e-9 m^2/s; A^2/ns = 1e-11 m^2/s.
    conversion = 1e-9 if args.msd_unit == "nm2" else 1e-11
    D_m2_s = slope * conversion / 6.0
    D_1e_minus_8 = D_m2_s / 1e-8

    print(f"slope = {slope:.8g} {args.msd_unit}/ns")
    print(f"D = {D_m2_s:.8e} m^2/s")
    print(f"D = {D_1e_minus_8:.6f} x 10^-8 m^2/s")
    print(f"fit interval = {args.fit_start:g} to {args.fit_end:g} ns")


if __name__ == "__main__":
    main()
