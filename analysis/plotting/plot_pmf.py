#!/usr/bin/env python3
"""
Apply geometry-specific Jacobian corrections to WHAM free-energy output.

This script converts WHAM output into PMFs for the aggregation geometries used in
this work:

    two_body    W(r) = F(r) + 2 R T ln(r)
    three_body  W(l) = F(l) +   R T ln(l)
    four_body   W(s) = F(s)

The corrected PMF is shifted so that the mean value in a user-specified tail
window is zero. By default, WHAM coordinates are assumed to be in nm and free
energies are assumed to be in kcal/mol.

Single-file example:
    python analysis/pmf/apply_jacobian_corrections.py \
        --input data/raw_wham/wham_output_ar_ar.dat \
        --geometry two_body \
        --output data/processed_pmf/licl_ar_two_body_pmf.csv

Batch example:
    python analysis/pmf/apply_jacobian_corrections.py \
        --manifest metadata/pmf_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


R_KCAL_MOL_K = 0.00198720425864083
KCAL_TO_KJ = 4.184
EPS = 1.0e-12

GEOMETRY_TO_POWER = {
    "two_body": 2,
    "three_body": 1,
    "four_body": 0,
}


def read_wham(path: str | Path) -> pd.DataFrame:
    """Read a standard WHAM output file."""
    names = ["coordinate_nm", "free_kcal_mol", "free_err_kcal_mol", "prob", "prob_err"]
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=names, engine="python")

    if df.empty:
        raise ValueError(f"No data were read from {path}")

    return df


def apply_jacobian(
    free_kcal_mol: np.ndarray,
    coordinate_nm: np.ndarray,
    geometry: str,
    temperature_K: float,
) -> np.ndarray:
    """Apply the Jacobian correction for the requested PMF geometry."""
    if geometry not in GEOMETRY_TO_POWER:
        allowed = ", ".join(GEOMETRY_TO_POWER)
        raise ValueError(f"Unknown geometry '{geometry}'. Choose one of: {allowed}")

    power = GEOMETRY_TO_POWER[geometry]
    if power == 0:
        return free_kcal_mol.copy()

    coord = np.clip(coordinate_nm.astype(float), EPS, None)
    correction = power * R_KCAL_MOL_K * temperature_K * np.log(coord)
    return free_kcal_mol + correction


def shift_tail_to_zero(
    coordinate_A: np.ndarray,
    pmf_kcal_mol: np.ndarray,
    tail_min_A: float,
    tail_max_A: float,
) -> tuple[np.ndarray, float]:
    """Shift the PMF so that the average value in the tail window is zero."""
    mask = (coordinate_A >= tail_min_A) & (coordinate_A <= tail_max_A)

    if np.any(mask):
        shift = float(np.mean(pmf_kcal_mol[mask]))
    else:
        n_tail = min(20, len(pmf_kcal_mol))
        shift = float(np.mean(pmf_kcal_mol[-n_tail:]))

    return pmf_kcal_mol - shift, shift


def process_wham_file(
    input_path: str | Path,
    output_path: str | Path,
    geometry: str,
    temperature_K: float = 1173.0,
    tail_min_A: float = 10.0,
    tail_max_A: float = 15.0,
) -> pd.DataFrame:
    """Read WHAM output, apply corrections, shift the tail, and write CSV output."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_wham(input_path)
    coordinate_nm = df["coordinate_nm"].to_numpy(float)
    coordinate_A = 10.0 * coordinate_nm
    free = df["free_kcal_mol"].to_numpy(float)

    pmf = apply_jacobian(
        free_kcal_mol=free,
        coordinate_nm=coordinate_nm,
        geometry=geometry,
        temperature_K=temperature_K,
    )
    pmf_shifted, tail_shift = shift_tail_to_zero(
        coordinate_A=coordinate_A,
        pmf_kcal_mol=pmf,
        tail_min_A=tail_min_A,
        tail_max_A=tail_max_A,
    )

    out = pd.DataFrame({
        "coordinate_nm": coordinate_nm,
        "coordinate_A": coordinate_A,
        "wham_free_kcal_mol": free,
        "pmf_kcal_mol": pmf_shifted,
        "pmf_kJ_mol": pmf_shifted * KCAL_TO_KJ,
        "free_err_kcal_mol": df["free_err_kcal_mol"].to_numpy(float),
        "prob": df["prob"].to_numpy(float),
        "prob_err": df["prob_err"].to_numpy(float),
    })

    out.to_csv(output_path, index=False)

    metadata = (
        f"input_file={input_path}\n"
        f"geometry={geometry}\n"
        f"temperature_K={temperature_K}\n"
        f"tail_window_A={tail_min_A}-{tail_max_A}\n"
        f"tail_shift_kcal_mol={tail_shift}\n"
    )
    output_path.with_suffix(output_path.suffix + ".metadata.txt").write_text(metadata)

    return out


def process_manifest(manifest_path: str | Path) -> None:
    """
    Process all rows in a PMF manifest.

    Required columns:
        input_file, geometry, output_file

    Optional columns:
        temperature_K, tail_min_A, tail_max_A
    """
    manifest = pd.read_csv(manifest_path)

    required = {"input_file", "geometry", "output_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    for _, row in manifest.iterrows():
        process_wham_file(
            input_path=row["input_file"],
            output_path=row["output_file"],
            geometry=row["geometry"],
            temperature_K=float(row.get("temperature_K", 1173.0)),
            tail_min_A=float(row.get("tail_min_A", 10.0)),
            tail_max_A=float(row.get("tail_max_A", 15.0)),
        )
        print(f"Wrote {row['output_file']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Jacobian corrections to WHAM output.")
    parser.add_argument("--input", help="Input WHAM output file.")
    parser.add_argument(
        "--geometry",
        choices=sorted(GEOMETRY_TO_POWER),
        help="PMF geometry: two_body, three_body, or four_body.",
    )
    parser.add_argument("--output", help="Output CSV file.")
    parser.add_argument("--temperature", type=float, default=1173.0, help="Temperature in K.")
    parser.add_argument("--tail-min", type=float, default=10.0, help="Tail-window minimum in Angstrom.")
    parser.add_argument("--tail-max", type=float, default=15.0, help="Tail-window maximum in Angstrom.")
    parser.add_argument("--manifest", help="Optional CSV manifest for batch processing.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.manifest:
        process_manifest(args.manifest)
        return

    if not (args.input and args.geometry and args.output):
        parser.error("Either provide --manifest or provide --input, --geometry, and --output.")

    process_wham_file(
        input_path=args.input,
        output_path=args.output,
        geometry=args.geometry,
        temperature_K=args.temperature,
        tail_min_A=args.tail_min,
        tail_max_A=args.tail_max,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
