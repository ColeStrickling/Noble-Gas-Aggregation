#!/usr/bin/env python3
"""
Compute time-origin averaged MSD(t) for each atom type across multiple replicas,
then average MSD across runs with SEM(t), and compute D with per-run fits.

Assumptions:
- LAMMPS dump is "custom" format and includes id, type, xu, yu, zu
- dump is sorted by id (dump_modify sort id)
- atom types are stable across time

Expected folder layout:
  ./run_1_seed_12345/
      all_traj.lammpstrj   (or .gz)
      gas_traj.lammpstrj   (or .gz) [optional]
  ./run_2_seed_22345/
      ...

Outputs (per type):
  msd_type{t}_mean_sem.dat
  msd_type{t}_per_run.dat
  D_summary.csv
  D_per_run.csv
"""

import glob
import gzip
import numpy as np
from pathlib import Path

# -------------------------
# USER SETTINGS (edit these)
# -------------------------
RUN_GLOB = "run_*_seed_*"

# Trajectory file names inside each run dir
ALL_DUMP = "all_traj.lammpstrj"   # used for ions (and gas if GAS_DUMP missing)
GAS_DUMP = "gas_traj.lammpstrj"   # optional, used for type-1 gas if present

# LAMMPS integration timestep (fs) used to convert timestep -> ps
DT_FS = 1.0

# Atom-type mapping you want to analyze
# (works for LiCl/KCl and Ar/Xe as long as you keep gas as type 1)
TYPES_TO_ANALYZE = [1, 2, 3]  # 1=gas, 2=Cl, 3=Li or K

# Fit window (ns) for D. Keep it fixed across systems for comparability.
FIT_TMIN_NS = 6.0
FIT_TMAX_NS = 10.0

# For time-origin MSD, avoid long-lag bias: don't fit beyond half trajectory.
ENFORCE_TMAX_HALF_TRAJ = True

# Weighted fit (recommended): weights ~ sqrt(# time origins) ~ sqrt(T - lag_index)
USE_WEIGHTED_FIT = True

# Truncate runs to the minimum number of frames (per type) so all align
TRUNCATE_TO_MIN_FRAMES = True

# Bootstrap uncertainty on mean D across runs (per type)
BOOTSTRAP = True
NBOOT = 500
BOOT_SEED = 123


# -------------------------
# Dump parsing
# -------------------------
def open_maybe_gz(path: Path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "rt")

def resolve_dump(path_no_gz: Path):
    if path_no_gz.exists():
        return path_no_gz
    gz = Path(str(path_no_gz) + ".gz")
    if gz.exists():
        return gz
    return None

def read_dump_subset(path: Path, select_types, want=("id", "type", "xu", "yu", "zu")):
    """
    Reads a LAMMPS dump and stores ONLY atoms whose 'type' is in select_types.

    Returns:
      steps: (T,)
      out: dict {atype: pos (T, Ntype, 3)} with stable ordering (requires sort id)
      dt_ps_per_frame: inferred from timestep differences + DT_FS
    """
    select_types = set(select_types)
    steps = []
    out_lists = {t: [] for t in select_types}
    ids_ref = {t: None for t in select_types}

    with open_maybe_gz(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"{path}: expected ITEM: TIMESTEP, got: {line.strip()}")

            step = int(f.readline().strip())
            steps.append(step)

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(f"{path}: expected ITEM: NUMBER OF ATOMS")
            n = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"{path}: expected ITEM: BOX BOUNDS")
            # 3 lines box bounds
            f.readline(); f.readline(); f.readline()

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError(f"{path}: expected ITEM: ATOMS")
            header = line.strip().split()[2:]
            idx = {name: i for i, name in enumerate(header)}
            missing = [c for c in want if c not in idx]
            if missing:
                raise RuntimeError(f"{path}: missing columns {missing}. Have: {header}")

            ncols = len(header)
            atom_lines = [f.readline() for _ in range(n)]
            block = "".join(atom_lines)
            arr = np.fromstring(block, sep=" ", dtype=np.float64)
            if arr.size != n * ncols:
                raise RuntimeError(f"{path}: could not parse atom block at step {step}")
            arr = arr.reshape((n, ncols))

            ids = arr[:, idx["id"]].astype(np.int64)
            atypes = arr[:, idx["type"]].astype(np.int64)

            # For each selected type, grab subset rows (requires sort id for stable ordering)
            for t in select_types:
                m = (atypes == t)
                if not np.any(m):
                    raise RuntimeError(f"{path}: no atoms of type {t} found (step {step})")

                ids_t = ids[m]
                pos_t = np.stack(
                    [arr[m, idx["xu"]], arr[m, idx["yu"]], arr[m, idx["zu"]]],
                    axis=1
                ).astype(np.float32)

                if ids_ref[t] is None:
                    ids_ref[t] = ids_t.copy()
                else:
                    if not np.array_equal(ids_t, ids_ref[t]):
                        raise RuntimeError(
                            f"{path}: atom ID ordering changed for type {t}. "
                            f"Use 'dump_modify ... sort id'."
                        )

                out_lists[t].append(pos_t)

    steps = np.array(steps, dtype=np.int64)
    # stack to arrays
    out = {t: np.stack(out_lists[t], axis=0) for t in select_types}  # (T,N,3)

    # infer dt ps
    dt_ps = infer_dt_ps(steps, DT_FS)
    return steps, out, dt_ps

def infer_dt_ps(steps, dt_fs):
    d = np.diff(steps)
    dpos = d[d > 0]
    stride = float(np.median(dpos)) if dpos.size else 1.0
    return (stride * dt_fs) / 1000.0  # ps per frame


# -------------------------
# Time-origin MSD via FFT
# -------------------------
def msd_time_origin_fft(pos):
    """
    pos: (T, N, 3)
    returns msd: (T,) for lag indices 0..T-1
    """
    pos = np.asarray(pos, dtype=np.float32)
    T, N, _ = pos.shape
    nfft = 2 * T

    S2_total = np.zeros(T, dtype=np.float64)
    Psum = None

    for d in range(3):
        X = pos[:, :, d].astype(np.float32)  # (T,N)
        S2_total += np.sum(X * X, axis=1, dtype=np.float64)

        F = np.fft.rfft(X, n=nfft, axis=0)
        Pd = np.sum(F * np.conjugate(F), axis=1)  # sum over particles
        Psum = Pd if Psum is None else (Psum + Pd)

    C_total = np.fft.irfft(Psum, n=nfft)[:T].astype(np.float64)

    prefix = np.concatenate(([0.0], np.cumsum(S2_total)))
    tau = np.arange(T, dtype=np.int64)

    sumA = prefix[T - tau] - prefix[0]
    sumB = prefix[T] - prefix[tau]

    numerator = sumA + sumB - 2.0 * C_total
    denom = (T - tau).astype(np.float64) * float(N)

    msd = numerator / denom
    msd[0] = 0.0
    return msd


# -------------------------
# Fit D from MSD
# -------------------------
def fit_diffusion(time_ps, msd_A2, tmin_ns, tmax_ns, use_weighted=True):
    """
    Fit MSD = slope*t + intercept over [tmin,tmax], return D and diagnostics.
    Weighted fit uses weights ~ sqrt(# time origins) = sqrt(T - lag_index).
    """
    tmin_ps = tmin_ns * 1000.0
    tmax_ps = tmax_ns * 1000.0

    m = (time_ps >= tmin_ps) & (time_ps <= tmax_ps)
    if np.count_nonzero(m) < 10:
        raise RuntimeError("Not enough points in fit window.")

    x = time_ps[m]
    y = msd_A2[m]

    if use_weighted:
        idx = np.where(m)[0].astype(np.int64)
        T = len(time_ps)
        origins = (T - idx).astype(np.float64)
        w = np.sqrt(np.maximum(origins, 1.0))
        slope, intercept = np.polyfit(x, y, 1, w=w)
        yhat = slope * x + intercept

        w2 = w * w
        ybar_w = np.sum(w2 * y) / np.sum(w2)
        ss_res_w = np.sum(w2 * (y - yhat) ** 2)
        ss_tot_w = np.sum(w2 * (y - ybar_w) ** 2)
        r2_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan
    else:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        r2_w = np.nan

    # unweighted R^2
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # D = slope/6 in Å^2/ps; convert Å^2/ps -> cm^2/s via 1e-4
    D_cm2_s = (slope / 6.0) * 1e-4
    return D_cm2_s, r2, r2_w, slope, intercept


# -------------------------
# Main analysis
# -------------------------
def main():
    run_dirs = sorted(glob.glob(RUN_GLOB))
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {RUN_GLOB}")

    # Collect per-type per-run MSD arrays
    msd_by_type = {t: [] for t in TYPES_TO_ANALYZE}
    dt_by_type = {t: [] for t in TYPES_TO_ANALYZE}
    run_names = []

    for rd_str in run_dirs:
        rd = Path(rd_str)
        # prefer gas dump for type-1 if present
        gas_path = resolve_dump(rd / GAS_DUMP)
        all_path = resolve_dump(rd / ALL_DUMP)

        if all_path is None:
            print(f"Skipping {rd.name}: missing {ALL_DUMP} (or .gz)")
            continue

        # Types 2/3 (and 1 if we don't have gas dump) come from all_traj
        need_from_all = set(TYPES_TO_ANALYZE)
        if gas_path is not None and 1 in need_from_all:
            need_from_all.remove(1)

        # Read from all_traj
        steps_all, pos_all_dict, dt_all = read_dump_subset(all_path, need_from_all)
        # Read from gas_traj (type 1) if available
        if gas_path is not None and 1 in TYPES_TO_ANALYZE:
            steps_g, pos_g_dict, dt_g = read_dump_subset(gas_path, [1])
        else:
            pos_g_dict = {}

        # Compute MSD per type for this run
        for t in TYPES_TO_ANALYZE:
            if t == 1 and (t in pos_g_dict):
                pos_t = pos_g_dict[t]
                dt_ps = dt_g
            else:
                pos_t = pos_all_dict[t]
                dt_ps = dt_all

            msd_t = msd_time_origin_fft(pos_t)
            msd_by_type[t].append(msd_t)
            dt_by_type[t].append(dt_ps)

        run_names.append(rd.name)

    if not run_names:
        raise SystemExit("No usable runs found.")

    n_runs = len(run_names)
    print(f"\nLoaded {n_runs} runs: {', '.join(run_names[:3])}" + (" ..." if n_runs > 3 else ""))

    # For each type: align frames (truncate), compute mean/SEM MSD, compute per-run D, summarize
    D_rows = []
    per_run_rows = []

    for t in TYPES_TO_ANALYZE:
        msd_list = msd_by_type[t]
        dt_list = dt_by_type[t]

        # Ensure dt consistent for this type
        dt0 = dt_list[0]
        if any(abs(dt - dt0) > 1e-6 for dt in dt_list):
            raise RuntimeError(f"Type {t}: runs have different dt_ps_per_frame: {dt_list}")

        # Align lengths
        lens = [len(m) for m in msd_list]
        T_min = min(lens)
        if TRUNCATE_TO_MIN_FRAMES and any(L != T_min for L in lens):
            msd_list = [m[:T_min] for m in msd_list]

        msd_mat = np.vstack(msd_list)  # (Nruns, T)
        time_ps = np.arange(msd_mat.shape[1], dtype=np.float64) * dt0

        mean_msd = np.mean(msd_mat, axis=0)
        std_msd = np.std(msd_mat, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_msd)
        sem_msd = std_msd / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_msd)

        # Write MSD mean+SEM
        out_msd = np.column_stack([time_ps, mean_msd, sem_msd, std_msd])
        np.savetxt(
            f"msd_type{t}_mean_sem.dat",
            out_msd,
            header="time_ps  msd_mean_A2  msd_SEM_A2  msd_STD_A2",
            fmt="%.6f %.8e %.8e %.8e"
        )
        print(f"Wrote msd_type{t}_mean_sem.dat  (dt={dt0:.3f} ps, T={len(time_ps)})")

        # Also write per-run MSDs (handy for debugging/plotting)
        out_pr = np.column_stack([time_ps] + [msd_mat[i] for i in range(n_runs)])
        header = "time_ps " + " ".join([f"msd_run{i+1}_A2" for i in range(n_runs)])
        np.savetxt(
            f"msd_type{t}_per_run.dat",
            out_pr,
            header=header,
            fmt="%.6f " + " ".join(["%.8e"] * n_runs)
        )

        # Choose fit window (simple + consistent)
        tmax_ns = FIT_TMAX_NS
        if ENFORCE_TMAX_HALF_TRAJ:
            traj_ns = (time_ps[-1] / 1000.0)
            half_ns = 0.5 * traj_ns
            if tmax_ns > half_ns:
                tmax_ns = half_ns  # auto-shrink to half-length
        tmin_ns = FIT_TMIN_NS

        # Per-run D fits
        Ds = []
        r2s = []
        r2ws = []
        slopes = []

        for i in range(n_runs):
            D, r2, r2w, slope, _ = fit_diffusion(
                time_ps, msd_mat[i],
                tmin_ns, tmax_ns,
                use_weighted=USE_WEIGHTED_FIT
            )
            Ds.append(D); r2s.append(r2); r2ws.append(r2w); slopes.append(slope)
            per_run_rows.append((t, run_names[i], D, r2, r2w, slope, tmin_ns, tmax_ns))

        Ds = np.array(Ds, dtype=float)
        D_mean = float(np.mean(Ds))
        D_std = float(np.std(Ds, ddof=1)) if n_runs > 1 else float("nan")
        D_sem = D_std / np.sqrt(n_runs) if n_runs > 1 else float("nan")

        # Bootstrap CI on mean D (across runs)
        if BOOTSTRAP and n_runs > 1:
            rng = np.random.default_rng(BOOT_SEED + 1000 * t)
            boots = np.empty(NBOOT, dtype=float)
            for b in range(NBOOT):
                picks = rng.integers(0, n_runs, size=n_runs)
                boots[b] = float(np.mean(Ds[picks]))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        else:
            ci_lo, ci_hi = (np.nan, np.nan)

        D_rows.append((t, n_runs, D_mean, D_std, D_sem, ci_lo, ci_hi, tmin_ns, tmax_ns))

    # Write D summary
    with open("D_summary.csv", "w") as f:
        f.write("type,N_runs,D_mean_cm2_s,D_std_cm2_s,D_SEM_cm2_s,boot_ci2p5,boot_ci97p5,fit_tmin_ns,fit_tmax_ns\n")
        for row in D_rows:
            f.write(",".join([str(row[0]), str(row[1]),
                              f"{row[2]:.10e}", f"{row[3]:.10e}", f"{row[4]:.10e}",
                              f"{row[5]:.10e}", f"{row[6]:.10e}",
                              f"{row[7]:.6f}", f"{row[8]:.6f}"]) + "\n")
    print("\nWrote D_summary.csv")

    # Write per-run D
    with open("D_per_run.csv", "w") as f:
        f.write("type,run_dir,D_cm2_s,R2,R2_weighted,slope_A2_per_ps,fit_tmin_ns,fit_tmax_ns\n")
        for (t, rname, D, r2, r2w, slope, tmin, tmax) in per_run_rows:
            f.write(f"{t},{rname},{D:.10e},{r2:.6f},{r2w:.6f},{slope:.10e},{tmin:.6f},{tmax:.6f}\n")
    print("Wrote D_per_run.csv")

    print("\nDONE.")
    print("Reportable MSD uncertainty: SEM(t) in msd_type*_mean_sem.dat")
    print("Reportable D uncertainty: mean±SEM across runs (D_summary.csv) and bootstrap CI")


if __name__ == "__main__":
    main()

