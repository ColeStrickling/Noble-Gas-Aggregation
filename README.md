# Noble Gas Aggregation and Dynamics in Molten Alkali Chloride Salts

This repository contains analysis scripts, example input files, force-field parameter files, and processed data templates needed to reproduce the simulations and analyses reported in:

> Cole Strickling, Yong Zhang, Luke Gibson, Vyacheslav Bryantsev, and Edward J. Maginn,  
> **Noble Gas Aggregation and Dynamics in Molten Alkali Chloride Salts**.  
> Manuscript in preparation / submitted. Update citation information after publication.

## Repository contents

```text
.
├── analysis/              # Python scripts used for post-processing and figure generation
├── examples/              # Minimal runnable examples for key workflows
├── force_fields/          # Salt-salt, gas-gas, and salt-gas parameters
├── inputs/                # LAMMPS, PLUMED, CP2K, and ORCA input examples
├── CITATION.cff
├── environment.yml
├── requirements.txt
└── README.md
```

## Software used

The work uses several simulation and analysis tools. Record the exact versions used on your cluster in `metadata/software_versions.txt`.

- LAMMPS for classical molecular dynamics
- PLUMED interfaced with LAMMPS for umbrella sampling
- WHAM for free-energy reconstruction
- CP2K/Quickstep for AIMD reference simulations
- ORCA for gas-ion quantum mechanical interaction scans
- Python for RDF, PMF, MSD, diffusion, and plotting analyses

## Workflow

1. **Develop salt-gas interaction parameters**
   - Run ORCA gas-ion interaction scans for Ar-ion and Xe-ion pairs.
   - Fit gas-ion interaction profiles to the RIM functional form.
   - Compare gas-ion RDFs from classical MD against AIMD reference RDFs.
   - Final screening parameters used in the manuscript were `kappa = 1.2` for KCl and `kappa = 1.8` for LiCl.

2. **Run classical MD systems**
   - Equilibrate each LAMMPS system in NPT to determine the average density and final cell dimensions.
   - Run NVT production simulations using the equilibrated box size.
   - For umbrella sampling, use PLUMED restraints for the two-body, three-body, and four-body geometries.

3. **Reconstruct PMFs**
   - Use WHAM on each umbrella-sampling set.
   - Apply the appropriate geometric correction:
     - two-body: `W2(r) = F(r) + 2 kBT ln(r)`
     - three-body: `W3(l) = F(l) + kBT ln(l)`
     - four-body: `W4(s) = F(s)`
   - Shift each PMF so that the largest sampled separation is approximately zero.

4. **Compute diffusion coefficients**
   - Run eight independent 20 ns NVT trajectories for each gas/salt system.
   - Compute MSDs using time-origin averaging.
   - Fit the MSD over the 6--10 ns interval using the Einstein relation.

## Reproducing the analysis

Install the Python environment:

```bash
conda env create -f environment.yml
conda activate noble-gas-aggregation
```

Apply PMF corrections:

```bash
python analysis/pmf/apply_jacobian_corrections.py \
    --input results/pmfs/example_wham_output.dat \
    --geometry two_body \
    --temperature 1173 \
    --output results/pmfs/example_pmf_corrected.dat
```

Compute a diffusion coefficient from an MSD file:

```bash
python analysis/diffusion/compute_diffusion_from_msd.py \
    --input results/diffusion/example_msd.dat \
    --time-column time_ns \
    --msd-column msd_nm2 \
    --fit-start 6 \
    --fit-end 10
```

## Citation

If you use this repository, please cite the corresponding paper. Update this section with the final DOI after publication.
