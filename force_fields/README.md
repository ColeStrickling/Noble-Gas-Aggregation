# Force-field parameters

This directory contains the force-field parameters used for the classical molecular dynamics simulations in the manuscript.

The simulations used a rigid ion model (RIM) for molten LiCl and KCl. Salt--salt and salt--gas interactions were described using the following functional form:

```math
U_{\mathrm{RIM}} = \sum_{i \lt j} \left[ \frac{q_i q_j}{4 \pi \epsilon_0 r_{ij}} + A_{ij}\exp\left(\frac{\sigma_{ij}-r_{ij}}{\rho_{ij}}\right) - \frac{C_{ij}}{r_{ij}^{6}} - \frac{D_{ij}}{r_{ij}^{8}} \right]
```

where `r_ij` is the distance between atoms or ions `i` and `j`, `q_i` and `q_j` are the charges, `A_ij`, `sigma_ij`, and `rho_ij` define the short-range repulsive interaction, and `C_ij` and `D_ij` define the dispersion terms.

Gas--gas interactions were described using a Lennard-Jones potential:

```math
U_{\mathrm{LJ}}(r_{ij}) =
4\epsilon_{ij}
\left[
\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
-
\left(\frac{\sigma_{ij}}{r_{ij}}\right)^6
\right]
```

where `sigma_ij` is the Lennard-Jones size parameter and `epsilon_ij` is the well depth.

## Files

* `rim_parameters_manuscript_units.csv` contains the RIM parameters reported in the manuscript for LiCl, KCl, and the fitted salt--gas interactions.
* `gas_lj_parameters_manuscript_units.csv` contains the Lennard-Jones parameters used for Ar--Ar and Xe--Xe interactions.
* `lammps_include/` contains example LAMMPS include files or converted parameter files used to run the classical MD simulations.

## Important notes

* Values in `rim_parameters_manuscript_units.csv` are stored in the units reported in the manuscript.
* Energy units should be converted as needed for the selected LAMMPS `units` setting.
* The CSV files are intended to provide a clear record of the manuscript parameters. The files in `lammps_include/` should be used as the direct input examples for reproducing the LAMMPS simulations.
* The fitted salt--gas parameters are salt-specific. The LiCl parameters should be used with the LiCl RIM model, and the KCl parameters should be used with the KCl RIM model.
