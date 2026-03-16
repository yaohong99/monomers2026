# Molecular Dynamics Simulation – GROMACS Workflow

This repository contains the forcefield parameter files and topology files required to reproduce the all-atom (AA) molecular dynamics simulations described in the manuscript. The simulations were performed using **GROMACS**.

---

## Repository Contents

| File | Description |
|------|-------------|
| `AN.gro` | Acetonitrile molecule coordinate file |
| `AN.top` / `AN.itp` | Acetonitrile topology and forcefield parameters |
| `C2.gro` | Nucleobase monomer coordinate file |
| `topol.top` | System topology file (monomer + solvent) |
| `em.mdp` | Energy minimization parameters |
| `eq.mdp` | NPT equilibration parameters |
| `run.mdp` | Production run parameters |

---

## Prerequisites

- [GROMACS](https://www.gromacs.org/) ≥ 2020
- Basic familiarity with GROMACS command-line tools

---

## Simulation Workflow

### Step 1 – Prepare the Solvent Box

Build a pure acetonitrile solvent box by inserting solvent molecules into a periodic simulation cell, then run energy minimization followed by NPT equilibration to obtain a density-equilibrated solvent configuration.

### Step 2 – Insert the Monomer

Place the nucleobase monomer (`C.gro`) into the equilibrated acetonitrile box using random orientation, then solvate the system to update the topology accordingly.

### Step 3 – Energy Minimization

Minimize the potential energy of the combined system (monomer + solvent) using the steepest descent algorithm to remove any steric clashes introduced during insertion.

### Step 4 – NPT Equilibration

Run a short NPT ensemble simulation to equilibrate the system temperature and pressure before the production run. The `eq.mdp` file defines the thermostat and barostat settings used in the manuscript.

### Step 5 – Production Run

Perform the production MD simulation using the settings in `run.mdp`. Trajectory output frequency and total simulation length are defined therein. Adjust the number of MPI ranks and OpenMP threads (`-ntmpi`, `-ntomp`) according to your hardware.

---

## Notes

- All simulations were conducted in acetonitrile solvent at 298 K and 1 bar.
- The GAFF forcefield was used for the nucleobase monomer.
- The CG simulation workflow is analogous to the AA procedure described above and will not be repeated further, please refer to the CG directory.

---

## Citation

If you use these files, please cite the original manuscript (DOI to be added upon acceptance).
