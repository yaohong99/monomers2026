#### Coarse-Grained Monomers Simulation and Analysis Codes

Simulation codes for "Coarse-Grained modeling of hydrodynamic behavior
in DNA synthesis monomers".

__Installation__

To install use 'pip' and 'python' version 12.1 or greater 

```
cd ./src
pip install -r requirements.txt
```

To install use LAMMPS version 2024.8

## Installation

### Prerequisites

The following dependencies are required. Examples are provided for Arch Linux, but these can be adapted for other distributions (e.g., Ubuntu/CentOS).

* **CUDA Toolkit**: Required for the GPU-accelerated version.
    ```bash
    sudo pacman -S cuda
    ```
* **LAMMPS**: Must be built from source with specific packages enabled.
    ```bash
    wget https://download.lammps.org/tars/lammps-stable.tar.gz
    tar -xzf lammps-stable.tar.gz
    cd lammps-22Jul2025/src/
    make yes-molecule yes-extra-dump
    make mpi -j4
    ```

### Building the Project

1.  Open the `Makefile` and update the `LAMMPS_PATH` variable to point to your LAMMPS installation directory.
2.  Run the build command:
    ```bash
    make
    ```

---

This will generate the executable selm_cuda.

## Usage

To run a simulation, copy the compiled executable to your test directory (e.g., `./cases/SELM/test`).

```bash
CUDA_VISIBLE_DEVICES=0 nohup selm_cuda > /dev/null 2>&1 &
```

### Results of the paper 

Results can be run by using python parameter files found in the 
./cases folder for each model.  This is typically of the form
```
cd ../src/Boltzmann Inversion
python U.py -p ../cases/Boltzmann Inversion/B_S_values.txt
```

```
cd ../src/compare_168base
python compare_MSD.py -p ../cases/SELM/compare_168base/dirichlet_168base_X.txt
```

```
cd ../src/compare_dirichlet_2MSD
python compare_MSD.py -p ../cases/SELM/compare_dirichlet_2MSD/16base_X.txt
```

```
cd ../src/compare_period_2MSD
python compare_MSD.py -p ../cases/SELM/compare_period_2MSD/1base.txt
```

```
cd ../src/diffuison coefficient
python MSD.py -p ../cases/SELM/Diffuison Coefficient/1base.dcd
```

For more details, see the individual folders.
