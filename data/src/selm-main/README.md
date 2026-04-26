# SELM: Stochastic Euler-Lagrangian Method

A high-performance parallel implementation of the **Stochastic Euler-Lagrangian Method (SELM)**, featuring both MPI-based CPU parallelism and CUDA-accelerated GPU computing.

---

## Features

* **Hybrid Parallelism**: Optimized versions for distributed memory clusters (MPI) and NVIDIA GPUs (CUDA).
* **LAMMPS Integration**: Leverages LAMMPS for robust molecular dynamics and force field calculations.
* **Fast Fourier Transforms**: Utilizes FFTW for efficient reciprocal space computations.

---

## Installation

### Prerequisites

The following dependencies are required. Examples are provided for Arch Linux, but these can be adapted for other distributions (e.g., Ubuntu/CentOS).

* **OpenMPI**: High-performance message passing library.
    ```bash
    sudo pacman -S openmpi
    ```
* **CUDA Toolkit**: Required for the GPU-accelerated version.
    ```bash
    sudo pacman -S cuda
    ```
* **FFTW**: Fast Fourier Transform library (with MPI support).
    ```bash
    sudo pacman -S fftw fftw-openmpi
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

Two build systems are supported: **CMake** (recommended) and the legacy **Makefiles**.

---

#### Option A — CMake (Recommended)

CMake 3.18 or later is required. 
**1. Configure**

```bash
cmake -S src -B build -DLAMMPS_PATH=/path/to/lammps/src
```

Replace `/path/to/lammps/src` with the actual path to your LAMMPS `src` directory
(the folder that contains `lammps.h`). This value is required; omitting it will
produce a clear error at configure time.

Additional CMake options:

| Option | Default | Description |
|---|---|---|
| `LAMMPS_PATH` | *(required)* | Path to the LAMMPS `src` directory |
| `BUILD_MPI` | `ON` | Build the MPI version (`selm_mpi`) |
| `BUILD_CUDA` | `ON` | Build the CUDA version (`selm_cuda`) |
| `CMAKE_CUDA_ARCHITECTURES` | auto-detected | Override the GPU arch (e.g. `86` for Ampere) |

Examples:

```bash
# Build both targets (default)
cmake -S src -B build -DLAMMPS_PATH=$HOME/lammps-22Jul2025/src

# Build only the MPI version
cmake -S src -B build -DLAMMPS_PATH=$HOME/lammps-22Jul2025/src -DBUILD_CUDA=OFF

# Build only the CUDA version
cmake -S src -B build -DLAMMPS_PATH=$HOME/lammps-22Jul2025/src -DBUILD_MPI=OFF

# Override GPU architecture manually (e.g. for cross-compilation)
cmake -S src -B build -DLAMMPS_PATH=$HOME/lammps-22Jul2025/src \
      -DCMAKE_CUDA_ARCHITECTURES=86
```

**2. Compile**

```bash
cmake --build build -j$(nproc)
```

The executables `selm_mpi` and/or `selm_cuda` are placed in the `build/` directory
(under `build/mpi/` and `build/cuda/` respectively).

**3. Clean**

```bash
rm -rf build/
```

---

#### Option B — Legacy Makefiles

1. Navigate to the desired implementation directory: `src/mpi` for CPU or `src/cuda` for GPU.
2. Open the `Makefile` and update the `LAMMPS_PATH` variable to point to your LAMMPS `src` directory.
3. Run the build command:
    ```bash
    make
    ```

This will generate the executable `selm_mpi` or `selm_cuda` in the same directory.

---

## Usage

To run a simulation, copy the compiled executable to your test directory (e.g., `test/OneParticle`).

### Running the CUDA Version
```bash
./selm_cuda
```

### Running the MPI Version
Use mpirun to specify the number of processes (e.g., 4):
```bash
mpirun -np 4 ./selm_mpi
```

## Project Structure

- `src/mpi`: Source code for the MPI parallel implementation.
- `src/cuda`: Source code for the CUDA GPU implementation.
- `test/`: Contains example cases.
