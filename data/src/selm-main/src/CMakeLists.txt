cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# GPU architecture auto-detection
# Must be set BEFORE project() so CMake's CUDA compiler probing picks it up.
# ---------------------------------------------------------------------------
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    # CMake 3.24+ can query the attached GPU directly at configure time.
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    # Fallback: ask nvidia-smi for the compute capability (e.g. "7.5") and
    # strip the dot to get the arch number CMake expects (e.g. "75").
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _gpu_cc
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_gpu_cc STREQUAL "")
        message(WARNING
            "nvidia-smi did not return a compute capability. "
            "Defaulting to sm_75. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>.")
        set(CMAKE_CUDA_ARCHITECTURES 75)
    else()
        # Handle multi-GPU output: take only the first line.
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _gpu_cc_first "${_gpu_cc}")
        string(REPLACE "." "" _gpu_arch "${_gpu_cc_first}")
        set(CMAKE_CUDA_ARCHITECTURES "${_gpu_arch}")
    endif()
endif()

project(SELM LANGUAGES CXX CUDA)

message(STATUS "SELM: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------------------------
# LAMMPS path — must be supplied by the user
# ---------------------------------------------------------------------------
set(LAMMPS_PATH "" CACHE PATH "Path to the LAMMPS source directory (the 'src' folder)")

if(NOT LAMMPS_PATH)
    message(FATAL_ERROR
        "LAMMPS_PATH is not set.\n"
        "Please configure with:  cmake -DLAMMPS_PATH=/path/to/lammps/src ...")
endif()

if(NOT EXISTS "${LAMMPS_PATH}/lammps.h")
    message(FATAL_ERROR
        "LAMMPS_PATH='${LAMMPS_PATH}' does not appear to be a valid LAMMPS src directory "
        "(lammps.h not found).")
endif()

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(BUILD_MPI  "Build the MPI version  (selm_mpi)"  ON)
option(BUILD_CUDA "Build the CUDA version (selm_cuda)" ON)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(BUILD_MPI)
    add_subdirectory(mpi)
endif()

if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()
