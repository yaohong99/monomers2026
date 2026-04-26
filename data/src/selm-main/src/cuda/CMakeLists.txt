find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Collect all .cu files, excluding callback.cu (matches original Makefile filter).
file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*callback\\.cu$")

add_executable(selm_cuda ${CUDA_SOURCES})

# -rdc=true: enable relocatable device code (required for device-code linking
# across multiple translation units).
set_property(TARGET selm_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(selm_cuda PRIVATE cxx_std_17)

target_compile_options(selm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_include_directories(selm_cuda PRIVATE
    ${LAMMPS_PATH}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_directories(selm_cuda PRIVATE ${LAMMPS_PATH})

target_link_libraries(selm_cuda PRIVATE
    lammps_mpi
    MPI::MPI_CXX
    CUDA::cufft
    CUDA::curand
)
