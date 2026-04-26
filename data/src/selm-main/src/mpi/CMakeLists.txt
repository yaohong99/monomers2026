find_package(MPI REQUIRED)

file(GLOB MPI_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(selm_mpi ${MPI_SOURCES})

target_compile_features(selm_mpi PRIVATE cxx_std_17)

target_compile_options(selm_mpi PRIVATE -O3 -g)

target_include_directories(selm_mpi PRIVATE
    ${LAMMPS_PATH}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_directories(selm_mpi PRIVATE ${LAMMPS_PATH})

target_link_libraries(selm_mpi PRIVATE
    MPI::MPI_CXX
    lammps_mpi
    fftw3_mpi
    fftw3
)
