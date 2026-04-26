#include "SELM.h"
#include <mpi.h>
#include <stdio.h>
using namespace SELM;
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    Selm selm;
	selm.initial();
    selm.run();
	selm.final();
    MPI_Finalize();
}
