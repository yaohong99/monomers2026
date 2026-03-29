#include "SELM.cuh"
#include "SELM_ParmParse.cuh"
#include "SELM_Eulerian_Period.cuh"
#include "SELM_Lagrangian_LAMMPS.cuh"
#include "check.cuh"

#include "lammps.h"
#include "update.h"

#include <string>
#include <mpi.h>

using namespace SELM;
using namespace std;
using namespace LAMMPS_NS;

Selm::Selm(){
    ParmParse pp;
    string LagrangianType,EulerianType;
    pp.load("Model.SELM");
    pp.get("rho",rho);
    pp.get("T",T);
    pp.get("mu",mu);
    pp.get("nsteps",nsteps);
    pp.get("KB",KB);
    pp.get("LagrangianType",LagrangianType);
    pp.get("EulerianType",EulerianType);
    pp.query("flagStochasticDriving",flagStochasticDriving);
	pp.query("SELM_Seed",SELM_Seed);
    if(LagrangianType=="Lagrangian_LAMMPS"){
        lagrangian = new Lagrangian_LAMMPS();
        lagrangian->selm = this;
		pp.query("saveSkipSimulationData",lagrangian->saveSkipSimulationData);
		pp.query("writeParticalV",lagrangian->writeParticalV);
		pp.query("writeParticalF",lagrangian->writeParticalF);
		pp.query("writeParticalX",lagrangian->writeParticalX);
        deltaT = lagrangian->lmp->update->dt; 
    }else{
        lagrangian = nullptr;
    }
    if(EulerianType=="Eulerian_Period"){
        eulerian = new Eulerian_Period();
        eulerian->selm = this;
        pp.get("numMeshPtsPerDir", eulerian->dim);
    }else{
        eulerian = nullptr;
    }
}

Selm::~Selm(){
    delete eulerian;
    delete lagrangian;
}

void Selm::initial(){
    lagrangian->initial();
    eulerian->initial();
}

void Selm::final(){
    lagrangian->final();
    eulerian->final();
}

void Selm::compute_f(){
    eulerian->compute_f();
    lagrangian->compute_f();
}

void Selm::run(){
	MPI_Barrier(MPI_COMM_WORLD);
	total_time=-MPI_Wtime();
    for(int i=0; i<nsteps; i++){
        timestep++;
		printf("\rstep: %d of %d", timestep, nsteps);
		
		fflush(stdout); 
		
        lagrangian->run();
		
		CHECK_RUNTIME(cudaStreamSynchronize(lagrangian->stream));
		
        eulerian->run();
        compute_f();
		
		CHECK_RUNTIME(cudaStreamSynchronize(eulerian->stream));
		CHECK_RUNTIME(cudaStreamSynchronize(lagrangian->stream));
    }
	MPI_Barrier(MPI_COMM_WORLD);
	total_time=+MPI_Wtime();
	printf("\n");
	printf("nGpu: 1\n");
	printf("Total time: %.3fs\n", total_time);
	printf("Total time average: %.3fs\n", total_time/nsteps);
}
