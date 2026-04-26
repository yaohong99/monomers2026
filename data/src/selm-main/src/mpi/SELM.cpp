#include "SELM.h"
#include "SELM_ParmParse.h"
#include "SELM_Eulerian_Period.h"
#include "SELM_Lagrangian_LAMMPS.h"

#include "lammps.h"
#include "update.h"

#include <cstdio>
#include <fftw3-mpi.h>
#include <string>

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
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
}

Selm::~Selm(){
    delete eulerian;
    delete lagrangian;
}

void Selm::initial(){
    fftw_mpi_init();
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
		// if(!rank) printf("\nstep: %d of %d\n", timestep, nsteps);
		// fflush(stdout);
        lagrangian->run();
        eulerian->run();
        compute_f();
    }
	// if(!rank) printf("\n");
	MPI_Barrier(MPI_COMM_WORLD);
	total_time=+MPI_Wtime();
	if(!rank) {
		printf("process             %d\n",size);
		printf("steps               %d\n",nsteps);
		printf("Elapedtime/s        %.2f\n", total_time);
		printf("TimePerStep/s       %.2f\n\n", total_time/nsteps);
	}
}
