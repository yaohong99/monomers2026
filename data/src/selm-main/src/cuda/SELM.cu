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
		// printf("\nstep: %d of %d\n", timestep, nsteps);
        // if(timestep == nsteps / 2) total_time=-MPI_Wtime();
		// printf("\rstep: %d of %d", timestep, nsteps);
		// 必须的，一般默认输出\n才会刷新缓冲区
		// fflush(stdout); 
		// 拉格朗日点的更新
        lagrangian->run();
		// 必须确保拉格朗日的更新完成才能进行下一步
		CHECK_RUNTIME(cudaStreamSynchronize(lagrangian->stream));
		// 欧拉的更新和欧拉和拉格朗日的力的计算
        eulerian->run();
        compute_f();
		// 确保当前迭代已完成
		CHECK_RUNTIME(cudaStreamSynchronize(eulerian->stream));
		CHECK_RUNTIME(cudaStreamSynchronize(lagrangian->stream));
    }
	MPI_Barrier(MPI_COMM_WORLD);
	total_time=+MPI_Wtime();
	printf("\n");
	printf("dt: %f\n", deltaT);
	printf("nGpu: 1\n");
	printf("Total time: %fs\n", total_time);
	printf("Total time average: %fs\n", total_time/nsteps);
}
