#include "lammps.h"
#include "update.h"
#include "atom.h"
#include "domain.h"
#include "integrate.h"
#include "input.h"
#include "output.h"

#include "check.cuh"
#include "SELM.cuh"
#include "SELM_Eulerian.cuh"
#include "SELM_Lagrangian_LAMMPS.cuh"
#include "SELM_Kernel.cuh"
#include <cmath>
#include <mpi.h>


using namespace LAMMPS_NS;
using namespace SELM;

Lagrangian_LAMMPS::Lagrangian_LAMMPS(){

	const char *argv[]={"liblammps","-sc","none"};
	
    lmp = new LAMMPS(3, (char **)argv, MPI_COMM_WORLD);
	
	lmp->input->file("Model.LAMMPS_script");
}

Lagrangian_LAMMPS::~Lagrangian_LAMMPS(){
	
    delete lmp;
}


void Lagrangian_LAMMPS::initial(){

	
	int nsteps=selm->nsteps;
    lmp->update->nsteps = nsteps;
    lmp->update->firststep = lmp->update->ntimestep;
    lmp->update->laststep = lmp->update->ntimestep + nsteps;
    lmp->update->beginstep = lmp->update->firststep;
    lmp->update->endstep = lmp->update->laststep;
	
    lmp->update->whichflag = 1;
	
	lmp->init();
	
    lmp->update->integrate->setup(1);
	
	Atom* atom = lmp->atom;
    nlocal = atom->nlocal;
	
    x = *atom->x;
    v = *atom->v;
    f = *atom->f;

	/* double *x_bak=x;
	x=(double*)malloc(sizeof(double)*nlocal*3);
	memcpy(x,x_bak,sizeof(double)*nlocal*3);
	for(int i=0;i<nlocal;i++){
		atom->x[i]=x+i*3;
	}
	free(x_bak);
*/
	CHECK_RUNTIME(cudaMallocHost((void**)&x_pin,sizeof(double)*nlocal*3));
	CHECK_RUNTIME(cudaMallocHost((void**)&f_pin,sizeof(double)*nlocal*3));
	CHECK_RUNTIME(cudaMallocHost((void**)&v_pin,sizeof(double)*nlocal*3));
	lo = lmp->domain->boxlo;
	hi = lmp->domain->boxhi;
	
    m = new double[nlocal];
    for(int i=0; i<nlocal; i++){
		
        m[i] = atom->mass[atom->type[i]];
    }
	
	d_states=nullptr;
	
	CHECK_RUNTIME(cudaStreamCreate(&stream));
	
	if(selm->flagStochasticDriving){
		CHECK_RUNTIME(cudaMalloc((void**)&d_states,sizeof(curandState)*nlocal*3));
		int block_size=128;
		int grid_size=1+(nlocal*3-1)/block_size;
		initCurandStates<<<grid_size,block_size,0,stream>>>(d_states,selm->SELM_Seed,nlocal*3);
		CHECK_RUNTIME(cudaStreamSynchronize(stream));
	}
	
	size_t size = sizeof(double)*nlocal;
	CHECK_RUNTIME(cudaMalloc((void**)&d_x,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_v,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_f,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_m,size));
	
	CHECK_RUNTIME(cudaMemcpy(d_x,x,size*3,cudaMemcpyHostToDevice));
	CHECK_RUNTIME(cudaMemcpy(d_v,v,size*3,cudaMemcpyHostToDevice));
	CHECK_RUNTIME(cudaMemcpy(d_f,f,size*3,cudaMemcpyHostToDevice));
	CHECK_RUNTIME(cudaMemcpy(d_m,m,size,cudaMemcpyHostToDevice));

}

void Lagrangian_LAMMPS::final(){
	delete m;
	CHECK_RUNTIME(cudaFree(d_x));
	CHECK_RUNTIME(cudaFree(d_v));
	CHECK_RUNTIME(cudaFree(d_f));
	CHECK_RUNTIME(cudaFree(d_m));
	CHECK_RUNTIME(cudaFreeHost(x_pin));
	CHECK_RUNTIME(cudaFreeHost(v_pin));
	CHECK_RUNTIME(cudaFreeHost(f_pin));
	CHECK_RUNTIME(cudaStreamDestroy(stream));
	CHECK_RUNTIME(cudaFree(d_states));
}


void Lagrangian_LAMMPS::compute_f(){

	size_t size = sizeof(double)*nlocal*3;
	
	CHECK_RUNTIME(cudaMemcpyAsync(x,d_x,size,cudaMemcpyDeviceToHost,stream));
	CHECK_RUNTIME(cudaMemcpyAsync(v,d_v,size,cudaMemcpyDeviceToHost,stream));
	CHECK_RUNTIME(cudaMemcpyAsync(f,d_f,size,cudaMemcpyDeviceToHost,stream));
	CHECK_RUNTIME(cudaStreamSynchronize(stream));

	writedata();
    int ntimestep=++lmp->update->ntimestep;
    lmp->update->integrate->setup_minimal(1);
    
	if (ntimestep == lmp->output->next){
		lmp->output->write(ntimestep);
	}
	CHECK_RUNTIME(cudaMemcpyAsync(d_f,f,size,cudaMemcpyHostToDevice,stream));
}

void Lagrangian_LAMMPS::run(){
	dim3 blocksize(128,3);
	int gridsize=1+(nlocal-1)/128;
	double dx=selm->eulerian->dx;
	double upsilon=6.0*PI*2*dx*selm->mu;
	double coef=sqrt(2.0*selm->KB*selm->T*upsilon*selm->deltaT);
	double *u=selm->eulerian->d_u;
	double *u_f=selm->eulerian->d_f;
	int *dim=selm->eulerian->dim;
	lagrangian_update<<<gridsize,blocksize,0,stream>>>(d_x,d_v,d_f,d_m,u,u_f,d_states,dx,selm->deltaT,nlocal,dim[0],dim[1],dim[2],lo[0],hi[0],upsilon,coef,selm->flagStochasticDriving);
}

void Lagrangian_LAMMPS::writedata(){
	if(saveSkipSimulationData>0&&selm->timestep%saveSkipSimulationData==0){
		int i;
		double *af=*lmp->atom->f;
		double *ax=*lmp->atom->x;
		double *av=*lmp->atom->v;
		if(writeParticalF){
			FILE *file=fopen("particleF.dat","a");
			for(i=0;i<nlocal;i++){
				fprintf(file,"%g %g %g\n",af[3*i],af[3*i+1],af[3*i+2]);
			}
			fclose(file);
		}
		if(writeParticalV){
			FILE *file=fopen("particleV.dat","a");
			for(i=0;i<nlocal;i++){
				fprintf(file,"%g %g %g\n",av[3*i],av[3*i+1],av[3*i+2]);
			}
			fclose(file);
		}
		if(writeParticalX){
			FILE *file=fopen("particleX.dat","a");
			for(i=0;i<nlocal;i++){
				fprintf(file,"%g %g %g\n",ax[3*i],ax[3*i+1],ax[3*i+2]);
			}
			fclose(file);
		}
	}
}
