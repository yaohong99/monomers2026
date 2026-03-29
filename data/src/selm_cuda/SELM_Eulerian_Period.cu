#include "SELM.cuh"
#include "SELM_Eulerian.cuh"
#include "SELM_Lagrangian.cuh"
#include "SELM_Eulerian_Period.cuh"

#include "check.cuh"
#include "SELM_Kernel.cuh"
#include <cub/cub.cuh>

#include <cmath>

using namespace SELM;

void Eulerian_Period::initial(){
	dx=(selm->lagrangian->hi[0]-selm->lagrangian->lo[0])/dim[0];
	int N=dim[0]*dim[1]*dim[2];
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	CHECK_RUNTIME(cudaStreamCreate(&stream));
	cufftPlan3d(&plan_D2Z, dim[2], dim[1], dim[0], CUFFT_D2Z);
	cufftPlan3d(&plan_Z2D, dim[2],dim[1],dim[0], CUFFT_Z2D);
	cufftSetStream(plan_D2Z,stream);
	cufftSetStream(plan_Z2D,stream);
	size_t size=sizeof(double)*N*3;
	CHECK_RUNTIME(cudaMalloc((void**)&d_u,size));
	CHECK_RUNTIME(cudaMalloc((void**)&d_f,size));
	CHECK_RUNTIME(cudaMemset(d_u,0,size));	
	size = sizeof(double2)*N2;
	CHECK_RUNTIME(cudaMalloc((void**)&d_workarea,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_sum,sizeof(double)*3));
	d_cubarea = nullptr;
    cub_size = 0;
	CHECK_RUNTIME(cub::DeviceReduce::Sum(nullptr, cub_size, d_u, d_sum, N, 0));
	CHECK_RUNTIME(cudaMalloc((void**)&d_cubarea,cub_size));
	d_states=nullptr;
	if(selm->flagStochasticDriving){
		size=sizeof(curandState)*N2;
		CHECK_RUNTIME(cudaMalloc((void**)&d_states,size));
		initCurandStates<<<1+(N2-1)/128,128,0,stream>>>(d_states,selm->SELM_Seed,N2,1);
		CHECK_RUNTIME(cudaStreamSynchronize(stream));
	}
	int n[3]={dim[2],dim[1],dim[0]};
	int idist=N;
	int odist=N2;
	int inembed[3]={dim[2],dim[1],dim[0]};
	int onembed[3]={dim[2],dim[1],dim[0]/2+1};
	cufftPlanMany(&plan_Z2D_three,3,n,onembed,1,odist,inembed,1,idist,CUFFT_Z2D,3);
	cufftSetStream(plan_Z2D_three,stream);
	compute_f();
	CHECK_RUNTIME(cudaStreamSynchronize(stream));
}

void Eulerian_Period::final(){
	CHECK_RUNTIME(cudaFree(d_u));
	CHECK_RUNTIME(cudaFree(d_workarea));
	CHECK_RUNTIME(cudaFree(d_f));
	CHECK_RUNTIME(cudaFree(d_cubarea));
	CHECK_RUNTIME(cudaFree(d_sum));
	CHECK_RUNTIME(cudaFree(d_states));
	CHECK_RUNTIME(cudaStreamDestroy(stream));
	cufftDestroy(plan_D2Z);
	cufftDestroy(plan_Z2D);
	cufftDestroy(plan_Z2D_three);
}

Eulerian_Period::~Eulerian_Period(){}


void Eulerian_Period::run(){
	int N=dim[0]*dim[1]*dim[2];
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	int blocksize=128;
	int gridsize=1+(N-1)/blocksize;
	int gridsize2=1+(N2-1)/blocksize;
	u2uAsterisk<<<gridsize,blocksize,0,stream>>>(d_u,d_f,dx,selm->deltaT,selm->mu,selm->rho,dim[0],dim[1],dim[2]);
	div<<<gridsize,blocksize,0,stream>>>(d_u,dim[0],dim[1],dim[2],dx,d_workarea);
	cufftExecD2Z(plan_D2Z, d_workarea, (cufftDoubleComplex*)(d_workarea+N));
	d2z_data_transform<<<gridsize2,blocksize,0,stream>>>((double2*)(d_workarea+N),dim[0],dim[1],dim[2],dx);
	cufftExecZ2D(plan_Z2D, (cufftDoubleComplex*)(d_workarea+N), d_workarea);
	uAsterisk2u<<<gridsize,blocksize,0,stream>>>(d_workarea,dim[0],dim[1],dim[2],dx,d_u);
	for (int d = 0; d < 3; d++) {
		CHECK_RUNTIME(cub::DeviceReduce::Sum(d_cubarea, cub_size, d_u + d * N, d_sum + d, N, stream));
	}
	subtract_mean<<<gridsize,blocksize,0,stream>>>(d_u,d_sum,N);
}

void Eulerian_Period::compute_f(){
	int N=dim[0]*dim[1]*dim[2];
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	int blocksize=128;
	int gridsize=1+(N-1)/blocksize;
	int gridsize2=1+(N2-1)/blocksize;
	size_t size=sizeof(double)*N*3;
	double scale=sqrt(2*selm->KB*selm->T*selm->deltaT*selm->mu/N);
	if(selm->flagStochasticDriving){
		gthm_generate<<<gridsize2,blocksize,0,stream>>>((double2*)d_workarea,d_states,dim[0],dim[1],dim[2],dx);
		cufftExecZ2D(plan_Z2D_three, (cufftDoubleComplex*)d_workarea, d_f);
		gthm_scale<<<gridsize,blocksize,0,stream>>>(d_f,scale,N);
	}else{
		CHECK_RUNTIME(cudaMemsetAsync(d_f,0,size,stream));	
	}
}