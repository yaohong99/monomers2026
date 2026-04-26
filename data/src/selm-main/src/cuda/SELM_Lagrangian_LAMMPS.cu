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
	// 创建lammps对象
    lmp = new LAMMPS(3, (char **)argv, MPI_COMM_WORLD);
	// 必须以这种方式读取文件，直接使用命令行参数时单进程运行会有问题（MPI_Sendrecv）
	lmp->input->file("Model.LAMMPS_script");
}

Lagrangian_LAMMPS::~Lagrangian_LAMMPS(){
	// 释放内存
    // MPI_Comm_free(&lagcomm);
    delete lmp;
}


void Lagrangian_LAMMPS::initial(){

	// updata时间步的设置，使用lammps输入时必需的设置
	int nsteps=selm->nsteps;
    lmp->update->nsteps = nsteps;
    lmp->update->firststep = lmp->update->ntimestep;
    lmp->update->laststep = lmp->update->ntimestep + nsteps;
    lmp->update->beginstep = lmp->update->firststep;
    lmp->update->endstep = lmp->update->laststep;
	// 必须先设置为1，不然update.init不会被调用
    lmp->update->whichflag = 1;
	// 初始化
	lmp->init();
	// 以及atom.setup以及构建邻居节点和力的计算
    lmp->update->integrate->setup(1);
	// 粒子的个数
	Atom* atom = lmp->atom;
    nlocal = atom->nlocal;
	// 注意atom->x/v/f为[nlocal][3]类型
    x = *atom->x;
    v = *atom->v;
    f = *atom->f;


	// 盒子的尺寸，直接使用ammps中的指针
	lo = lmp->domain->boxlo;
	hi = lmp->domain->boxhi;
	// 原子的质量
    m = new double[nlocal];
    for(int i=0; i<nlocal; i++){
		// 注意atom->type一般只有几种，必须通过这种方式获取mass
        m[i] = atom->mass[atom->type[i]];
    }
	// 初始化d_states为空指针
	d_states=nullptr;
	// 创建流
	CHECK_RUNTIME(cudaStreamCreate(&stream));
	// 初始化各个点的随机数state
	if(selm->flagStochasticDriving){
		CHECK_RUNTIME(cudaMalloc((void**)&d_states,sizeof(curandState)*nlocal*3));
		int block_size=128;
		int grid_size=1+(nlocal*3-1)/block_size;
		initCurandStates<<<grid_size,block_size,0,stream>>>(d_states,selm->SELM_Seed,nlocal*3);
		CHECK_RUNTIME(cudaStreamSynchronize(stream));
	}
	// 分配内存
	size_t size = sizeof(double)*nlocal;
	CHECK_RUNTIME(cudaMalloc((void**)&d_x,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_v,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_f,size*3));
	CHECK_RUNTIME(cudaMalloc((void**)&d_m,size));
	// 初始化
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
	CHECK_RUNTIME(cudaStreamDestroy(stream));
	CHECK_RUNTIME(cudaFree(d_states));
}

// 需要先确保GPU中x v f已经同步
void Lagrangian_LAMMPS::compute_f(){
	// 传回CPU中利用LAMMPS计算f再传回GPU中
	// 需要传递x v f,因为lammps对f的计算需要用到x v f
	size_t size = sizeof(double)*nlocal*3;
	// 异步传输，d_x d_f d_v
	CHECK_RUNTIME(cudaMemcpyAsync(x,d_x,size,cudaMemcpyDeviceToHost,stream));
	CHECK_RUNTIME(cudaMemcpyAsync(v,d_v,size,cudaMemcpyDeviceToHost,stream));
	CHECK_RUNTIME(cudaMemcpyAsync(f,d_f,size,cudaMemcpyDeviceToHost,stream));

	// 确保传输完成
	CHECK_RUNTIME(cudaStreamSynchronize(stream));
	// 写入数据
	writedata();
	// 构建邻居节点，完成f的更新
    int ntimestep=++lmp->update->ntimestep;
    lmp->update->integrate->setup_minimal(1);
    // lmp->update->integrate->ev_set(ntimestep);
	// lammps的输出
	if (ntimestep == lmp->output->next){
		lmp->output->write(ntimestep);
	}
	// 只需要传回f，lammps只用于完成f的更新
	CHECK_RUNTIME(cudaMemcpyAsync(d_f,f,size,cudaMemcpyHostToDevice,stream));
}

void Lagrangian_LAMMPS::run(){
	// dim3 blocksize(128,3);
	// int gridsize=1+(nlocal-1)/128;
	int blocksize = 384;
	int gridsize=1+(3*nlocal-1)/blocksize;
	double dx=selm->eulerian->dx;
	double upsilon=6.0*PI*2*dx*selm->mu;
	double coef=sqrt(2.0*selm->KB*selm->T*upsilon*selm->deltaT);
	double *u=selm->eulerian->d_u[selm->eulerian->cur_u];
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
