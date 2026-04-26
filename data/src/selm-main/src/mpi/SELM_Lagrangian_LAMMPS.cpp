#include "lammps.h"
#include "update.h"
#include "atom.h"
#include "domain.h"
#include "integrate.h"
#include "input.h"
#include "output.h"
#include "random_mars.h"

#include "SELM.h"
#include "SELM_Eulerian.h"
#include "SELM_Lagrangian_LAMMPS.h"

#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

using namespace LAMMPS_NS;
using namespace SELM;

Lagrangian_LAMMPS::Lagrangian_LAMMPS(){

	// 进程的rank
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	// 为每个进程创建一个通信域
	MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &lagcomm);
	const char *argv[]={"liblammps","-sc","none"};
	// 创建lammps对象
    lmp = new LAMMPS(3, (char **)argv, lagcomm);
	// 必须以这种方式读取文件，直接使用命令行参数时单进程运行会有问题（MPI_Sendrecv）
	lmp->input->file("Model.LAMMPS_script");
}

Lagrangian_LAMMPS::~Lagrangian_LAMMPS(){
	// 释放内存
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
	// atom.setup以及构建邻居节点和力的计算
    lmp->update->integrate->setup(1);
	// 粒子的个数
	Atom* atom = lmp->atom;
    nlocal = atom->nlocal;
	// 注意atom->x/v/f为[nlocal][3]类型
    x = *atom->x;
    v = *atom->v;
    f = *atom->f;
	// 盒子的尺寸
	lo = lmp->domain->boxlo;
	hi = lmp->domain->boxhi;
	// 储存gamma(u)
    f_tmp = new double[nlocal*3];
	// 储存x的网格坐标
	x_grid=new double[nlocal*3];
	// 原子的质量
    m = new double[nlocal];
    for(int i=0; i<nlocal; i++){
		// 注意atom->type一般只有几种，必须通过这种方式获取mass
        m[i] = atom->mass[atom->type[i]];
    }
	// NOTE:不同于标准库和curand，只要随机数种子是一样的，生成的正态分布序列就是一样的
    rand = new RanMars(lmp, selm->SELM_Seed);
}

void Lagrangian_LAMMPS::compute_f(){
	// 写入数据
	if(!rank) writedata();
	// 构建邻居节点，完成f的更新
    ++lmp->update->ntimestep;
    lmp->update->integrate->setup_minimal(1);
	if (!rank && lmp->update->ntimestep == lmp->output->next){
		lmp->output->write(lmp->update->ntimestep);
	}
}

void Lagrangian_LAMMPS::run(){
	double dx=selm->eulerian->dx;
	double upsilon=6.0*M_PI*2*dx*selm->mu;
	double coef=sqrt(2.0*selm->KB*selm->T*upsilon*selm->deltaT);
	double *u=selm->eulerian->u[selm->eulerian->cur_u];
	double *u_f=selm->eulerian->f;
	int *dim=selm->eulerian->dim;
	int local_z_start=selm->eulerian->local_z_start;
	int local_z=selm->eulerian->local_z;
	int local_z_end=local_z+local_z_start;

	compute_gamma(u,dx,dim[0],dim[1],dim[2],local_z_start,local_z_end);
	MPI_Allreduce(MPI_IN_PLACE,f_tmp,nlocal*3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	lagrangian_update(u_f,dx,dim[0],dim[1],dim[2],local_z_start,local_z_end,upsilon,coef,selm->flagStochasticDriving);
}

void Lagrangian_LAMMPS::final(){
	delete f_tmp;
	delete m;
	delete rand;
	delete x_grid;
	// 释放分配的通信域
    MPI_Comm_free(&lagcomm);
}

void Lagrangian_LAMMPS::lagrangian_update(double *u_f,double dx,int Nx,int Ny,int Nz,int local_z_start,int local_z_end,double upsilon,double coef,bool rand_flag){

	int N_lag=nlocal*3;
	int i,d;
	double dt=lmp->update->dt;
	// 总Eulerian点数量
	int N = Nx*Ny*(local_z_end-local_z_start);
	// 每次迭代开始位置的索引
	int ind;
	// 将-upsilon(v-gamma(u))*dt写入f_tmp中
	for(i=0;i<N_lag;i++){
		f_tmp[i]=-upsilon*(v[i]-f_tmp[i])*dt;
		// f_tmp+=Fthm
		if(rand_flag) f_tmp[i]+=rand->gaussian() * coef;
	}
	// u_f-=lambda(f_tmp),u_f必须先初始化
	for(i=0;i<nlocal;i++){
		ind=3*i;
		lambda(x_grid[ind],x_grid[ind+1]-0.5,x_grid[ind+2]-0.5,Nx,Ny,Nz,f_tmp[ind],u_f,local_z_start,local_z_end);
		lambda(x_grid[ind]-0.5,x_grid[ind+1],x_grid[ind+2]-0.5,Nx,Ny,Nz,f_tmp[ind+1],u_f+N,local_z_start,local_z_end);
		lambda(x_grid[ind]-0.5,x_grid[ind+1]-0.5,x_grid[ind+2],Nx,Ny,Nz,f_tmp[ind+2],u_f+2*N,local_z_start,local_z_end);
	}

	double tmp_x;
	double L=hi[0]-lo[0];
	// 更新f v x
	for(i=0;i<N_lag;i++){
		// 更新f
		// f_tmp[i]+=100;
		f_tmp[i]+=dt*f[i];
		f[i]=f_tmp[i]/dt;
		// 更新x
		tmp_x=v[i]*dt;
		// 某个粒子的位移超过盒子的一半代表模拟中出现了错误，可能是时间间隔太大
		if(fabs(tmp_x)>L/2) {
			printf("Error:  The displacement of a particle exceeds half the size of the box\n");
		}
		tmp_x+=x[i];
		if(tmp_x>hi[0]) tmp_x-=L;
		else if(tmp_x<lo[0]) tmp_x+=L;
		x[i]=tmp_x;
		// 更新v
		v[i]+=f_tmp[i]/m[i/3];
	}
}

void Lagrangian_LAMMPS::compute_gamma(double *u,double dx,int Nx,int Ny,int Nz,int local_z_start,int local_z_end){

	int N_lag=nlocal*3;
	int i,d;
	// 初始化x_grid
	for(i=0;i<N_lag;i++){
		x_grid[i]=(x[i]-lo[0])/dx;
	}
	// 总Eulerian点数量
	int N = Nx*Ny*(local_z_end-local_z_start);
	// 每次迭代开始位置的索引
	int ind;
	for(i=0;i<nlocal;i++){
		ind=3*i;
		// f_tmp=gamma(u)
		f_tmp[ind]=gamma(x_grid[ind],x_grid[ind+1]-0.5,x_grid[ind+2]-0.5,Nx,Ny,Nz,u,local_z_start,local_z_end);
		f_tmp[ind+1]=gamma(x_grid[ind]-0.5,x_grid[ind+1],x_grid[ind+2]-0.5,Nx,Ny,Nz,u+N,local_z_start,local_z_end);
		f_tmp[ind+2]=gamma(x_grid[ind]-0.5,x_grid[ind+1]-0.5,x_grid[ind+2],Nx,Ny,Nz,u+2*N,local_z_start,local_z_end);
	}
}

double Lagrangian_LAMMPS::gamma(double rx,double ry,double rz,int Nx, int Ny,int Nz, double *u,int local_z_start, int local_z_end){
	int p[3];
	int p0=floor(rx)-1;
	int p1=floor(ry)-1;
	int p2=floor(rz)-1;
	int i,j,k;
	int n;
	double r[3];
	double delta_k, delta_j, delta_i;
	double rv=0.0;

	for(k=0;k<4;k++){
		delta_k=1.0;
		p[2]=p2+k;
		r[2]=fabs(rz-p[2]);
		p[2]=(Nz+p[2])%Nz;
		if(p[2]<local_z_start || p[2]>=local_z_end) continue;
		if(r[2]<1.0){
			delta_k*=(3.0-2.0*r[2]+sqrt(1.0+4.0*r[2]*(1.0-r[2])))/8.0;
		}else if(r[2]<2.0){
			delta_k*=(5.0-2.0*r[2]-sqrt(-7.0+4.0*r[2]*(3.0-r[2])))/8.0;
		}
		for(j=0;j<4;j++){
			delta_j=delta_k;
			p[1]=p1+j;
			r[1]=fabs(ry-p[1]);
			p[1]=(Ny+p[1])%Ny;
			if(r[1]<1.0){
				delta_j*=(3.0-2.0*r[1]+sqrt(1.0+4.0*r[1]*(1.0-r[1])))/8.0;
			}else if(r[1]<2.0){
				delta_j*=(5.0-2.0*r[1]-sqrt(-7.0+4.0*r[1]*(3.0-r[1])))/8.0;
			}
			for(i=0;i<4;i++){
				delta_i=delta_j;
				p[0]=p0+i;
				r[0]=fabs(rx-p[0]);
				p[0]=(Nx+p[0])%Nx;
				if(r[0]<1.0){
					delta_i*=(3.0-2.0*r[0]+sqrt(1.0+4.0*r[0]*(1.0-r[0])))/8.0;
				}else if(r[0]<2.0){
					delta_i*=(5.0-2.0*r[0]-sqrt(-7.0+4.0*r[0]*(3.0-r[0])))/8.0;
				}
				n=((p[2]-local_z_start)*Ny+p[1])*Nx+p[0];
				rv+=u[n]*delta_i;
			}
		}
	}
	return rv;
}

void Lagrangian_LAMMPS::lambda(double rx,double ry,double rz,int Nx, int Ny,int Nz, double f, double *u_f, int local_z_start, int local_z_end){
	int p[3];
	int p0=floor(rx)-1;
	int p1=floor(ry)-1;
	int p2=floor(rz)-1;
	int i,j,k;
	int n;
	double r[3];
	double delta_k, delta_j, delta_i;

	for(k=0;k<4;k++){
		delta_k=1.0;
		p[2]=p2+k;
		r[2]=fabs(rz-p[2]);
		p[2]=(Nz+p[2])%Nz;
		if(p[2]<local_z_start || p[2]>=local_z_end) continue;
		if(r[2]<1.0){
			delta_k*=(3.0-2.0*r[2]+sqrt(1.0+4.0*r[2]*(1.0-r[2])))/8.0;
		}else if(r[2]<2.0){
			delta_k*=(5.0-2.0*r[2]-sqrt(-7.0+4.0*r[2]*(3.0-r[2])))/8.0;
		}
		for(j=0;j<4;j++){
			delta_j=delta_k;
			p[1]=p1+j;
			r[1]=fabs(ry-p[1]);
			p[1]=(Ny+p[1])%Ny;
			if(r[1]<1.0){
				delta_j*=(3.0-2.0*r[1]+sqrt(1.0+4.0*r[1]*(1.0-r[1])))/8.0;
			}else if(r[1]<2.0){
				delta_j*=(5.0-2.0*r[1]-sqrt(-7.0+4.0*r[1]*(3.0-r[1])))/8.0;
			}
			for(i=0;i<4;i++){
				delta_i=delta_j;
				p[0]=p0+i;
				r[0]=fabs(rx-p[0]);
				p[0]=(Nx+p[0])%Nx;
				if(r[0]<1.0){
					delta_i*=(3.0-2.0*r[0]+sqrt(1.0+4.0*r[0]*(1.0-r[0])))/8.0;
				}else if(r[0]<2.0){
					delta_i*=(5.0-2.0*r[0]-sqrt(-7.0+4.0*r[0]*(3.0-r[0])))/8.0;
				}
				n=((p[2]-local_z_start)*Ny+p[1])*Nx+p[0];
				u_f[n]-=f*delta_i;
			}
		}
	}
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
