#include "SELM.h"
#include "SELM_Eulerian_Period.h"
#include "SELM_Eulerian.h"
#include "SELM_Lagrangian.h"

#include "lammps.h"
#include "domain.h"
#include "random_mars.h"

#include <cstring>
#include <fftw3-mpi.h>
#include <fftw3.h>
#include <math.h>

using namespace LAMMPS_NS;
using namespace SELM;

void Eulerian_Period::initial(){
    cur_u = 0;
	// 初始化rank 
	rank=selm->rank;
	// 初始化z y的开始和结束索引以及z y分块的大小
	ptrdiff_t tmp[4];
	ptrdiff_t dim_half[3]={dim[2],dim[1],dim[0]/2+1};
	ptrdiff_t dim_real[3]={dim[2],dim[1],dim[0]};
    // 获取本地数据尺寸
	ptrdiff_t alloc_local = fftw_mpi_local_size_many_transposed(3,dim_half,3,FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, tmp, tmp+1, tmp+2, tmp+3);
	local_z=tmp[0];
	local_z_start=tmp[1];
	local_y=tmp[2];
	local_y_start=tmp[3];
	workarea=fftw_alloc_real(alloc_local*2);
	// 创建fftw plan
    plan_c2r_three = fftw_mpi_plan_many_dft_c2r(3,dim_real,3,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, (fftw_complex*)workarea, workarea, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
	plan_r2c = fftw_mpi_plan_dft_r2c_3d(dim[2],dim[1],dim[0],workarea,(fftw_complex*)workarea,MPI_COMM_WORLD,FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
	plan_c2r = fftw_mpi_plan_dft_c2r_3d(dim[2],dim[1],dim[0],(fftw_complex*)workarea,workarea,MPI_COMM_WORLD,FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
	// 初始化dx，假定各维的dx,hi,lo,Nx Ny Nz都是一样的
	dx=(selm->lagrangian->hi[0]-selm->lagrangian->lo[0])/dim[0];
	// 总的元素数量，延z维划分
	int N=dim[0]*dim[1]*local_z;
	u[0]=new double[N*3]();
	u[1]=new double[N*3]();
	f=new double[N*3];
    rand = new RanMars(selm->lagrangian->lmp, selm->SELM_Seed+rank+1);
	int Nyx=dim[1]*dim[0];
	head = new double[Nyx*3];
	tail = new double[Nyx*3];
	sum = new double[3];

	compute_f();
}

void Eulerian_Period::final(){
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_destroy_plan(plan_c2r_three);
	fftw_free(workarea);
    delete rand;
	delete[] head;
	delete[] tail;
	delete[] u[0];
	delete[] u[1];
	delete[] f;
	delete[] sum;
}

Eulerian_Period::~Eulerian_Period(){}

void Eulerian_Period::compute_f(){
	int N=3*local_z*dim[1]*dim[0];	
	// scale
	double scale=sqrt(2*selm->KB*selm->T*selm->deltaT*selm->mu/(dim[0]*dim[1]*dim[2]));
	if(selm->flagStochasticDriving){
		gthm_generate((fftw_complex*)workarea,dim[0],dim[1],dim[2],dx,local_y_start,local_y);
		fftw_execute(plan_c2r_three);
		gthm_scale(workarea,f,scale,dim[0],dim[1],local_z);
	}else{
		//重置f为0
		/* for(int n=0;n<N;n++){
			f[n]=0;
		} */
		memset(f,0,sizeof(double)*N);
	}
}

void Eulerian_Period::run(){
	int N=local_z*dim[1]*dim[0];
	int norm=dim[0]*dim[1]*dim[2];
	distribute_head(u[cur_u],3);
	distribute_tail(u[cur_u],3);
    cur_u = (cur_u +1) % 2;
	u2uAsterisk(selm->deltaT,selm->mu,selm->rho,dim[0],dim[1],local_z,u[(cur_u+1)%2],u[cur_u],f,head,tail);
	distribute_tail(u[cur_u]+2*N);
	div(u[cur_u],dim[0],dim[1],local_z,dx,workarea,tail);
    fftw_execute(plan_r2c);
	r2c_data_transform((fftw_complex*)workarea,dim[0],dim[1],dim[2],dx,local_y,local_y_start);
    fftw_execute(plan_c2r);
	distribute_head_p(workarea);
	uAsterisk2u(workarea,dim[0],dim[1],local_z,dx,u[cur_u],norm,head);
	Sum(u[cur_u]);
	subtract_mean(u[cur_u],sum,N,norm);
}

// Nz=local_z
void Eulerian_Period::u2uAsterisk(double dt,double mu,double rho,int Nx,int Ny,int Nz,double *u, double *u_star, double *f, double *head, double *tail){
	// 每维的网格点数
	int N=Nx*Ny*Nz;
	int Nyx = Ny * Nx;
	int k,j,i;
	// 用于储存中心差分中后一个元素相对当前元素的地址偏移
	int i1, j1;
	// 用于储存中心差分中前一个元素相对当前元素的地址偏移
	int i0, j0;
	// 对于z维由于涉及到边界点，单独计算前后元素的值
	double z_next,z_prev;
	// dx的平方
	double dxsq = dx * dx;
	// 储存lap_u
	double lap_u;
	int n;
	// for循环，分别计算每个维度
	for(int d=0;d<3;d++){
		n=0;
		for(k=0;k<Nz;k++){
			for(j=0;j<Ny;j++){
				for(i=0;i<Nx;i++){
					// 计算前后地址偏移,[ijk]_1代表后一个元素的偏移，[ijk]_0代表前一个元素的偏移，xyz都是周期边界
					i1 = (i != Nx - 1) ? 1 : -i;
					j1 = (j != Ny - 1) ? Nx : -j * Nx;

					i0 = (i != 0) ? -1 : Nx - 1;
					j0 = (j != 0) ? -Nx : (Ny - 1) * Nx;
					z_next = (k != Nz - 1) ? u[n+Nyx] : tail[n % Nyx];
					z_prev = (k != 0) ? u[n-Nyx] : head[n];
					// 计算lap(u),假定dx=dy=dz
					lap_u=(u[n + i1] + u[n + i0] + u[n + j1] + u[n + j0] + z_next + z_prev - 6 * u[n]) / dxsq;
					// 计算u*
                    // printf("[%d][%d]  lap_u: %f  u:  %f  u[%d]: %f  u[%d]: %f  u[%d]: %f  u[%d]: %f  z_next: %f  z_prev: %f\n", d, n, lap_u, u[n], 
                    //         n+i1, u[n+i1], n+i0, u[n+i0],n+j1, u[n+j1],n+j0, u[n+j0],z_next, z_prev);
					u_star[n]=u[n]+(mu*lap_u*dt+f[n])/rho;
                    // printf("[%d][%d]  u*: %f  f: %f  lap_u: %f  u: %f\n", d, n, u_star[n], f[n], lap_u, u[n]);
					n++;
				}
			}
		}
		//计算下一个维度的地址偏移
		u+=N;
		u_star+=N;
		f+=N;
		head+=Nyx;
		tail+=Nyx;
	}
}

// 注意p是按local_z*Ny*(Nx/2+1)*2组织的
void Eulerian_Period::uAsterisk2u(double *p, int Nx, int Ny, int Nz, double dx, double *u, int norm, double *head){
	// 每维的网格点数
	int N=Nx*Ny*Nz;
	int N_x = (Nx/2+1)*2;
	// 当前处理的网格点
	int n;
	// p的地址偏移
	int ind;
	// yx维的元素数量
	int N_yx = Ny * N_x;
	// grad算子前一个元素的值
	double p_prev;
	// 储存grad
	double g;
	int i,j,k;
	for(int d=0; d<3; d++){
		n=0;
		for(k=0;k<Nz;k++){
			for(j=0;j<Ny;j++){
				for(i=0;i<Nx;i++){
					ind=(k*Ny+j)*N_x+i;
					switch(d){
						case 0:
							p_prev = (i != 0) ? p[ind-1] : p[ind+Nx - 1];
							break;
						case 1:
							p_prev = (j != 0) ? p[ind-N_x] : p[ind+(Ny - 1) * N_x];
							break;
						case 2:
							p_prev = (k != 0) ? p[ind-N_yx] : head[ind];
							break;
					}
					// g=grad(ilap(div(u*)))
					// ilap没进行norm，需要除以norm
					g = (p[ind] - p_prev) / (dx * norm);
					// u=u*-grad(ilap(div(u*)))
					u[n] -= g;
					n++;
				}
			}
		}
		// 计算下一个维度的地址偏移
		u+=N;
	}
}

void Eulerian_Period::Sum(double *u){
	int N=dim[0]*dim[1]*local_z;
	for(int d=0;d<3;d++){
		sum[d]=0;
		for(int n=0;n<N;n++){
			sum[d]+=u[n];
		}
		u+=N;
	}
	MPI_Allreduce(MPI_IN_PLACE,sum,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

// Nz=local_z
// 注意为了进行fftw mpi r2c变换，out需要组织为local_z*Ny*(Nx/2+1)*2
void Eulerian_Period::div(double *in, int Nx, int Ny, int Nz, double dx, double *out, double *tail){
	int N = Nx * Ny * Nz;
	int Nyx = Ny * Nx;
	int N_x = (Nx/2+1)*2;
	double *ptr_in[3]={in,in+N,in+2*N};
	double z_next;
	int i,j,k;
	int i1,j1;
	int n=0;
	int ind;
	for(k=0;k<Nz;k++){
		for(j=0;j<Ny;j++){
			for(i=0;i<Nx;i++){
				i1 = (i != Nx - 1) ? 1 : -i;
				j1 = (j != Ny - 1) ? Nx : -j * Nx;
				z_next = (k != Nz - 1) ? ptr_in[2][n+Nyx] : tail[n % Nyx];
				ind=(k*Ny+j)*N_x+i;
				out[ind] = (ptr_in[0][n + i1] - ptr_in[0][n] + ptr_in[1][n + j1] - ptr_in[1][n] + z_next - ptr_in[2][n]) / dx;
				n++;
			}
		}
	}
}

void Eulerian_Period::subtract_mean(double *u, double *sum,int N, int norm){
	double mean;
	int n;
	for(int d=0;d<3;d++){
		mean = sum[d]/norm;
		for(n=0;n<N;n++){
			u[n]-=mean;
		}
		u+=N;
	}
}

void Eulerian_Period::r2c_data_transform(fftw_complex *in, int Nx, int Ny, int Nz, double dx, int local_y, int local_y_start){
	// 要处理的元素相对于in的地址偏移
	int n = 0;
	// 实际x维的尺寸
	int N_x=Nx/2+1;
	int local_y_end=local_y_start+local_y;
	int i,j,k;
	// dx的平方
	double dxsq = dx * dx;
	//注意coef也是共轭对称的,特别注意此时的Nx得使用逻辑上的Nx
	double coef;
	// 用于储存复数的值
	for(j=local_y_start;j<local_y_end;j++){
		for(k=0;k<Nz;k++){
			for(i=0;i<N_x;i++){
				coef = (2 * (cos(2 * M_PI * i / Nx) - 1) + 2 * (cos(2 * M_PI * j / Ny) - 1) + 2 * (cos(2 * M_PI * k / Nz) - 1)) / dxsq;
				if((i+j+k)!=0){
					in[n][0] /= coef;
					in[n][1] /= coef;
				}else{
					in[n][0] = 0.0;
					in[n][1] = 0.0;
				}
				n++;
			}
		}
	}
}

// 生成gthm,输入数组in为local_y*Nz*(Nx/2+1)*3
void Eulerian_Period::gthm_generate(fftw_complex *in, int Nx, int Ny, int Nz, double dx, int local_y_start, int local_y){
	// 要处理的元素相对于in的地址偏移
	int n = 0;
	// 实际x维的尺寸
	int N_x=Nx/2+1;
	int local_y_end=local_y_start+local_y;
	int i,j,k,d;
	int i1,j1,k1;
	// dx的平方
	double dxsq = dx * dx;
	//注意coef也是共轭对称的,特别注意此时的Nx得使用逻辑上的Nx
	// 注意是1-cos,保证结果>=0
	double coef;
	//实部和虚部的系数
	double real,imag;
	for(j=local_y_start;j<local_y_end;j++){
		for(k=0;k<Nz;k++){
			for(i=0;i<N_x;i++){
				coef = (2 * (1 - cos(2 * M_PI * i / Nx)) + 2 * (1 - cos(2 * M_PI * j / Ny)) + 2 * (1 - cos(2 * M_PI * k / Nz))) / dxsq;
				// 计算平方根
				coef = sqrt(coef);
				// 计算共轭对称的坐标ijk_1
				k1 = (Nz-k)%Nz;
				j1 = (Ny-j)%Ny;
				i1 = (Nx-i)%Nx;
				if(k1 != k || j1 != j || i1 != i){
					real = coef / sqrt(2.0);
					imag = real;
				}else{
					real = coef;
					imag = 0.0;
				}
				for(d=0;d<3;d++){
					in[n][0]=rand->gaussian() * real;
					in[n][1]=rand->gaussian() * imag;
					n++;
				}
			}
		}
	}
}

//Nz=local_z
void Eulerian_Period::gthm_scale(double *in, double *out, double scale, int Nx, int Ny, int Nz){
	// 要处理的元素相对于in的地址偏移
	int n=0;
	int ind;
	int i,j,k;
	int d;
	int N_x=(Nx/2+1)*2;
	int N=Nx*Ny*Nz;
	double *ptr_out[3]={out,out+N,out+2*N};
	// 三个维度
	for(k=0;k<Nz;k++){
		for(j=0;j<Ny;j++){
			for(i=0;i<Nx;i++){
				ind=((k*Ny+j)*N_x+i)*3;
				for(d=0;d<3;d++){
					ptr_out[d][n]=in[ind+d]*scale;
				}
				n++;
			}
		}
	}
}

void Eulerian_Period::distribute_head(double *in,int many){
	// 获取进程rank和总的进程数p
	int r;
	int p;
	MPI_Comm_rank(MPI_COMM_WORLD,&r);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	// 每个进程发送最后一层数据给后一个进程succ，并从前一个进程prev接收数据
	int prev=(r-1+p)%p;
	int succ=(r+1+p)%p;
	// 需要发送的数据量
	int Nyx=dim[1]*dim[0];
	// 总的元素数量
	int N=local_z*Nyx;
	// 需要发送数据地址的起始偏移
	double *sendbuf=in+(local_z-1)*Nyx;
	// 进行发送
	for(int d=0;d<many;d++){
		MPI_Sendrecv(sendbuf+d*N,Nyx,MPI_DOUBLE,succ,0,head+d*Nyx,Nyx,MPI_DOUBLE,prev,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
}

void Eulerian_Period::distribute_head_p(double *in){
	// 获取进程rank和总的进程数p
	int r;
	int p;
	MPI_Comm_rank(MPI_COMM_WORLD,&r);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	// 每个进程发送最后一层数据给后一个进程succ，并从前一个进程prev接收数据
	int prev=(r-1+p)%p;
	int succ=(r+1+p)%p;
	// 需要发送的数据量
	int Nyx=dim[1]*dim[0];
	// p是local_z*Ny*N_x维数组
	int N_x=(dim[0]/2+1)*2;
	int N_yx=dim[1]*N_x;
	// 需要发送数据地址的起始偏移
	double *sendbuf=in+(local_z-1)*N_yx;
	MPI_Sendrecv(sendbuf,N_yx,MPI_DOUBLE,succ,0,head,N_yx,MPI_DOUBLE,prev,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

void Eulerian_Period::distribute_tail(double *in,int many){
	// 获取进程rank和总的进程数p
	int r;
	int p;
	MPI_Comm_rank(MPI_COMM_WORLD,&r);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	// 每个进程发送第一层数据给前一个进程prev，并从后一个进程succ接收数据
	int prev=(r-1+p)%p;
	int succ=(r+1+p)%p;
	// 需要发送的数据量
	int Nyx=dim[1]*dim[0];
	// 总的元素数量
	int N=local_z*Nyx;
	// 需要发送数据地址的起始偏移
	double *sendbuf=in;
	// 进行发送
	for(int d=0;d<many;d++){
		MPI_Sendrecv(sendbuf+d*N,Nyx,MPI_DOUBLE,prev,0,tail+d*Nyx,Nyx,MPI_DOUBLE,succ,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
}

