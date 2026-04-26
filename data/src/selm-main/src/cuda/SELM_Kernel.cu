#include "SELM_Kernel.cuh"

void __global__ d2z_data_transform(double2 *in, int Nx, int Ny, int Nz, double dx){
	// 要处理的元素相对于in的地址偏移
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// 实际x维的尺寸
	int N_x=Nx/2+1;
	// 总的复数的个数N
	int N = N_x * Ny * Nz;
	// yx维的尺寸
	int Nyx = Ny * N_x;
	// 网格坐标(i,j,k)
	int k = n / Nyx;
	int j = n % Nyx / N_x;
	int i = n % N_x;
	// dx的平方
	double dxsq = dx * dx;
	//注意coef也是共轭对称的,特别注意此时的Nx得使用逻辑上的Nx
	double coef = (2 * (cos(2 * PI * i / Nx) - 1) + 2 * (cos(2 * PI * j / Ny) - 1) + 2 * (cos(2 * PI * k / Nz) - 1)) / dxsq;
	// 用于储存复数的值
	double2 val;
	// 只需要处理N个复数
	if(n < N){
		val=in[n];
		if(n != 0){
			val.x /= coef;
			val.y /= coef;
		}else{
			val.x = 0.0;
			val.y = 0.0;
		}
		in[n]=val;
	}
}

// 生成gthm,输入数组in为3*Nz*Ny*(Nx/2+1),state为Nz*Ny*(Nx/2+1)
void __global__ gthm_generate(double2 *in, curandState *states, int Nx, int Ny, int Nz, double dx){
	// 要处理的元素相对于in的地址偏移
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// 实际x维的尺寸
	int N_x=Nx/2+1;
	// 总的复数的个数N
	int N = N_x * Ny * Nz;
	// 多余的线程退出
	if(n>=N) return;
	// yx维的尺寸
	int Nyx = Ny * N_x;
	// 网格坐标(i,j,k)
	int k = n / Nyx;
	int j = n % Nyx / N_x;
	int i = n % N_x;
	// dx的平方
	double dxsq = dx * dx;
	//注意coef也是共轭对称的,特别注意此时的Nx得使用逻辑上的Nx
	// 注意是1-cos,保证结果>=0
	double coef = (2 * (1 - cos(2 * PI * i / Nx)) + 2 * (1 - cos(2 * PI * j / Ny)) + 2 * (1 - cos(2 * PI * k / Nz))) / dxsq;
	// 计算平方根
	coef = sqrt(coef);
	// 计算共轭对称的坐标ijk_1
	int k1 = (Nz-k)%Nz;
	int j1 = (Ny-j)%Ny;
	int i1 = (Nx-i)%Nx;
	//实部和虚部的系数
	double real,imag;
	if(k1 != k || j1 != j || i1 != i){
		real = coef / sqrt(2.0);
		imag = real;
	}else{
		real = coef;
		imag = 0.0;
	}
	// 读取states
	curandState state=states[n];
	// 用于储存复数的值
	double2 val;
	// 总共三个维度
	for(int d=0;d<3;d++){
		val.x=curand_normal_double(&state) * real;
		val.y=curand_normal_double(&state) * imag;
		in[n]=val;
		in+=N;
	}
	// 将state写回
	states[n]=state;
}

// 进行缩放
void __global__ gthm_scale(double *in, double scale, int N){
	// 要处理的元素相对于in的地址偏移
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// 多余的进程退出
	if(n>=N) return;
	// 三个维度
	for(int d=0;d<3;d++){
		in[n]*=scale;
		in+=N;
	}
}

void __global__ div(double *in, int Nx, int Ny, int Nz, double dx, double *out){
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int N = Nx * Ny * Nz;
	int Nyx = Ny * Nx;
	int k = n / Nyx;
	int j = (n - k * Nyx) / Nx;
	int i = n - k * Nyx -j * Nx;
	double *ptr_in[3]={in,in+N,in+2*N};

	if(n < N){	
		i = (i != Nx - 1) ? 1 : -i;
		j = (j != Ny - 1) ? Nx : -j * Nx;
		k = (k != Nz - 1) ? Nyx : -k * Nyx;

		out[n] = (ptr_in[0][n + i] - ptr_in[0][n] + ptr_in[1][n + j] - ptr_in[1][n] + ptr_in[2][n + k] - ptr_in[2][n]) / dx;
	}
}

// 由u*计算u_n+1, blocksize为128, p代表ilap(div(u*))
void __global__ uAsterisk2u(double *p, int Nx, int Ny, int Nz, double dx, double *u){
	// 每维的网格点数
	int N=Nx*Ny*Nz;
	// 当前thread处理的网格点
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// 多余的thread退出
	if(n>=N) return;
	// yx维的元素数量
	int Nyx = Ny * Nx;
	// grad算子前一个元素的偏移
	int offset;
	// 储存grad
	double g;
	for(int d=0; d<3; d++){
		switch(d){
			case 0:
				offset = (n % Nx != 0) ? -1 : Nx - 1;
				break;
			case 1:
				offset = (n % Nyx / Nx != 0) ? -Nx : (Ny - 1) * Nx;
				break;
			case 2:
				offset = (n / Nyx != 0) ? -Nyx : (Nz - 1) * Nyx;
				break;
		}
		// g=grad(ilap(div(u*)))
		// 如果ilap没进行norm，可改为除dx*N
		g = (p[n] - p[n + offset]) / (dx * N);
		// double g = (p[n] - p[n + offset]) / dx;
		u[n] -= g;
		// 计算下一个维度的地址偏移
		u+=N;
	}
}

// 由u计算u*,blocksize为128
void __global__ u2uAsterisk(double *u,double *f,double *uAsterisk, double dx,double dt,double mu,double rho,int Nx,int Ny,int Nz){
	// 每维的网格点数
	int N=Nx*Ny*Nz;
	// 当前thread处理的网格点
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// 多余的thread退出
	if(n>=N) return;
	// 计算网格坐标
	int Nyx = Ny * Nx;
	int k = n / Nyx;
	int j = n % Nyx / Nx;
	int i = n % Nx;
	// 用于储存中心差分中后一个元素相对当前元素的地址偏移
	int i1, j1, k1;
	// dx的平方
	double dxsq = dx * dx;
	// 计算前后地址偏移,[ijk]_1代表后一个元素的偏移，[ijk]代表前一个元素的偏移，xyz都是周期边界
	i1 = (i != Nx - 1) ? 1 : -i;
	j1 = (j != Ny - 1) ? Nx : -j * Nx;
	k1 = (k != Nz - 1) ? Nyx : -k * Nyx;

	i = (i != 0) ? -1 : Nx - 1;
	j = (j != 0) ? -Nx : (Ny - 1) * Nx;
	k = (k != 0) ? -Nyx : (Nz - 1) * Nyx;
	// 储存lap_u;
	double lap_u;
	double reg_u;
	// for循环，分别计算每个维度
	for(int d=0;d<3;d++){
		// 计算lap(u),假定dx=dy=dz
        reg_u = u[n];
		lap_u=(u[n + i1] + u[n + i] + u[n + j1] + u[n + j] + u[n + k1] + u[n + k] - 6 * reg_u) / dxsq;
		// 计算u*
		uAsterisk[n] = reg_u + (mu*lap_u*dt+f[n])/rho;
        // printf("[%d][%d]  u*: %f  f: %f  lap_u: %f  u: %f\n", d, n, uAsterisk[n], f[n], lap_u, reg_u);
		//计算下一个维度的地址偏移
		u+=N;
		uAsterisk+=N;
		f+=N;
	}
}

void __global__ subtract_mean(double *u, double *sum,int N){
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	double mean;
	if(n<N){
		for(int d=0;d<3;d++){
			mean = sum[d]/N;
			u[n]-=mean;
			u+=N;
		}
	}
}

void __global__ initCurandStates(curandState* states, unsigned long long seed, int N, int start){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        curand_init(seed, start+n, 0, states+n);
    }
}

/* // 假定块尺寸为128*3,gridDim.x=1+(3*n_lag-1)/(128*3)=1+(n_lag-1)/128
// 假定各维的lo hi deltaX Nz Ny Nx都是一样的
void __global__ lagrangian_update(double *x,double *v,double *f,const double * const __restrict__ m,double *u,double *u_f,curandState* states,double dx,double dt,int n_lag,int Nx,int Ny,int Nz,double lo,double hi,double upsilon,double coef,bool rand_flag){

	// 每个thread处理一个点，多余的thread退出
	if(blockIdx.x*blockDim.x+threadIdx.x>=n_lag) return;
	// 实际的x维的块尺寸，blockDim_x代表了每个块实际处理的Larangian点的个数 对于最后一个block,将其视为blockDim_x*blockDim.y的块
	int blockDim_x=(blockIdx.x!=gridDim.x-1)? blockDim.x:n_lag-blockDim.x*blockIdx.x;
	// 块内索引
	int tid=threadIdx.y*blockDim_x+threadIdx.x;
	// 全局索引
	int gid=blockIdx.x*blockDim.x*blockDim.y+tid;
	// 使用共享内存储存x的网格坐标,blocksize为128*3
	__shared__ double s_x[384];
	// 临时储存计算过程中的f,多分配22个元素，分别用于x与y,y与z之间的11个元素的填充，以避免bank冲突
	__shared__ double s_f[406];
	// 每个线程对x的备份
	double t_x=x[gid];
	// 将x在网格中的坐标写入s_x
	s_x[tid]=(t_x-lo)/dx;
	// 确保s_x写入完成
	__syncthreads();
	// 总Eulerian点数量
	int N = Nx*Ny*Nz;
	// 每个thread开始位置的索引
	int ind=3*threadIdx.x;

	// Gamma算子的计算
	// s_f=Gamma(u); 按xyz交替排列
	// blockIdx.y分别代表三个x,y,z方向上的力
	switch(threadIdx.y){
		case 0:
			s_f[ind]=gamma(s_x[ind],s_x[ind+1]-0.5,s_x[ind+2]-0.5,Nx,Ny,Nz,u);
			break;
		case 1:
			s_f[ind+1]=gamma(s_x[ind]-0.5,s_x[ind+1],s_x[ind+2]-0.5,Nx,Ny,Nz,u+N);
			break;
		case 2:
			s_f[ind+2]=gamma(s_x[ind]-0.5,s_x[ind+1]-0.5,s_x[ind+2],Nx,Ny,Nz,u+2*N);
			break;
	}
	// 确保s_y写入完成
	__syncthreads();
	// t_f=-upsilon(v-gamma(u))*dt  dt必须先乘上，因为lambda算子计算得到的力默认是带dt的 以及 生成的随机场也是带dt的.
	double t_f =-upsilon*(v[gid]-s_f[tid])*dt;
	// printf("t_f:%f (%d, %d, %d) -%f*(%f-%f)\n",t_f,blockIdx.x,threadIdx.x,threadIdx.y,upsilon,v[gid],s_f[tid]);
	// t_f+=Fthm
	if(rand_flag) t_f+=curand_normal_double(states+gid) * coef;
	// 将数据写回s_f，用于Lambda算子的计算，先排列完x,再排列y,最后排列z，中间填充11个元素,避免写入时的bank冲突
	ind=(tid%3)*(blockDim.x+11)+tid/3;
	s_f[ind]=t_f;
	// 确保s_y写入完成
	__syncthreads();
	// u_f-=lambda(t_f),u_f必须先初始化
	ind=3*threadIdx.x;
	switch(threadIdx.y){
		case 0:
			lambda(s_x[ind],s_x[ind+1]-0.5,s_x[ind+2]-0.5,Nx,Ny,Nz,s_f[threadIdx.x],u_f);
			break;
		case 1:
			lambda(s_x[ind]-0.5,s_x[ind+1],s_x[ind+2]-0.5,Nx,Ny,Nz,s_f[threadIdx.x+blockDim.x+11],u_f+N);
			break;
		case 2:
			lambda(s_x[ind]-0.5,s_x[ind+1]-0.5,s_x[ind+2],Nx,Ny,Nz,s_f[threadIdx.x+2*(blockDim.x+11)],u_f+2*N);
			break;
	}
	// 更新f
    t_f += 100;
	t_f+=dt*f[gid];
	f[gid]=t_f/dt;
	// 更新x
	double tmp_x=v[gid]*dt;
	double L=hi-lo;
	// 某个粒子的位移超过盒子的一半代表模拟中出现了错误，可能是时间间隔太大
	if(fabs(tmp_x)>L/2) {
		printf("g.x: %d b.x: %d t.x:%d t.y: %d gid: %d dt: %f v[%d]: %f x[%d]: %f L: %f\n", gridDim.x,blockIdx.x,threadIdx.x,threadIdx.y,gid,dt,gid,v[gid],gid,x[gid],L);
	}
	t_x+=tmp_x;
	if(t_x>hi) t_x-=L;
	else if(t_x<lo) t_x+=L;
	x[gid]=t_x;
	// 更新v
	v[gid]+=t_f/m[gid/3];

} */

void __global__ lagrangian_update(double *x,double *v,double *f,const double * const __restrict__ m,double *u,double *u_f,curandState* states,double dx,double dt,int n_lag,int Nx,int Ny,int Nz,double lo,double hi,double upsilon,double coef,bool rand_flag){

	// 全局索引, 总共ceil(nlocal*3/blockDim.x)个块, blockDim.x=384
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
    // 线程块内索引, 0到3*128-1
    int tid=threadIdx.x;
	// 每个thread处理原子的一个维度，多余的thread退出
	if(gid>=n_lag*3) return;
	// 使用共享内存储存x的网格坐标,blocksize为128*3
	__shared__ double s_x[384];
	// 每个线程对x的备份
	double t_x=x[gid];
	// 将x在网格中的坐标写入s_x
	s_x[tid]=(t_x-lo)/dx;
	// 确保s_x写入完成
	__syncthreads();
	// 总Eulerian点数量
	int N = Nx*Ny*Nz;
	// 每个thread开始位置的索引

	// Gamma算子的计算
	// t_f=Gamma(u); 按xyz交替排列
    double t_f;
    // ind表示s_x的起始位置
    int ind = (tid / 3) * 3;
	// tid % 3 分别代表三个x,y,z方向上的力
	switch(tid % 3){
		case 0:
			t_f=gamma(s_x[ind],s_x[ind+1]-0.5,s_x[ind+2]-0.5,Nx,Ny,Nz,u);
			break;
		case 1:
			t_f=gamma(s_x[ind]-0.5,s_x[ind+1],s_x[ind+2]-0.5,Nx,Ny,Nz,u+N);
			break;
		case 2:
			t_f=gamma(s_x[ind]-0.5,s_x[ind+1]-0.5,s_x[ind+2],Nx,Ny,Nz,u+2*N);
			break;
	}
	// t_f=-upsilon(v-gamma(u))*dt  dt必须先乘上，因为lambda算子计算得到的力默认是带dt的 以及 生成的随机场也是带dt的.
	t_f =-upsilon*(v[gid]-t_f)*dt;
	// t_f+=Fthm
	if(rand_flag) t_f+=curand_normal_double(states+gid) * coef;
	// u_f-=lambda(t_f),u_f必须先初始化
	switch(tid%3){
		case 0:
			lambda(s_x[ind],s_x[ind+1]-0.5,s_x[ind+2]-0.5,Nx,Ny,Nz,t_f,u_f);
			break;
		case 1:
			lambda(s_x[ind]-0.5,s_x[ind+1],s_x[ind+2]-0.5,Nx,Ny,Nz,t_f,u_f+N);
			break;
		case 2:
			lambda(s_x[ind]-0.5,s_x[ind+1]-0.5,s_x[ind+2],Nx,Ny,Nz,t_f,u_f+2*N);
			break;
	}
	// 更新f
    // t_f += 100;
	t_f+=dt*f[gid];
	f[gid]=t_f/dt;
	// 更新x
	double tmp_x=v[gid]*dt;
	double L=hi-lo;
	// 某个粒子的位移超过盒子的一半代表模拟中出现了错误，可能是时间间隔太大
	if(fabs(tmp_x)>L/2) {
		printf("g.x: %d b.x: %d t.x:%d t.y: %d gid: %d dt: %f v[%d]: %f x[%d]: %f L: %f\n", gridDim.x,blockIdx.x,threadIdx.x,threadIdx.y,gid,dt,gid,v[gid],gid,x[gid],L);
	}
	t_x+=tmp_x;
	if(t_x>hi) t_x-=L;
	else if(t_x<lo) t_x+=L;
	x[gid]=t_x;
	// 更新v
	v[gid]+=t_f/m[gid/3];

}

double __device__ gamma(double rx,double ry,double rz,int Nx, int Ny,int Nz, double *u){
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
				n=(p[2]*Ny+p[1])*Nx+p[0];
				rv+=u[n]*delta_i;
			}
		}
	}
	return rv;
}

void __device__ lambda(double rx,double ry,double rz,int Nx, int Ny,int Nz, double f, double *u_f){
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
				n=(p[2]*Ny+p[1])*Nx+p[0];
				// 必须使用原子函数，以避免竞争读写的问题，u_f必须先初始化
				atomicAdd(u_f+n,-f*delta_i);
			}
		}
	}
}
