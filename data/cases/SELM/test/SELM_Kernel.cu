#include "SELM_Kernel.cuh"

void __global__ d2z_data_transform(double2 *in, int Nx, int Ny, int Nz, double dx){
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int N_x=Nx/2+1;
	int N = N_x * Ny * Nz;
	int Nyx = Ny * N_x;
	int k = n / Nyx;
	int j = n % Nyx / N_x;
	int i = n % N_x;
	double dxsq = dx * dx;
	double coef = (2 * (cos(2 * PI * i / Nx) - 1) + 2 * (cos(2 * PI * j / Ny) - 1) + 2 * (cos(2 * PI * k / Nz) - 1)) / dxsq;
	double2 val;
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

void __global__ gthm_generate(double2 *in, curandState *states, int Nx, int Ny, int Nz, double dx){
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int N_x=Nx/2+1;
	int N = N_x * Ny * Nz;
	if(n>=N) return;
	int Nyx = Ny * N_x;
	int k = n / Nyx;
	int j = n % Nyx / N_x;
	int i = n % N_x;
	double dxsq = dx * dx;
	double coef = (2 * (1 - cos(2 * PI * i / Nx)) + 2 * (1 - cos(2 * PI * j / Ny)) + 2 * (1 - cos(2 * PI * k / Nz))) / dxsq;
	coef = sqrt(coef);
	int k1 = (Nz-k)%Nz;
	int j1 = (Ny-j)%Ny;
	int i1 = (Nx-i)%Nx;
	double real,imag;
	if(k1 != k || j1 != j || i1 != i){
		real = coef / sqrt(2.0);
		imag = real;
	}else{
		real = coef;
		imag = 0.0;
	}
	curandState state=states[n];
	double2 val;
	for(int d=0;d<3;d++){
		val.x=curand_normal_double(&state) * real;
		val.y=curand_normal_double(&state) * imag;
		in[n]=val;
		in+=N;
	}
	states[n]=state;
}

void __global__ gthm_scale(double *in, double scale, int N){
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n>=N) return;
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

void __global__ uAsterisk2u(double *p, int Nx, int Ny, int Nz, double dx, double *u){
	int N=Nx*Ny*Nz;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n>=N) return;
	int Nyx = Ny * Nx;
	int offset;
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
		g = (p[n] - p[n + offset]) / (dx * N);
		u[n] -= g;
		u+=N;
	}
}

void __global__ u2uAsterisk(double *u,double *f,double dx,double dt,double mu,double rho,int Nx,int Ny,int Nz){
	int N=Nx*Ny*Nz;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n>=N) return;
	int Nyx = Ny * Nx;
	int k = n / Nyx;
	int j = n % Nyx / Nx;
	int i = n % Nx;
	int i1, j1, k1;
	double dxsq = dx * dx;
	i1 = (i != Nx - 1) ? 1 : -i;
	j1 = (j != Ny - 1) ? Nx : -j * Nx;
	k1 = (k != Nz - 1) ? Nyx : -k * Nyx;

	i = (i != 0) ? -1 : Nx - 1;
	j = (j != 0) ? -Nx : (Ny - 1) * Nx;
	k = (k != 0) ? -Nyx : (Nz - 1) * Nyx;
	double lap_u;
	for(int d=0;d<3;d++){
		lap_u=(u[n + i1] + u[n + i] + u[n + j1] + u[n + j] + u[n + k1] + u[n + k] - 6 * u[n]) / dxsq;
		u[n]+=(mu*lap_u*dt+f[n])/rho;
		u+=N;
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

void __global__ lagrangian_update(double *x,double *v,double *f,const double * const __restrict__ m,double *u,double *u_f,curandState* states,double dx,double dt,int n_lag,int Nx,int Ny,int Nz,double lo,double hi,double upsilon,double coef,bool rand_flag){

	if(blockIdx.x*blockDim.x+threadIdx.x>=n_lag) return;
	int blockDim_x=(blockIdx.x!=gridDim.x-1)? blockDim.x:n_lag-blockDim.x*blockIdx.x;
	int tid=threadIdx.y*blockDim_x+threadIdx.x;
	int gid=blockIdx.x*blockDim.x*blockDim.y+tid;
	__shared__ double s_x[384];
	__shared__ double s_f[406];
	double t_x=x[gid];
	s_x[tid]=(t_x-lo)/dx;
	__syncthreads();
	int N = Nx*Ny*Nz;
	int ind=3*threadIdx.x;

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
	__syncthreads();
	double t_f =-upsilon*(v[gid]-s_f[tid])*dt;
	if(rand_flag) t_f+=curand_normal_double(states+gid) * coef;
	ind=(tid%3)*(blockDim.x+11)+tid/3;
	s_f[ind]=t_f;
	__syncthreads();
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
	t_f+=dt*f[gid];
	f[gid]=t_f/dt;
	double tmp_x=v[gid]*dt;
	double L=hi-lo;
	if(fabs(tmp_x)>L/2) {
		printf("g.x: %d b.x: %d t.x:%d t.y: %d gid: %d dt: %f v[%d]: %f x[%d]: %f L: %f\n", gridDim.x,blockIdx.x,threadIdx.x,threadIdx.y,gid,dt,gid,v[gid],gid,x[gid],L);
	}
	t_x+=tmp_x;
	if(t_x>hi) t_x-=L;
	else if(t_x<lo) t_x+=L;
	x[gid]=t_x;
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
				atomicAdd(u_f+n,-f*delta_i);
			}
		}
	}
}