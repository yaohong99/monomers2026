#ifndef OPERATOR_H
#define OPERATOR_H
#define PI 3.14159265358979323846

#include <curand_kernel.h>

void __global__ d2z_data_transform(double2 *in, int Nx, int Ny, int Nz, double dx);
void __global__ gthm_generate(double2 *in, curandState *states, int Nx, int Ny, int Nz, double dx);
void __global__ gthm_scale(double *in, double scale, int N);
void __global__ u2uAsterisk(double *u,double *f,double * uAsterisk, double dx,double dt,double mu,double rho,int Nx,int Ny,int Nz);
void __global__ uAsterisk2u(double *p, int Nx, int Ny, int Nz, double dx, double *u);
void __global__ div(double *in, int Nx, int Ny, int Nz, double dx, double *out);
void __global__ subtract_mean(double *u, double *sum,int N);
void __global__ initCurandStates(curandState* states, unsigned long long seed, int N, int start=0);
void __global__ lagrangian_update(double *x,double *v,double *f,const double * const __restrict__ m,double *u,double *u_f,curandState* states,double dx,double dt,int n_lag,int Nx,int Ny,int Nz,double lo,double hi,double upsilon,double coef,bool rand_flag);
double __device__ gamma(double rx,double ry,double rz,int Nx, int Ny,int Nz, double *u);
void __device__ lambda(double rx,double ry,double rz,int Nx, int Ny,int Nz, double f, double *u_f);

#endif
