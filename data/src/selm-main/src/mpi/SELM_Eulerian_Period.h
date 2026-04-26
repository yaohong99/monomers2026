#ifndef SELM_EULERIAN_PERIOD_H
#define SELM_EULERIAN_PERIOD_H

#include "SELM_Eulerian.h"
#include <fftw3-mpi.h>

namespace LAMMPS_NS {
    class RanMars;
}
namespace SELM {

class Eulerian_Period: public Eulerian {
    public: 
        virtual void initial() override;
        virtual void compute_f() override;
        virtual void run() override;
        virtual void final() override;
        virtual ~Eulerian_Period() override;
        LAMMPS_NS::RanMars *rand;
		int rank;
		double *workarea;
		double *sum;
		fftw_plan plan_r2c, plan_c2r;
		fftw_plan plan_c2r_three;
		double *head,*tail;
		void u2uAsterisk(double dt,double mu,double rho,int Nx,int Ny,int Nz,double *u,double *u_star,double *f, double *head, double *tail);
		void uAsterisk2u(double *p, int Nx, int Ny, int Nz, double dx, double *u, int norm, double *head);
		void div(double *in, int Nx, int Ny, int Nz, double dx, double *out, double *tail);
		void subtract_mean(double *u, double *sum,int N, int norm);
		void r2c_data_transform(fftw_complex *in, int Nx, int Ny, int Nz, double dx, int local_y, int local_y_start);
		void distribute_head(double *in,int many=1);
		void distribute_head_p(double *in);
		void distribute_tail(double *in,int many=1);
		void Sum(double *u);
		void gthm_scale(double *in, double *out, double scale, int Nx, int Ny, int Nz);
		void gthm_generate(fftw_complex *in, int Nx, int Ny, int Nz, double dx, int local_y_start, int local_y);
};

}

#endif
