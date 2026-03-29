#ifndef SELM_LAGRANGIAN_LAMMPS_H
#define SELM_LAGRANGIAN_LAMMPS_H

#include "SELM_Lagrangian.cuh"

#include <curand_kernel.h>

namespace SELM {

class Lagrangian_LAMMPS : public Lagrangian{ 
    public:
        Lagrangian_LAMMPS();
        ~Lagrangian_LAMMPS() override;
        void initial() override;
        void compute_f() override;
        void run() override;
        void final() override;
    private:
		void writedata();
		
		curandState* d_states;
		
		double *x_pin;
		double *v_pin;
		double *f_pin;
};

}

#endif
