#ifndef SELM_EULERIAN_PERIOD_H
#define SELM_EULERIAN_PERIOD_H

#include "SELM_Eulerian.cuh"
#include <cufft.h>
#include <curand_kernel.h>

namespace SELM {

class Eulerian_Period: public Eulerian {
    public: 
        virtual void initial() override;
        virtual void compute_f() override;
        virtual void run() override;
        virtual void final() override;
        virtual ~Eulerian_Period() override;
		double *d_workarea;
		cufftHandle plan_D2Z,plan_Z2D;
		
		double *d_sum;
		
		void *d_cubarea;
		size_t cub_size;
		
		curandState* d_states;
		
		cufftHandle plan_Z2D_three;
};

}

#endif
