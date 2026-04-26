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
		// workarea,用于临时储存数据
		double *d_workarea;
		// cufft plan,用于ilap
		cufftHandle plan_D2Z,plan_Z2D;
		// 储存cub归约的值
		double *d_sum;
		// cub的工作空间
		void *d_cubarea;
		// cub workarea size
		size_t cub_size;
		// curand states
		curandState* d_states;
		// cufft plan,用于gthm
		cufftHandle plan_Z2D_three;
};

}

#endif
