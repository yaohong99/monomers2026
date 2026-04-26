#ifndef SELM_H
#define SELM_H

namespace SELM {

class Selm {
    public:
        int nsteps;
		int rank;
		int size;
        int timestep = 0;
        int SELM_Seed;
		double total_time;
		double euler_time=0,lag_time=0;
		double coupling_time=0,lag_update_time=0,lmp_time=0,lagcomm_time=0;
        int flagStochasticDriving = 0;
        double KB,T,deltaT,mu,rho;
        class Eulerian *eulerian;
        class Lagrangian *lagrangian;
        Selm();
        ~Selm();
        void initial();
        void compute_f();
        void run();
        void final();
};

}

#endif

