#ifndef SELM_H
#define SELM_H

namespace SELM {

class Selm {
    public:
		
        int nsteps;
		
        int timestep = 0;
		double total_time;
		
        int SELM_Seed;
		
        bool flagStochasticDriving = 0;
		
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

