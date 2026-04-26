#ifndef SELM_H
#define SELM_H

namespace SELM {

class Selm {
    public:
		// 总共需要迭代的时间步
        int nsteps;
		// 当前时间步
        int timestep = 0;
		double total_time;
		// 随机数种子
        int SELM_Seed = 123456;
		// 是否生成随机场
        bool flagStochasticDriving = 0;
		// 一些常量
        double KB,T,deltaT,mu,rho;
		// Euler和Lagrangian类对象指针
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

