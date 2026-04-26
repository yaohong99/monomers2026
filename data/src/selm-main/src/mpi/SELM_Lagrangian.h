#ifndef SELM_LAGRANGIAN_H
#define SELM_LAGRANGIAN_H

namespace LAMMPS_NS {
    class LAMMPS;
}

namespace SELM {

class Lagrangian {
    public:
       double *x;
       double *v;
       double *f;
       double *m;
	   // box尺寸,指向lmp->domain->boxlo/hi的指针
	   double *lo;
	   double *hi;
       int nlocal;
	   int saveSkipSimulationData=0;
	   int writeParticalV=0,writeParticalF=0,writeParticalX=0;
       LAMMPS_NS::LAMMPS *lmp;
       class Selm *selm;
       virtual ~Lagrangian()=default;
       virtual void initial() = 0;
       virtual void compute_f() = 0;
       virtual void run() = 0;
       virtual void final() = 0;

};

}

#endif
