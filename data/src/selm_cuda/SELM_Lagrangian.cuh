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
	   double *lo;
	   double *hi;
	   cudaStream_t stream;
       double *d_x;
       double *d_v;
       double *d_f;
       double *d_m;
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