#ifndef SELM_EULERIAN_H
#define SELM_EULERIAN_H

namespace SELM {

class Eulerian {
    public:
        int cur_u;
        double *u[2];
        double *f;
        double dx;
        int dim[3];
        int local_z, local_z_start;
        int local_y, local_y_start;
        class Selm *selm;
        virtual void initial() = 0;
        virtual void compute_f() = 0;
        virtual void run() = 0;
        virtual void final() = 0;
        virtual ~Eulerian()=default;

};

}

#endif
