#ifndef SELM_EULERIAN_H
#define SELM_EULERIAN_H

namespace SELM {

class Eulerian {
    public:
		
        int dim[3];
        double dx;
		
        double *d_u;
        double *d_f;
		cudaStream_t stream;
		
        class Selm *selm;
        virtual void initial() = 0;
        virtual void compute_f() = 0;
        virtual void run() = 0;
        virtual void final() = 0;
        virtual ~Eulerian()=default;

};

}

#endif
