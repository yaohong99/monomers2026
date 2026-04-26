#ifndef SELM_EULERIAN_H
#define SELM_EULERIAN_H

namespace SELM {

class Eulerian {
    public:
		// 每维的点数和DeltaX
        int dim[3];
        double dx;
		// u和f存放在GPU中
		// d_f是存储3*N个double类型的数组
        double *d_f;
        // 分配两块u内存, cur_u表示当前u在哪一块内存, 另一块存储u*
        int cur_u;
        double *d_u[2];
        // 储存压力，N个double数组
        double *d_p;
		// cuda stream,用于overlap拉格朗日点在LAMMPS中的计算
		cudaStream_t stream;
		// selm类对象指针
        class Selm *selm;
        virtual void initial() = 0;
        virtual void compute_f() = 0;
        virtual void run() = 0;
        virtual void final() = 0;
        virtual ~Eulerian()=default;

};

}

#endif
