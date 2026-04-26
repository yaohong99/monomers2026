#ifndef SELM_LAGRANGIAN_H
#define SELM_LAGRANGIAN_H

namespace LAMMPS_NS {
    class LAMMPS;
}

namespace SELM {

class Lagrangian {
    public:
		// 粒子的坐标 速度 作用力 质量
       double *x;
       double *v;
       double *f;
       double *m;
	   // box尺寸,指向lmp->domain->boxlo/hi的指针
	   double *lo;
	   double *hi;
	   // stream
	   cudaStream_t stream;
	   // GPU中的数据
       double *d_x;
       double *d_v;
       double *d_f;
       double *d_m;
	   // 粒子的个数
       int nlocal;
	   // 输出的步长
	   int saveSkipSimulationData=0;
	   // 是否输出V F X
	   int writeParticalV=0,writeParticalF=0,writeParticalX=0;
	   // lammps类指针
       LAMMPS_NS::LAMMPS *lmp;
	   // selm类指针
       class Selm *selm;
	   // 必须声明为虚函数，以便能够子类的析构函数
       virtual ~Lagrangian()=default;
	   // 纯虚函数，为不同的子类提供统一的接口
	   // 初始化
       virtual void initial() = 0;
	   // 力的计算
       virtual void compute_f() = 0;
	   // 更新位移 速度 力
       virtual void run() = 0;
	   // 内存释放相关的收尾工作
       virtual void final() = 0;

};

}

#endif
