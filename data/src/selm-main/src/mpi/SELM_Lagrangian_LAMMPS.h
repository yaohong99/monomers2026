#ifndef SELM_LAGRANGIAN_LAMMPS_H
#define SELM_LAGRANGIAN_LAMMPS_H

#include "SELM_Lagrangian.h"
#include <mpi.h>

namespace LAMMPS_NS {
    class RanMars;
}

namespace SELM {

class Lagrangian_LAMMPS:public Lagrangian{ 
    public:
        Lagrangian_LAMMPS();
        ~Lagrangian_LAMMPS() override;
        void initial() override;
        void compute_f() override;
        void run() override;
        void final() override;
    private:
		// 进程rank
        int rank;
		// 每个rank一个通信域
        MPI_Comm lagcomm;
		// 随机数生成器
        LAMMPS_NS::RanMars *rand;
		// 储存gamma(u)
		double *f_tmp;
		// 储存x的网格坐标
		double *x_grid;
		// 写入数据
		void writedata();
		// 拉格朗日点的更新
		void lagrangian_update(double *u_f,double dx,int Nx,int Ny,int Nz,int local_z_start,int local_z_end,double upsilon,double coef,bool rand_flag);
		// gamma算子的计算
		void compute_gamma(double *u,double dx,int Nx,int Ny,int Nz,int local_z_start,int local_z_end);
		double gamma(double rx,double ry,double rz,int Nx, int Ny,int Nz, double *u,int local_z_start, int local_z_end);
		// lambda算子的计算
		void lambda(double rx,double ry,double rz,int Nx, int Ny,int Nz, double f, double *u_f, int local_z_start, int local_z_end);
};

}

#endif
