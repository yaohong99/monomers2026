#include "SELM.cuh"
#include "SELM_Eulerian.cuh"
#include "SELM_Lagrangian.cuh"
#include "SELM_Eulerian_Period.cuh"

#include "check.cuh"
#include "SELM_Kernel.cuh"
#include <cub/cub.cuh>

#include <cmath>

using namespace SELM;

void Eulerian_Period::initial(){
    cur_u = 0;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*1024*1024); // 10MB
	// 初始化dx，假定各维的dx,hi,lo,Nx Ny Nz都是一样的
	dx=(selm->lagrangian->hi[0]-selm->lagrangian->lo[0])/dim[0];
	// 总的元素数量
	int N=dim[0]*dim[1]*dim[2];
	// d2z变换后复数的数量
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	// 初始化stream
	CHECK_RUNTIME(cudaStreamCreate(&stream));
	// 初始化cufft plan
	cufftPlan3d(&plan_D2Z, dim[2], dim[1], dim[0], CUFFT_D2Z);
	cufftPlan3d(&plan_Z2D, dim[2],dim[1],dim[0], CUFFT_Z2D);
	// 将计划与stream绑定
	cufftSetStream(plan_D2Z,stream);
	cufftSetStream(plan_Z2D,stream);
	// 每个维度需要分配的内存大小
	size_t size=sizeof(double)*N*3;
	// 在GPU中为u和f分配内存，并初始化u为0
	CHECK_RUNTIME(cudaMalloc((void**)d_u,size));
	CHECK_RUNTIME(cudaMalloc((void**)(d_u+1),size));
	CHECK_RUNTIME(cudaMalloc((void**)&d_f,size));
	CHECK_RUNTIME(cudaMalloc((void**)&d_p,sizeof(double)*N));
	CHECK_RUNTIME(cudaMemset(d_u[0],0,size));	
	CHECK_RUNTIME(cudaMemset(d_u[1],0,size));	
	// 为workarea分配内存，最后一维需要填充以容纳D2Z变换后的大小
	size = sizeof(double2)*N2;
	CHECK_RUNTIME(cudaMalloc((void**)&d_workarea,size*3));
	// 分配sum的内存
	CHECK_RUNTIME(cudaMalloc((void**)&d_sum,sizeof(double)*3));
	// 确定cub所需的工作空间大小
	d_cubarea = nullptr;
    cub_size = 0;
	CHECK_RUNTIME(cub::DeviceReduce::Sum(nullptr, cub_size, d_u[cur_u], d_sum, N, 0));
	CHECK_RUNTIME(cudaMalloc((void**)&d_cubarea,cub_size));
	// 初始化d_states为nullptr
	d_states=nullptr;
	if(selm->flagStochasticDriving){
		// 分配d_states的内存
		size=sizeof(curandState)*N2;
		CHECK_RUNTIME(cudaMalloc((void**)&d_states,size));
		initCurandStates<<<1+(N2-1)/128,128,0,stream>>>(d_states,selm->SELM_Seed,N2,1);
		CHECK_RUNTIME(cudaStreamSynchronize(stream));
	}
	// 3d D2Z的维度
	int n[3]={dim[2],dim[1],dim[0]};
	// 相邻两个输入batch之间的距离，以double为单位
	int idist=N;
	// 相邻两个输出batch之间的距离，以complex为单位
	int odist=N2;
	// 输入数组的维度
	int inembed[3]={dim[2],dim[1],dim[0]};
	// 输出数组的维度
	int onembed[3]={dim[2],dim[1],dim[0]/2+1};
	// 批次为3,输入输出步长为1
	cufftPlanMany(&plan_Z2D_three,3,n,onembed,1,odist,inembed,1,idist,CUFFT_Z2D,3);
	// 将计划与stream绑定
	cufftSetStream(plan_Z2D_three,stream);
	// 生成力
	compute_f();
	CHECK_RUNTIME(cudaStreamSynchronize(stream));
}

void Eulerian_Period::final(){
	CHECK_RUNTIME(cudaFree(d_u[0]));
	CHECK_RUNTIME(cudaFree(d_u[1]));
	CHECK_RUNTIME(cudaFree(d_workarea));
	CHECK_RUNTIME(cudaFree(d_f));
	CHECK_RUNTIME(cudaFree(d_p));
	CHECK_RUNTIME(cudaFree(d_cubarea));
	CHECK_RUNTIME(cudaFree(d_sum));
	CHECK_RUNTIME(cudaFree(d_states));
	CHECK_RUNTIME(cudaStreamDestroy(stream));
	cufftDestroy(plan_D2Z);
	cufftDestroy(plan_Z2D);
	cufftDestroy(plan_Z2D_three);
}

Eulerian_Period::~Eulerian_Period(){}


void Eulerian_Period::run(){
	// 总的元素数量
	int N=dim[0]*dim[1]*dim[2];
	// d2z变换的总的复数元素
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	// 块尺寸
	int blocksize=128;
	// 网格尺寸
	int gridsize=1+(N-1)/blocksize;
	int gridsize2=1+(N2-1)/blocksize;
    // 交换u的存储位置
    cur_u = (cur_u +1) % 2;
	// 计算u*
	u2uAsterisk<<<gridsize,blocksize,0,stream>>>(d_u[(cur_u+1)%2],d_f,d_u[cur_u],dx,selm->deltaT,selm->mu,selm->rho,dim[0],dim[1],dim[2]);
	// 计算散度
	div<<<gridsize,blocksize,0,stream>>>(d_u[cur_u],dim[0],dim[1],dim[2],dx,d_p);
	// 计算d2z
	cufftExecD2Z(plan_D2Z, d_p, (cufftDoubleComplex*)d_workarea);
	// 对数据进行缩放
	d2z_data_transform<<<gridsize2,blocksize,0,stream>>>((double2*)d_workarea,dim[0],dim[1],dim[2],dx);
	// 计算z2d
	cufftExecZ2D(plan_Z2D, (cufftDoubleComplex*)d_workarea, d_p);
	// 计算u_{n+1}
	uAsterisk2u<<<gridsize,blocksize,0,stream>>>(d_p,dim[0],dim[1],dim[2],dx,d_u[cur_u]);
	// 计算各维的速度和
	for (int d = 0; d < 3; d++) {
		CHECK_RUNTIME(cub::DeviceReduce::Sum(d_cubarea, cub_size, d_u[cur_u] + d * N, d_sum + d, N, stream));
	}
	// 减去平均值
	subtract_mean<<<gridsize,blocksize,0,stream>>>(d_u[cur_u],d_sum,N);
}

void Eulerian_Period::compute_f(){
	// 总的元素数量
	int N=dim[0]*dim[1]*dim[2];
	// d2z变换的总的复数元素
	int N2=(dim[0]/2+1)*dim[1]*dim[2];
	// 块尺寸
	int blocksize=128;
	// 网格尺寸
	int gridsize=1+(N-1)/blocksize;
	int gridsize2=1+(N2-1)/blocksize;
	// 数据的大小
	size_t size=sizeof(double)*N*3;
	// scale
	double scale=sqrt(2*selm->KB*selm->T*selm->deltaT*selm->mu/N);
	if(selm->flagStochasticDriving){
		//生成随机场
		gthm_generate<<<gridsize2,blocksize,0,stream>>>((double2*)d_workarea,d_states,dim[0],dim[1],dim[2],dx);
		// 计算z2d
		cufftExecZ2D(plan_Z2D_three, (cufftDoubleComplex*)d_workarea, d_f);
		// 进行数据缩放
		gthm_scale<<<gridsize,blocksize,0,stream>>>(d_f,scale,N);
	}else{
		//重置f为0
		CHECK_RUNTIME(cudaMemsetAsync(d_f,0,size,stream));	
	}
}

