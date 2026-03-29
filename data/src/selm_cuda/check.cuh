#ifndef CHECK_H
#define CHECK_H

#include <stdio.h>

#define CHECK_RUNTIME(x)                                          \
{ auto const __err = x;                                               \
  if( __err != cudaSuccess )                                          \
  { printf("Runtime Error: line: %d    error: %s    file: %s\n", __LINE__, cudaGetErrorString(__err), __FILE__); exit(-1); } \
};

#define CHECK_KERNEL(x)                                          \
{                                                                       \
  x;                                                                    \
  auto const __err = cudaGetLastError();                                \
  if( __err != cudaSuccess )                                            \
  { printf("Kernel Error: %d %s\n", __LINE__, cudaGetErrorString(__err)); exit(-1); } \
};

#define CHECK_TENSOR(x)                                                   \
{ auto const __err = x;                                                   \
  if( __err != CUTENSOR_STATUS_SUCCESS )                                  \
  { printf("cuTensor Error: %d %s\n", __LINE__, cutensorGetErrorString(__err)); exit(-1); } \
};

#define CHECK_CUFFT(x)                                           \
{                                                                       \
  auto const __err = x;                                                 \
  if( __err != CUFFT_SUCCESS )                                          \
  { printf("cuFFT Error: %d Error code: %d\n", __LINE__, __err); exit(-1); } \
};

#define CHECK_CURAND(x)                                          \
{                                                                       \
  auto const __err = x;                                                 \
  if( __err != CURAND_STATUS_SUCCESS )                                  \
  { printf("cuRAND Error: %d Error code: %d\n", __LINE__, __err); exit(-1); } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, nullptr);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, nullptr);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, nullptr);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return static_cast<float>(time * 1e-3);
    }

private:
    cudaEvent_t start_, stop_;
};

#endif
