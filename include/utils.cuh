#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>

#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("Error: %s\n", cudaGetErrorString(x));                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

HOST_DEVICE double clamp(double val, double mini, double maxi) {
  return val < mini ? mini : (val > maxi ? maxi : val);
}

__global__ void print_dev_memory(int *mem, size_t size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (size_t i = 0; i + 2 < size; i++) {
      printf("mem[%ld]: %d, mem[%ld]: %d, mem[%ld]: %d\n", i, mem[i], i + 1, mem[i + 1], i + 2, mem[i + 2]);
    }
  }
}

#endif
