#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>
#include <curand_kernel.h>

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

#define CHECK_CURAND(x)                                                        \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      printf("Error: %d\n", x);                                                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

constexpr double inf = INFINITY;
constexpr double pi = 3.1415926535897932385;

HOST_DEVICE double clamp(double val, double mini, double maxi) {
  return val < mini ? mini : (val > maxi ? maxi : val);
}

HOST_DEVICE double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}

__global__ void print_dev_memory(int *mem, size_t size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (size_t i = 0; i + 2 < size; i++) {
      printf("mem[%ld]: %d, mem[%ld]: %d, mem[%ld]: %d\n", i, mem[i], i + 1,
             mem[i + 1], i + 2, mem[i + 2]);
    }
  }
}

#endif
