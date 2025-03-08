#ifndef RANDOM_GENERATOR_CUH
#define RANDOM_GENERATOR_CUH

#include "utils.cuh"
#include <curand_kernel.h>

DEVICE curandState *states = nullptr;
__global__ void init_curand_state_kernel(unsigned long long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &states[idx]);
}
void init_random(unsigned long long seed, int grid_x, int grid_y, int block_x,
                 int block_y) {
  init_curand_state_kernel<<<grid_x * grid_y, block_x * block_y>>>(seed);
}
DEVICE double random_double() {
  int idx = (blockDim.x * gridDim.x) * threadIdx.y + threadIdx.x;
  return curand_uniform(&states[idx]);
}
DEVICE double random_double(double mini, double maxi) {
  return mini + (maxi - mini) * random_double();
}

#endif
