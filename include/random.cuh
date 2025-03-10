#ifndef RANDOM_GENERATOR_CUH
#define RANDOM_GENERATOR_CUH

#include "utils.cuh"
#include <curand_kernel.h>
#include <random>

DEVICE curandState *states = nullptr;
__global__ void init_curand_state_kernel(unsigned long long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &states[idx]);
}
void init_random(
    unsigned long long seed, int grid_x, int grid_y, int block_x, int block_y) {
  curandState *d_states;
  CHECK_CUDA(cudaMalloc(
      &d_states, grid_x * grid_y * block_x * block_y * sizeof(curandState)));
  CHECK_CUDA(cudaMemcpyToSymbol(states, &d_states, sizeof(curandState *)));
  init_curand_state_kernel<<<grid_x * grid_y, block_x * block_y>>>(seed);
}

void cleanup_random() {
  curandState *d_states;
  CHECK_CUDA(cudaMemcpyFromSymbol(&d_states, states, sizeof(curandState *)));
  CHECK_CUDA(cudaFree(d_states));
}
DEVICE double random_double() {
  int idx = (blockDim.x * gridDim.x) * threadIdx.y + threadIdx.x;
  return curand_uniform(&states[idx]);
}
DEVICE double random_double(double mini, double maxi) {
  return mini + (maxi - mini) * random_double();
}

double random_double_host() {
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  static std::mt19937 gen;
  return dist(gen);
}

double random_double_host(double mini, double maxi) {
  return mini + (maxi - mini) * random_double_host();
}

#endif
