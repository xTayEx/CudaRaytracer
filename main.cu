#include "camera.cuh"
#include "utils.cuh"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

__global__ void initialize_camera(Camera *camera_ptr, const double focal_length,
                                  const double viewport_width,
                                  const double viewport_height, int image_width,
                                  int image_height) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    new (camera_ptr) Camera();
    Vec3 camera_center = Point3(0, 0, 0);
    Vec3 viewport_u = Vec3(viewport_width, 0, 0);
    Vec3 viewport_v = Vec3(0, -viewport_height, 0);
    const auto pixel_delta_u = viewport_u / image_width;
    const auto pixel_delta_v = viewport_v / image_height;
    camera_ptr->intialize(camera_center, focal_length, viewport_u, viewport_v,
                          image_width, image_height);
  }
}

__global__ void cleanup(Camera *camera_ptr, int *framebuffer) {
  camera_ptr->~Camera();

  cudaFree(camera_ptr);
  cudaFree(framebuffer);
}

__global__ void camera_render_launcher(Camera *camera_ptr, int *framebuffer,
                                       const int image_width,
                                       const int image_height) {
  camera_ptr->render_to_framebuffer(framebuffer, image_width, image_height);
}

int main(int argc, char **argv) {
  int image_width = std::stoi(argv[1]);
  int image_height = std::stoi(argv[2]);
  const auto aspect_ratio = double(image_width) / image_height;
  if (std::fabs(aspect_ratio - 16.0 / 9.0) > 0.001) {
    std::cerr << "Aspect ratio other than 16:9 is not supported now"
              << std::endl;
    return -1;
  }

  const double viewport_height = 2.0;
  const double viewport_width = aspect_ratio * viewport_height;

  const int grid_x = image_width / 16;
  const int grid_y = image_height / 9;
  const int block_x = std::min(grid_x / 16, 32);
  const int block_y = std::min(grid_y / 16, 32);
  const double focal_length = 1.0;

  Camera *camera_ptr;
  // TODO: Is there any neovim plugin to show the expanded macro?
  CHECK_CUDA(cudaMalloc(&camera_ptr, sizeof(Camera)));
  CHECK_CUDA(cudaDeviceSynchronize());

  initialize_camera<<<1, 1>>>(camera_ptr, focal_length, viewport_width,
                              viewport_height, image_width, image_height);

  int *framebuffer;
  CHECK_CUDA(
      cudaMalloc(&framebuffer, image_width * image_height * 3 * sizeof(int)));

  dim3 grid(grid_x, grid_y);
  dim3 block(block_x, block_y);
  std::clog << "Begin rendering" << std::endl;

  camera_render_launcher<<<grid, block>>>(camera_ptr, framebuffer, image_width,
                                          image_height);

  cudaDeviceSynchronize();
  std::clog << "End rendering" << std::endl;

  std::clog << "Begin writing to file" << std::endl;
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  int *framebuffer_host = new int[image_width * image_height * 3];
  CHECK_CUDA(cudaMemcpy(framebuffer_host, framebuffer,
                        image_width * image_height * 3 * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < image_height; row++) {
    for (int col = 0; col < image_width; col++) {
      auto pixel_index = row * image_width + col;
      std::cout << framebuffer_host[pixel_index + 0] << " "
                << framebuffer_host[pixel_index + 1] << " "
                << framebuffer_host[pixel_index + 2] << "\n";
    }
  }
  std::clog << "End writing to file" << std::endl;

  // TODO: free memory
  cleanup<<<1, 1>>>(camera_ptr, framebuffer);

  return 0;
}
