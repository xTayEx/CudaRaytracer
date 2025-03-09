#include "camera.cuh"
#include "hittable_list.cuh"
#include "random.cuh"
#include "sphere.cuh"
#include "utils.cuh"
#include <cmath>
#include <cstdio>
#include <iostream>

__global__ void initialize_camera(Camera *camera_ptr, const double focal_length,
                                  const double viewport_width,
                                  const double viewport_height, int image_width,
                                  int image_height) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (camera_ptr) Camera();
    Vec3 camera_center = Point3(0, 0, 0);
    Vec3 viewport_u = Vec3(viewport_width, 0, 0);
    Vec3 viewport_v = Vec3(0, -viewport_height, 0);
    const auto pixel_delta_u = viewport_u / image_width;
    const auto pixel_delta_v = viewport_v / image_height;
    camera_ptr->intialize(camera_center, focal_length, viewport_u, viewport_v,
                          image_width, image_height, 100);
  }
}

__global__ void initialize_world(HittableList *world_ptr, Sphere **spheres_ptr,
                                 int spheres_count) {
  new (spheres_ptr[0]) Sphere(Point3(0, 0, -1), 0.5);
  new (spheres_ptr[1]) Sphere(Point3(0, -100.5, -1), 100);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (world_ptr) HittableList(2);
    for (int i = 0; i < spheres_count; ++i) {
      world_ptr->add(spheres_ptr[i]);
    }
  }
}

__global__ void device_object_destructor(Camera *camera_ptr,
                                         HittableList *world,
                                         Sphere **spheres_ptr,
                                         int spheres_count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    camera_ptr->~Camera();
    for (int i = 0; i < spheres_count; ++i) {
      spheres_ptr[i]->~Sphere();
    }
  }
}

void cleanup_device(Camera *camera_ptr, HittableList *world_ptr,
                    Sphere **spheres_ptr, int spheres_count, int *framebuffer) {

  device_object_destructor<<<1, 1>>>(camera_ptr, world_ptr, spheres_ptr,
                                     spheres_count);
  CHECK_CUDA(cudaFree(camera_ptr));
  CHECK_CUDA(cudaFree(framebuffer));
}

void cleanup_host(int *framebuffer_host) { delete[] framebuffer_host; }

__global__ void camera_render_launcher(Camera *camera_ptr, int *framebuffer,
                                       HittableList *world,
                                       const int image_width,
                                       const int image_height) {
  camera_ptr->render_to_framebuffer(framebuffer, world, image_width,
                                    image_height);
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

  // Do some initialization
  const double viewport_height = 2.0;
  const double viewport_width = aspect_ratio * viewport_height;

  const int grid_x = image_width / 16;
  const int grid_y = image_height / 9;
  const int block_x = std::min(grid_x / 16, 32);
  const int block_y = std::min(grid_y / 16, 32);
  const double focal_length = 1.0;

  Camera *camera_ptr;
  // TODO: Is there any neovim plugin to show the expanded macro in a floating
  // window?
  CHECK_CUDA(cudaMalloc(&camera_ptr, sizeof(Camera)));
  CHECK_CUDA(cudaDeviceSynchronize());
  initialize_camera<<<1, 1>>>(camera_ptr, focal_length, viewport_width,
                              viewport_height, image_width, image_height);

  HittableList *world_ptr;
  CHECK_CUDA(cudaMalloc(&world_ptr, sizeof(HittableList)));

  constexpr int spheres_count = 2;
  Sphere **aux_spheres_ptr = new Sphere *[spheres_count];
  for (int i = 0; i < spheres_count; ++i) {
    CHECK_CUDA(cudaMalloc(&aux_spheres_ptr[i], sizeof(Sphere)));
  }

  Sphere **spheres_ptr;
  CHECK_CUDA(cudaMalloc(&spheres_ptr, spheres_count * sizeof(Sphere *)));
  CHECK_CUDA(cudaMemcpy(spheres_ptr, aux_spheres_ptr,
                        spheres_count * sizeof(Sphere *),
                        cudaMemcpyHostToDevice));

  initialize_world<<<1, 1>>>(world_ptr, spheres_ptr, spheres_count);
  CHECK_CUDA(cudaDeviceSynchronize());

  int *framebuffer;
  CHECK_CUDA(
      cudaMalloc(&framebuffer, image_width * image_height * 3 * sizeof(int)));

  init_random(1234ULL, grid_x, grid_y, block_x, block_y);
  // end initialization

  dim3 grid(grid_x, grid_y);
  dim3 block(block_x, block_y);
  std::clog << "Begin rendering" << std::endl;

  camera_render_launcher<<<grid, block>>>(camera_ptr, framebuffer, world_ptr,
                                          image_width, image_height);

  CHECK_CUDA(cudaDeviceSynchronize());
  std::clog << "End rendering" << std::endl;

  std::clog << "Begin writing to file" << std::endl;
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  int *framebuffer_host = new int[image_width * image_height * 3];
  CHECK_CUDA(cudaMemcpy(framebuffer_host, framebuffer,
                        image_width * image_height * 3 * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());
  for (int row = 0; row < image_height; row++) {
    for (int col = 0; col < image_width; col++) {
      auto pixel_index = (row * image_width + col) * 3;
      std::cout << framebuffer_host[pixel_index + 0] << " "
                << framebuffer_host[pixel_index + 1] << " "
                << framebuffer_host[pixel_index + 2] << "\n";
    }
  }
  std::clog << "End writing to file" << std::endl;

  // TODO: free memory

  cleanup_device(camera_ptr, world_ptr, spheres_ptr, spheres_count,
                 framebuffer);
  cleanup_host(framebuffer_host);
  cleanup_random();

  return 0;
}
