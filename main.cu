#include "camera.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "utils.cuh"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

DEVICE Color ray_color(const Ray &r) {
  Vec3 unit_direction = unit_vector(r.direction());
  auto a = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}

__global__ void render_to_framebuffer(int **framebuffer, const int image_width,
                                      const int image_height,
                                      const Vec3 &pixel00_viewport_loc,
                                      const Vec3 &camera_center,
                                      const Vec3 &pixel_delta_u,
                                      const Vec3 &pixel_delta_v) {
  auto pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (pixel_x >= image_width || pixel_y >= image_height) {
    return;
  }

  for (int row = pixel_y; row < image_height; row += blockDim.y * gridDim.y) {
    for (int col = pixel_x; col < image_width; col += blockDim.x * gridDim.x) {
      auto pixel_center =
          pixel00_viewport_loc + col * pixel_delta_u - row * pixel_delta_v;
      auto ray_direction = pixel_center - camera_center;
      Ray ray(camera_center, ray_direction);
      Color pixel_color = ray_color(ray);

      auto pixel_index = row * image_width + col;
      write_color_to_framebuffer(framebuffer, pixel_index, pixel_color);
    }
  }
}

__global__ void initialize_camera(const double viewport_width,
                                  const double viewport_height, int image_width,
                                  int image_height) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    Camera camera;
    camera.intialize(Vec3(0, 0, 0), Vec3(viewport_width, 0, 0),
                     Vec3(0, -viewport_height, 0), image_width, image_height);
  }
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

  initialize_camera<<<1, 1>>>(viewport_width, viewport_height, image_width,
                              image_height);
  // const auto camera_center = Vec3(0, 0, 0);
  // // horizontal
  // const auto viewport_u = Vec3(viewport_width, 0, 0);
  // // vertical
  // const auto viewport_v = Vec3(0, -viewport_height, 0);
  // const auto pixel_delta_u = viewport_u / image_width;
  // const auto pixel_delta_v = viewport_v / image_height;
  // const auto viewport_upper_left = camera_center - viewport_u / 2 +
  //                                  viewport_v / 2 - Vec3(0, 0, focal_length);
  // const auto pixel00_viewport_loc =
  //     viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

  int **framebuffer;
  cudaMalloc(&framebuffer, image_width * image_height * sizeof(int *));
  for (int i = 0; i < image_width * image_height; i++) {
    cudaMalloc(&framebuffer[i], 3 * sizeof(int));
  }
  dim3 grid(grid_x, grid_y);
  dim3 block(block_x, block_y);
  std::clog << "Begin rendering" << std::endl;

  // TODO: call render_to_framebuffer of camera.

  // render_to_framebuffer<<<grid, block>>>(framebuffer, image_width, image_height,
  //                                        pixel00_viewport_loc, camera_center,
  //                                        pixel_delta_u, pixel_delta_v);
  cudaDeviceSynchronize();
  std::clog << "End rendering" << std::endl;

  std::clog << "Begin writing to file" << std::endl;
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
  for (int row = 0; row < image_height; row++) {
    for (int col = 0; col < image_width; col++) {
      auto pixel_index = row * image_width + col;
      std::cout << framebuffer[pixel_index][0] << " "
                << framebuffer[pixel_index][1] << " "
                << framebuffer[pixel_index][2] << "\n";
    }
  }
  std::clog << "End writing to file" << std::endl;

  return 0;
}
