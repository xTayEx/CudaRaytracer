#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
__global__ void write_to_framebuffer(int **framebuffer, const int image_width,
                                     const int image_height) {
  auto pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (pixel_x >= image_width || pixel_y >= image_height) {
    return;
  }

  for (int row = pixel_y; row < image_height; row += blockDim.y * gridDim.y) {
    for (int col = pixel_x; col < image_width; col += blockDim.x * gridDim.x) {
      auto r = double(col) / (image_width - 1);
      auto g = double(row) / (image_height - 1);
      auto b = 0.0;

      int ir = int(255.999 * r);
      int ig = int(255.999 * g);
      int ib = int(255.999 * b);
      auto pixel_index = row * image_width + col;
      framebuffer[pixel_index][0] = ir;
      framebuffer[pixel_index][1] = ig;
      framebuffer[pixel_index][2] = ib;
    }
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

  const int grid_x = image_width / 16;
  const int grid_y = image_height / 9;
  const int block_x = std::min(grid_x / 16, 32);
  const int block_y = std::min(grid_y / 16, 32);
  int **framebuffer;
  cudaMallocManaged(&framebuffer, image_width * image_height * sizeof(int *));
  for (int i = 0; i < image_width * image_height; i++) {
    cudaMallocManaged(&framebuffer[i], 3 * sizeof(int));
  }
  dim3 grid(grid_x, grid_y);
  dim3 block(block_x, block_y);
  write_to_framebuffer<<<grid, block>>>(framebuffer, image_width, image_height);

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
  for (int row = 0; row < image_height; row++) {
    for (int col = 0; col < image_width; col++) {
      auto pixel_index = row * image_width + col;
      std::cout << framebuffer[pixel_index][0] << " "
                << framebuffer[pixel_index][1] << " "
                << framebuffer[pixel_index][2] << "\n";
    }
  }

  return 0;
}
