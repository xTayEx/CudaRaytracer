#ifndef COLOR_CUH
#define COLOR_CUH

#include "utils.cuh"
#include "vec3.cuh"
#include <cmath>
#include <cstdio>

using Color = Vec3;

DEVICE void write_color_to_framebuffer(int *framebuffer, int pixel_index,
                                       Color pixel_color) {
  int ir =
      static_cast<int>(clamp(std::ceil(255 * pixel_color.x()), 0.0, 255.0));
  int ig =
      static_cast<int>(clamp(std::ceil(255 * pixel_color.y()), 0.0, 255.0));
  int ib = static_cast<int>(clamp(std::ceil(255 * pixel_color.z()), 0, 255));

  framebuffer[pixel_index + 0] = ir;
  framebuffer[pixel_index + 1] = ig;
  framebuffer[pixel_index + 2] = ib;
}

#endif
