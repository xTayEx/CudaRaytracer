#ifndef COLOR_CUH
#define COLOR_CUH

#include "utils.cuh"
#include "vec3.cuh"
#include <cmath>
#include <cstdio>

using Color = Vec3;

DEVICE double linear_to_gamma2(double linear_component) {
  if (linear_component > 0.0) {
    return std::sqrt(linear_component);
  }

  return 0;
}

DEVICE void write_color_to_framebuffer(int *framebuffer, int pixel_index,
                                       Color pixel_color) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  r = linear_to_gamma2(r);
  g = linear_to_gamma2(g);
  b = linear_to_gamma2(b);

  int ir = static_cast<int>(clamp(std::ceil(255 * r), 0.0, 255.0));
  int ig = static_cast<int>(clamp(std::ceil(255 * g), 0.0, 255.0));
  int ib = static_cast<int>(clamp(std::ceil(255 * b), 0, 255));

  framebuffer[pixel_index + 0] = ir;
  framebuffer[pixel_index + 1] = ig;
  framebuffer[pixel_index + 2] = ib;
}

#endif
