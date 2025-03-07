#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "utils.cuh"
#include "vec3.cuh"

class Camera {
public:
  Camera() = default;

  DEVICE void intialize(Vec3 center, Vec3 viewport_u, Vec3 viewport_v,
                        int image_width, int image_height) {
    this->center = center;
    this->viewport_u = viewport_u;
    this->viewport_v = viewport_v;
    this->image_width = image_width;
    this->image_height = image_height;
    const auto pixel_delta_u = viewport_u / image_width;
    const auto pixel_delta_v = viewport_v / image_height;
    const auto viewport_upper_left = center - viewport_u / 2 + viewport_v / 2;
    const auto pixel00_viewport_loc =
        viewport_upper_left + pixel_delta_u / 2 - pixel_delta_v / 2;
  }

  DEVICE void render_to_framebuffer() {};

private:
  Vec3 center;
  Vec3 viewport_u;
  Vec3 viewport_v;
  int image_width;
  int image_height;
};

#endif
