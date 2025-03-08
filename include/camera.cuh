#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "color.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "ray.cuh"
#include "utils.cuh"
#include "vec3.cuh"

DEVICE double hit_sphere(const Point3 &center, double radius, const Ray &r) {
  Vec3 oc = center - r.origin();
  auto a = r.direction().length_squared();
  auto h = dot(oc, r.direction());
  auto c = oc.length_squared() - radius * radius;
  auto discriminant = h * h - a * c;

  if (discriminant < 0) {
    return -1.0;
  } else {
    return (h - std::sqrt(discriminant)) / a;
  }
}

class Camera {
public:
  DEVICE Camera() {}; 
  DEVICE ~Camera() {};

  DEVICE void intialize(Vec3 camera_center, double focal_length,
                        Vec3 viewport_u, Vec3 viewport_v, int image_width,
                        int image_height) {
    this->camera_center = camera_center;
    this->focal_length = focal_length;
    this->viewport_u = viewport_u;
    this->viewport_v = viewport_v;
    this->image_width = image_width;
    this->image_height = image_height;
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;
    const auto viewport_upper_left = camera_center - viewport_u / 2 -
                                     viewport_v / 2 - Vec3(0, 0, focal_length);
    pixel00_viewport_loc =
        viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
  }

  DEVICE void render_to_framebuffer(int *framebuffer, HittableList *world,
                                    const int image_width,
                                    const int image_height) {

    auto pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_x >= image_width || pixel_y >= image_height) {
      return;
    }

    for (int row = pixel_y; row < image_height; row += blockDim.y * gridDim.y) {
      for (int col = pixel_x; col < image_width;
           col += blockDim.x * gridDim.x) {
        auto pixel_center =
            pixel00_viewport_loc + col * pixel_delta_u + row * pixel_delta_v;
        auto ray_direction = pixel_center - camera_center;
        Ray ray(camera_center, ray_direction);
        Color pixel_color = ray_color(ray, world);

        auto pixel_index = (row * image_width + col) * 3;
        write_color_to_framebuffer(framebuffer, pixel_index, pixel_color);
      }
    }
  };

private:
  Vec3 camera_center;
  Vec3 viewport_u;
  Vec3 viewport_v;
  Vec3 pixel00_viewport_loc;
  Vec3 pixel_delta_u;
  Vec3 pixel_delta_v;
  int image_width;
  int image_height;
  double focal_length;

  DEVICE Color ray_color(const Ray &r, HittableList *world) {
    HitRecord rec;
    if (world->hit(r, 0, inf, rec)) {
      return 0.5 * (rec.normal + Color(1.0, 1.0, 1.0));
    }

    Vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
  }
};

#endif
