#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "color.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "random.cuh"
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
  DEVICE ~Camera(){};

  DEVICE void intialize(Vec3 camera_center, double focal_length,
                        Vec3 viewport_u, Vec3 viewport_v, int image_width,
                        int image_height, int samples_per_pixel) {
    this->camera_center = camera_center;
    this->focal_length = focal_length;
    this->viewport_u = viewport_u;
    this->viewport_v = viewport_v;
    this->image_width = image_width;
    this->image_height = image_height;
    this->samples_per_pixel = samples_per_pixel;
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
        Color pixel_color(0.0, 0.0, 0.0);
        for (int sample_idx = 0; sample_idx < samples_per_pixel; ++sample_idx) {
          auto ray_around = get_ray_around_pixel(row, col);
          auto color_around = ray_color(ray_around, world, 0);
          pixel_color += color_around;
        }

        auto pixel_index = (row * image_width + col) * 3;
        write_color_to_framebuffer(framebuffer, pixel_index,
                                   pixel_color * (1.0 / samples_per_pixel));
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
  const int max_hit_depth = 10;
  double focal_length;
  int samples_per_pixel;

  DEVICE Ray get_ray_around_pixel(int row, int col) {
    auto offset = sample_square();
    auto pixel_sample = pixel00_viewport_loc +
                        ((col + offset.x()) * pixel_delta_u) +
                        ((row + offset.y()) * pixel_delta_v);
    auto ray_origin = camera_center;
    auto ray_direction = pixel_sample - ray_origin;
    Ray r(ray_origin, ray_direction);

    return r;
  }

  DEVICE Vec3 sample_square() const {
    return Vec3(random_double() - 0.5, random_double() - 0.5, 0.0);
  }

  DEVICE Color ray_color(const Ray &r, HittableList *world, int cur_depth) {
    if (cur_depth >= max_hit_depth) {
      return Color(0.0, 0.0, 0.0);
    }
    HitRecord rec;
    if (world->hit(r, 0.001, inf, rec)) {
      Ray scattered;
      Color attenuation;
      if (rec.mat->scatter(r, rec, attenuation, scattered)) {
        return attenuation * ray_color(scattered, world, cur_depth + 1);
      }
      return Color(0.0, 0.0, 0.0);
    }

    Vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
  }
};

#endif
