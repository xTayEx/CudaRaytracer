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

  DEVICE void intialize(Point3 lookfrom,
                        Point3 lookat,
                        Vec3 vup,
                        int image_width,
                        int image_height,
                        int samples_per_pixel,
                        double vfov,
                        double defocus_angle,
                        double focus_dist) {
    this->camera_center = lookfrom;

    this->lookfrom = lookfrom;
    this->lookat = lookat;
    this->vup = vup;

    this->defocus_angle = defocus_angle;
    this->focus_dist = focus_dist;

    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta / 2);
    auto viewport_height = 2.0 * h * this->focus_dist;
    auto viewport_width =
        viewport_height * (double(image_width) / image_height);
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    this->viewport_u = viewport_width * u;
    this->viewport_v = viewport_height * (-v);
    this->image_width = image_width;
    this->image_height = image_height;
    this->samples_per_pixel = samples_per_pixel;
    this->vfov = vfov;
    pixel_delta_u = this->viewport_u / image_width;
    pixel_delta_v = this->viewport_v / image_height;
    const auto viewport_upper_left =
        this->camera_center - this->viewport_u / 2 - this->viewport_v / 2 -
        this->focus_dist * w;
    pixel00_viewport_loc =
        viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    auto defocus_radius =
        this->focus_dist * std::tan(degrees_to_radians(defocus_angle) / 2);
    this->defocus_disk_u = defocus_radius * u;
    this->defocus_disk_v = defocus_radius * v;
  }

  DEVICE void render_to_framebuffer(int *framebuffer,
                                    HittableList *world,
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
        write_color_to_framebuffer(
            framebuffer, pixel_index, pixel_color * (1.0 / samples_per_pixel));
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

  Point3 lookfrom;
  Point3 lookat;
  Vec3 vup;
  Vec3 u, v, w;
  Vec3 defocus_disk_u;
  Vec3 defocus_disk_v;

  double defocus_angle;
  double focus_dist;

  int image_width;
  int image_height;
  double vfov = 90.0;
  const int max_hit_depth = 10;
  int samples_per_pixel;

  DEVICE Ray get_ray_around_pixel(int row, int col) {
    auto offset = sample_square();
    auto pixel_sample = pixel00_viewport_loc +
                        ((col + offset.x()) * pixel_delta_u) +
                        ((row + offset.y()) * pixel_delta_v);
    auto ray_origin = (this->defocus_angle <= 0 ? this->camera_center
                                                : defocus_disk_sample());
    auto ray_direction = pixel_sample - ray_origin;
    Ray r(ray_origin, ray_direction);

    return r;
  }

  DEVICE Point3 defocus_disk_sample() const {
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk();
    return this->camera_center + (p[0] * defocus_disk_u) +
           (p[1] * defocus_disk_v);
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
