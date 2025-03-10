#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "color.cuh"
#include "hittable.cuh"
#include "ray.cuh"
#include "utils.cuh"

enum class MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

struct MaterialDescriptor {
  MaterialType type;
  double r, g, b;
  double fuzz;
  double refraction_index;
};

class Material {
public:
  DEVICE virtual ~Material() {}
  DEVICE virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                              Color &attenuation, Ray &scattered) const {
    return false;
  }
};

class Lambertian : public Material {
public:
  DEVICE Lambertian(const Color &albedo) : albedo(albedo) {}

  DEVICE bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation,
                      Ray &scattered) const override {
    auto scatter_direction = rec.normal + random_unit_vector();

    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

private:
  Color albedo;
};

class Metal : public Material {
public:
  DEVICE Metal(const Color &albedo, double fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {};

  DEVICE bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation,
                      Ray &scattered) const override {
    auto reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
    scattered = Ray(rec.p, reflected);
    attenuation = albedo;

    return (dot(scattered.direction(), rec.normal) > 0);
  }

private:
  Color albedo;
  double fuzz;
};

class Dielectric : public Material {
public:
  DEVICE Dielectric(double refraction_index)
      : refraction_index(refraction_index) {}

  DEVICE bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation,
                      Ray &scattered) const override {
    attenuation = Color(1, 1, 1);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;
    Vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    Vec3 direction;

    if (cannot_refract) {
      direction = reflect(unit_direction, rec.normal);
    } else {
      direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = Ray(rec.p, direction);
    return true;
  }

private:
  double refraction_index;
};

#endif
