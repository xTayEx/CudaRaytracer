#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "color.cuh"
#include "hittable.cuh"
#include "ray.cuh"
#include "utils.cuh"

enum class MaterialType { LAMBERTIAN, METAL };

struct MaterialDescriptor {
  MaterialType type;
  double r, g, b;
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
  DEVICE Metal(const Color &albedo) : albedo(albedo) {};

  DEVICE bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation,
                      Ray &scattered) const override {
    auto reflected = reflect(r_in.direction(), rec.normal);
    scattered = Ray(rec.p, reflected);
    attenuation = albedo;

    return true;
  }

private:
  Color albedo;
};

#endif
